# model.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
from contextlib import nullcontext
from peft.tuners.tuners_utils import BaseTuner
import wandb

from .layer import NonlinearLoraLinear

class NonlinearLoraModel(BaseTuner):
    prefix = "nlora_"                 # unique prefix for state dict filtering
    tuner_layer_cls = NonlinearLoraLinear

    def _prepare_adapter_config(self, peft_config, model_config):
        self.consolidation_updates = 0
        if peft_config.target_modules is None:
            raise ValueError("NonlinearLoraConfig.target_modules must be set.")
        return peft_config

    def _create_and_replace(self, config, adapter_name, target, target_name, parent, **kwargs):
        # Only wrap Linear for now (extend: Conv1D, Embedding, etc.)
        if isinstance(target, nn.Linear):
            new_module = NonlinearLoraLinear(target)
            new_module.update_layer(
                adapter_name=adapter_name,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                activation_fn=config.activation_fn,
            )
            setattr(parent, target_name, new_module)
    
    def _get_single_active_adapter(self) -> str:
        a = self.active_adapter
        if isinstance(a, list):
            if len(a) != 1:
                raise ValueError(f"Consolidation supports exactly 1 active adapter, got {a}")
            return a[0]
        return a
    
    def log_activation_spectrum(self, xxt: torch.Tensor, step: int, accelerator):
        """Log eigenvalue spectrum of gram matrix for diagnostic purposes."""
        if not accelerator.is_main_process:
            return
        
        with torch.no_grad():
            # xxt is [d, d], eigvalsh returns ascending sorted eigenvalues
            eigvals = torch.linalg.eigvalsh(xxt.float())  # float32 for stability
            eigvals = eigvals.flip(0)  # descending
            
            total = eigvals.sum()
            cumvar = torch.cumsum(eigvals, dim=0) / total
            
            # how many eigenvalues to capture 50%, 90%, 95%, 99%
            thresholds = [0.5, 0.9, 0.95, 0.99]
            rank_at = {}
            for t in thresholds:
                rank_at[t] = (cumvar < t).sum().item() + 1
            
            # ratio of top-r mean to trace/d for your current normalizer comparison
            d = eigvals.shape[0]
            r = 8  # your LoRA rank, or pass it in
            top_r_mean = eigvals[:r].mean().item()
            trace_over_d = eigvals.sum().item() / d
            
            wandb.log({
                "spectrum/rank_at_50pct": rank_at[0.5],
                "spectrum/rank_at_90pct": rank_at[0.9],
                "spectrum/rank_at_95pct": rank_at[0.95],
                "spectrum/rank_at_99pct": rank_at[0.99],
                "spectrum/top_r_mean": top_r_mean,
                "spectrum/trace_over_d": trace_over_d,
                "spectrum/ratio_top_r_to_trace_d": top_r_mean / (trace_over_d + 1e-8),
                "step": step,
            })

    @torch.no_grad()
    def consolidate(
        self,
        dataloader,
        *,
        adapter_name: str | None = None,
        lambda_: float | None = None,
        lr: float | None = None,
        offload_cpu: bool | None = None,
        accum_dtype: torch.dtype | None = None,
        scale_lambda_by_trace: bool | None = None,
        max_batches: int | None = 1,
        inplace_disable_adapter: bool = False,
        update_frequency: int | None = None,
        zeroshift: bool | None = None,
        global_step: int = 0,
        accelerator = None,
    ):
        """
        Data-dependent consolidation: fit ΔW per wrapped layer using ridge regression on calibration inputs,
        where targets are the adapter's current contribution.

        Call: peft_model.base_model.consolidate(calib_loader)
        """
        # TODO: support multiple adapters at once (currently requires separate calls or manual looping)
        if adapter_name is None:
            adapter_name = self._get_single_active_adapter()
        
        cfg = self.peft_config[adapter_name]

        if lambda_ is None:
            lambda_ = getattr(cfg, "consolidate_lambda", 1e-3)
        if lr is None:
            lr = getattr(cfg, "consolidate_lr", 1.0)
        if offload_cpu is None:
            offload_cpu = getattr(cfg, "consolidate_offload_cpu", True)
        if scale_lambda_by_trace is None:
            scale_lambda_by_trace = getattr(cfg, "consolidate_scale_lambda_by_trace", True)
        if max_batches is None:
            max_batches = getattr(cfg, "consolidate_batches", 1)  # allow None = all

        if accum_dtype is None:
            dtype_str = getattr(cfg, "consolidate_dtype", "float32")
            accum_dtype = torch.float64 if dtype_str == "float64" else torch.float32

        if update_frequency is None:
            update_frequency = getattr(cfg, "consolidate_layer_update_frequency", 1.0) # which layers to update 1 update all layers 2, update every other layer etc.
        
        if zeroshift is None:
            zeroshift = getattr(cfg, "consolidate_zero_shift", False)

        assert lr is not None, "LR must be specified for consolidation, either via argument or config"

        if lr >= 1e-8:
            layer_states: dict[NonlinearLoraLinear, dict] = {}
            hooks = []

            def make_hook(layer: NonlinearLoraLinear):
                def hook(module, inputs, output):
                    x = inputs[0] # (batch_size, seq_len, in_features)
                    layer.accumulate_consolidation_stats(
                        x=x,
                        adapter_name=adapter_name,
                        state=layer_states[layer],
                        accum_dtype=accum_dtype,
                        lambda_=lambda_,
                        scale_lambda_by_trace=scale_lambda_by_trace,
                    )
                return hook

            # register hooks + init states
            layers = []
            update_count = self.consolidation_updates % update_frequency
            for m in self.model.modules():
                if isinstance(m, NonlinearLoraLinear):
                    if (update_count % update_frequency) == 0:
                        layers.append(m)
                        layer_states[m] = {}
                        hooks.append(m.register_forward_hook(make_hook(m)))
                    update_count += 1
            # accumulate stats
            self.model.eval()
            dev = next(self.model.parameters()).device
            dev = next(self.model.parameters()).device

            if dev.type == 'cuda':
                ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif dev.type == 'mps':
                ctx = torch.autocast(device_type='mps', dtype=torch.bfloat16)
            else:
                ctx = nullcontext()

            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                if isinstance(batch, dict):
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    with ctx:
                       _ = self.model(**batch)
                else:
                    with ctx:
                       _ = self.model(**batch)

            for h in hooks:
                h.remove()

            if dist.is_available() and dist.is_initialized():
                for layer in layers:
                    state = layer_states[layer]
                    if layer.is_linear.get(adapter_name, False):
                        dist.all_reduce(state["xxt"], op=dist.ReduceOp.SUM)
                        dist.all_reduce(state["xzt"], op=dist.ReduceOp.SUM)
                        dist.all_reduce(state["zzt"], op=dist.ReduceOp.SUM)
                        # zeroshift stats - sum the gram matrix
                        if "zx_dW" in state:
                            dist.all_reduce(state["zx_dW"], op=dist.ReduceOp.SUM)

            if accelerator is not None:
                first_layer = next(iter(layer_states.keys()))
                if not first_layer.is_linear.get(adapter_name, False):
                    xxt = layer_states[first_layer]["xxt"]
                    self.log_activation_spectrum(xxt, step=global_step, accelerator=accelerator)

            for layer in layers:
                layer.solve_and_merge(
                    state=layer_states[layer],
                    lr_=lr,
                    lambda_w=lambda_,
                    scale_by_lambda_=scale_lambda_by_trace,
                    adapter_name=adapter_name,
                    inplace_disable_adapter=inplace_disable_adapter,
                    zeroshift=zeroshift,
                )

            self.consolidation_updates += 1

            return layer_states

