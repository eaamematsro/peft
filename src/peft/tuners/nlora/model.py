# model.py
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from peft.tuners.tuners_utils import BaseTuner
import wandb

from .layer import NonlinearLoraLinear
from .consolidate import consolidate as _consolidate

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

def consolidate(self, dataloader, *, lr=None, **kwargs):
    if lr is None:
        lr = getattr(self.peft_config[self._get_single_active_adapter()], 
                     "consolidate_lr", 1.0)
    return _consolidate(
        model=self.model,
        dataloader=dataloader,
        adapter_name=self._get_single_active_adapter(),
        lr=lr,
        **kwargs,
    )
