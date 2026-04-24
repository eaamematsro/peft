from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from typing import Optional

def _is_nlora_layer(module) -> bool:
    return type(module).__name__ == 'NonlinearLoraLinear'

def _is_lora_layer(module) -> bool:
    return hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'scaling')

def _is_consolidatable(module) -> bool:
    return _is_nlora_layer(module) or _is_lora_layer(module)

def _get_V(module, adapter_name: str) -> torch.Tensor:
    """Get V (input projection) weight. Shape: [r, d_in]"""
    if _is_nlora_layer(module):
        return module.nlora_V[adapter_name].weight
    return module.lora_A[adapter_name].weight

def _get_U(module, adapter_name: str) -> torch.Tensor:
    """Get U (output projection) weight. Shape: [d_out, r]"""
    if _is_nlora_layer(module):
        return module.nlora_U[adapter_name].weight
    return module.lora_B[adapter_name].weight

def _get_scaling(module, adapter_name: str) -> float:
    return module.scaling[adapter_name]

def _get_base_weight(module) -> torch.Tensor:
    """Get base layer weight. Shape: [d_out, d_in]"""
    if _is_nlora_layer(module):
        return module.base_layer.weight
    return module.get_base_layer().weight

def _get_in_features(module) -> int:
    if _is_nlora_layer(module):
        return module.in_features
    return module.in_features

def _get_out_features(module) -> int:
    if _is_nlora_layer(module):
        return module.out_features
    return module.out_features

def _get_phi(module, adapter_name: str):
    """Get activation function. Returns nn.Identity for standard LoRA."""
    if _is_nlora_layer(module):
        return module.phi[adapter_name]
    return nn.Identity()

def _adapter_exists(module, adapter_name: str) -> bool:
    if _is_nlora_layer(module):
        return adapter_name in module.nlora_V
    return adapter_name in module.lora_A

def _reset_U(module, adapter_name: str) -> None:
    """Reset U/lora_B to zeros."""
    if _is_nlora_layer(module):
        nn.init.zeros_(module.nlora_U[adapter_name].weight)
    else:
        nn.init.zeros_(module.lora_B[adapter_name].weight)

def _reset_V(module, adapter_name: str) -> None:
    """Reset V/lora_A to random."""
    dim = _get_V(module, adapter_name).shape[1]
    if _is_nlora_layer(module):
        nn.init.normal_(module.nlora_V[adapter_name].weight, std=1/math.sqrt(dim))
    else:
        nn.init.normal_(module.lora_A[adapter_name].weight, std=1/math.sqrt(dim))

def _accumulate_stats(
    module,
    x: torch.Tensor,
    adapter_name: str,
    state: dict,
    accum_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> None:
    """
    Accumulate consolidation statistics for any adapter layer.
    Computes xxt, xzt, zzt, zx_dW needed for the ridge regression solve.
    """
    if x.dim() == 3:
        x2 = x.reshape(-1, x.size(-1))
    else:
        x2 = x

    x2 = x2.to(dtype=accum_dtype)

    # compute adapter delta: scaling * phi(x @ V^T) @ U^T
    V       = _get_V(module, adapter_name)
    U       = _get_U(module, adapter_name)
    phi     = _get_phi(module, adapter_name)
    scaling = _get_scaling(module, adapter_name)

    with torch.no_grad():
        z     = phi(x2.to(V.dtype) @ V.T)               # [N, r]
        delta = (z @ U.T * scaling)                      # [N, d_out]

    z     = z.to(accum_dtype)
    delta = delta.to(accum_dtype)
    dev   = x2.device

    d = x2.size(1)
    m = delta.size(1)

    if "xxt" not in state:
        state["xxt"]       = torch.zeros((d, d), device=dev, dtype=accum_dtype)
        state["xzt"]       = torch.zeros((d, m), device=dev, dtype=accum_dtype)
        state["zzt"]       = torch.zeros((z.size(1), z.size(1)), device=dev, dtype=accum_dtype)
        state["zx_dW"]     = torch.zeros((z.size(1), d), device=dev, dtype=accum_dtype)
        state["count"]     = 0
        state["batch_count"] = 0

    state["xxt"].add_(x2.T @ x2)
    state["xzt"].add_(x2.T @ delta)
    state["zzt"].add_(z.T @ z)
    state["zx_dW"].add_(z.T @ x2)
    state["count"]      += x2.size(0)
    state["batch_count"] += 1


def _solve_dW(
    state: dict,
    lambda_: float,
    lambda_schedule: Optional[str],
    rank: int,
    **kwargs,
) -> torch.Tensor:
    """Solve ridge regression for dW. Returns dW of shape [d_in, d_out]."""
    samples = state["count"]
    xxt     = state["xxt"] / samples
    xzt     = state["xzt"] / samples
    d       = xxt.size(0)

    I = torch.eye(d, device=xxt.device, dtype=xxt.dtype)

    if lambda_schedule is None or lambda_schedule == "mean_eigenvalue":
        lam = lambda_ * (torch.trace(xxt) / d).clamp_min(1e-6)
    elif lambda_schedule == "max_eigenvalue":
        x = torch.randn((d, 1), device=xxt.device, dtype=xxt.dtype)
        for _ in range(10):
            x = xxt @ x
            x = x / x.norm()
        max_eig = (x.T @ xxt @ x) / (x.T @ x)
        lam = lambda_ * max_eig.item()
    elif lambda_schedule == "rank_adaptive":
        lam = lambda_ * (torch.trace(xxt) * rank / d).clamp_min(1e-6)
    elif lambda_schedule == "sample_adaptive":
        lam = lambda_ * (torch.trace(xxt) / (d * math.sqrt(state["count"]))).clamp_min(1e-6)
    elif lambda_schedule == "rank_and_sample_adaptive":
        lam = lambda_ * (torch.trace(xxt) * rank / (d * math.sqrt(state["count"]))).clamp_min(1e-6)
    else:
        lam = lambda_

    A  = xxt + lam * I
    dW = torch.linalg.solve(A, xzt)   # [d_in, d_out]
    return dW


def consolidate(
    model: nn.Module,
    dataloader,
    adapter_name: str = "default",
    lr: float = 1.0,
    lambda_: float = 1e-3,
    max_batches: int = 8,
    scale_lambda_by_trace: bool = True,
    reset_U: bool = False,
    reset_V: bool = False,
    lambda_schedule: Optional[str] = None,
    accum_dtype: torch.dtype = torch.float32,
    update_frequency: int = 1,
    accelerator=None,
    global_step: int = 0,
) -> dict:
    """
    Standalone consolidation function. Works with any model containing
    NLoRA (NonlinearLoraLinear) or standard HuggingFace PEFT LoRA layers.

    Merges adapter knowledge into base weights via ridge regression,
    optionally resets U/lora_B after merging.

    Args:
        model:               Any nn.Module with consolidatable adapter layers
        dataloader:          Calibration dataloader for accumulating statistics
        adapter_name:        Name of the adapter to consolidate
        lr:                  Consolidation learning rate (scales dW update)
        lambda_:             Ridge regression regularization strength
        max_batches:         Number of batches to use for statistics accumulation
        scale_lambda_by_trace: Scale lambda by trace of XXT for normalization
        reset_U:             Reset U/lora_B to zeros after merging
        reset_V:             Reset V/lora_A to random after merging
        lambda_schedule:     Lambda scaling schedule ('mean_eigenvalue', 'rank_adaptive', etc.)
        accum_dtype:         Dtype for accumulation (float32 recommended for stability)
        update_frequency:    Update every Nth layer (1 = all layers)
        accelerator:         Accelerate accelerator for distributed training
        global_step:         Current training step (for logging)

    Returns:
        dict: layer_states containing accumulated statistics per layer
    """
    if lr < 1e-8:
        return {}

    dev = next(model.parameters()).device

    if dev.type == 'cuda':
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif dev.type == 'mps':
        ctx = torch.autocast(device_type='mps', dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    # collect consolidatable layers
    layer_states: dict = {}
    hooks = []
    update_count = 0

    def make_hook(layer):
        def hook(module, inputs, output):
            if not _adapter_exists(module, adapter_name):
                return
            x = inputs[0]
            _accumulate_stats(
                module=module,
                x=x,
                adapter_name=adapter_name,
                state=layer_states[layer],
                accum_dtype=accum_dtype,
                lambda_=lambda_,
                scale_lambda_by_trace=scale_lambda_by_trace,
            )
        return hook

    for module in model.modules():
        if _is_consolidatable(module) and _adapter_exists(module, adapter_name):
            if (update_count % update_frequency) == 0:
                layer_states[module] = {}
                hooks.append(module.register_forward_hook(make_hook(module)))
            update_count += 1

    if not layer_states:
        print("Warning: no consolidatable layers found. "
              "Make sure the model has NLoRA or LoRA adapter layers.")
        return {}

    # accumulate statistics
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            if isinstance(batch, dict):
                batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            with ctx:
                model(**batch)

    for h in hooks:
        h.remove()

    # all-reduce statistics across processes if distributed
    if dist.is_available() and dist.is_initialized():
        for module, state in layer_states.items():
            for key in ("xxt", "xzt", "zzt", "zx_dW", "count", "batch_count"):
                if key in state:
                    dist.all_reduce(state[key], op=dist.ReduceOp.SUM)

    # before/after loss for diagnostics
    test_batch = next(iter(dataloader))
    if isinstance(test_batch, dict):
        test_batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                      for k, v in test_batch.items()}

    with torch.no_grad():
        with ctx:
            loss_before = model(**test_batch).loss.item()

    # solve and merge per layer
    for module, state in layer_states.items():
        rank   = _get_V(module, adapter_name).size(0)
        dW     = _solve_dW(
            state=state,
            lambda_=lambda_,
            adapter_name=adapter_name,
            lambda_schedule=lambda_schedule,
            rank=rank,
        )

        base_w = _get_base_weight(module)
        dW     = dW.to(base_w.device, dtype=base_w.dtype)
        base_w.data.add_(dW.T * lr)

        if reset_U:
            _reset_U(module, adapter_name)
        if reset_V:
            _reset_V(module, adapter_name)

    with torch.no_grad():
        with ctx:
            loss_after = model(**test_batch).loss.item()

    print(f"[consolidate] step={global_step} | "
          f"loss_before={loss_before:.6f} | "
          f"loss_after={loss_after:.6f} | "
          f"diff={loss_after - loss_before:.6f}")

    model.train()
    return layer_states
