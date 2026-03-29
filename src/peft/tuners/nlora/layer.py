# layer.py
import math
import pdb
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

def make_phi(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == 'linear':
        return nn.Identity()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation_fn={name}")

# TODO: Implement zero-shift consolidation


class NonlinearLoraLinear(nn.Module, BaseTunerLayer):
    """
    y = base(x) + scaling * ( phi(x @ V) @ U )
    V: [in, r], U: [r, out]
    """
    adapter_layer_names = ("nlora_V", "nlora_U")   # what PEFT saves
    other_param_names = ("r", "nlora_alpha", "scaling")

    def __init__(self, base_layer: nn.Linear):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("NonlinearLoraLinear supports nn.Linear only (extend as needed).")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = {}
        self.nlora_alpha = {}
        self.scaling = {}

        self.nlora_dropout = nn.ModuleDict()
        self.nlora_V = nn.ModuleDict()
        self.nlora_U = nn.ModuleDict()
        self.phi = nn.ModuleDict()
        self.is_linear = {}

        self._disable_adapters = False
        self.merged_adapters = []

    def update_layer(self, adapter_name: str, r: int, alpha: int, dropout: float, activation_fn: str):
        self.r[adapter_name] = r
        self.nlora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / max(1, r)

        self.nlora_V[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.nlora_U[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.nlora_dropout[adapter_name] = nn.Dropout(dropout)
        self.phi[adapter_name] = make_phi(activation_fn)
        self.is_linear[adapter_name] = activation_fn.lower() == "linear"

        # init: V random, U zeros => starts as base model
        nn.init.kaiming_uniform_(self.nlora_V[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.nlora_U[adapter_name].weight)

        self.set_adapter(adapter_name)

    def forward(self, x):
        out = self.base_layer(x)
        if self._disable_adapters:
            return out

        active = self.active_adapters if isinstance(self.active_adapters, list) else [self.active_adapters]
        for name in active:
            if name not in self.nlora_V:
                continue
            z = self.nlora_dropout[name](x)
            z = self.nlora_V[name](z)
            z = self.phi[name](z)
            out = out + self.nlora_U[name](z) * self.scaling[name]
        return out
    
    @torch.no_grad()
    def adapter_delta(self, x, adapter_name: str):
        z = self.inner_adapter_forward(x, adapter_name)
        delta = self.nlora_U[adapter_name](z) * self.scaling[adapter_name]
        return delta
    
    @torch.no_grad()
    def inner_adapter_forward(self, x, adapter_name: str):
        z = self.nlora_V[adapter_name](x)
        z = self.phi[adapter_name](z)
        return z

    @torch.no_grad()
    def accumulate_consolidation_stats(self, x, adapter_name: str, state: dict, off_load_to_cpu: bool = False, accum_dtype=torch.float32,
                                       **kwargs):
        """
        Docstring for accumulate_consolidation_stats
        x: [*, d_in]
        the state should hold
            -   "xxt":   [d, d]  sum of x_i x_i^T
            -   "xzt":   [d, m]  sum of x_i delta_i^T
            -   "zzt":   [r, r]  sum of z_i z_i^T
            -   "zx_dW": [r, d]  sum of z_i x_i^T (for zeroshift)
        :param self: Description
        :param x: Description
        :param adapter_name: Description
        :type adapter_name: str
        :param state: Description
        :type state: dict
        """

        if self.is_linear.get(adapter_name, False):
            return # skip stats accumulation for linear adapters since it's not needed for solving optimal U given V
        
        if x.dim() == 3:
            x2 = x.reshape(-1, x.size(-1))
        else:
            x2 = x
        
        x2 = x2.to(dtype=accum_dtype)

        delta = self.adapter_delta(x, adapter_name)  # [*, d_out]

        if delta.dim() == 3:
            delta2 = delta.reshape(-1, delta.size(-1))
        else:
            delta2 = delta

        dev = torch.device('cpu' if off_load_to_cpu else delta2.device)
        xA = x2.to(dev, dtype=accum_dtype)
        rA = delta2.to(dev, dtype=accum_dtype)
        d = xA.size(1)
        m = rA.size(1)
        if "xxt" not in state:
            state["xxt"] = torch.zeros((d, d), device=dev, dtype=accum_dtype)
            state["xzt"] = torch.zeros((d, m), device=dev, dtype=accum_dtype)
            state["count"] = 0
            
        state["xxt"].add_(xA.t() @ xA)
        state["xzt"].add_(xA.t() @ rA)
        state["count"] += xA.size(0)

        z = self.inner_adapter_forward(x, adapter_name)  # [*, r]
        if z.dim() == 3:
            z2 = z.reshape(-1, z.size(-1))
        else:
            z2 = z
        
        zA = z2.to(dev, dtype=accum_dtype)

        if "zzt" not in state:
            state["zzt"]    = zA.T @ zA           # [r, r]
            state["zx_dW"]  = zA.T @ xA           # [r, d] -- accumulate this instead
        else:
            state["zzt"]   += zA.T @ zA
            state["zx_dW"] += zA.T @ xA


    @torch.no_grad()
    def solve_dU(self, adapter_name: str, dW, state: dict, lambda_: float, scale_lambda_by_trace=True):
        """
        Solve for optimal U given V and the accumulated stats.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        Returns U of shape [r, out] which can be merged into base weights as base_w += (V @ U).T
        """
        # b = (U @ phi(x @ V).T - dW x).T = (phi(x @ V) @ U - dW.T x.T), this is the regression residual we want to minimize
        dev = state["zzt"].device
        accum_dtype = torch.float32
        samples = state["count"]
        dW = dW.to(dev, dtype=accum_dtype)
        alpha = self.scaling[adapter_name]
        zzt = state["zzt"] / samples  # [r, r]
        zzt = zzt.to(dtype=accum_dtype)

        zx_dW = state["zx_dW"] / samples  # [r, d]
        zx_dW = zx_dW.to(dtype=accum_dtype)
        target = -1/alpha * zx_dW @ dW  # [r, d] @ [d, m] = [r, m]

        if scale_lambda_by_trace:
            lambda_scaled = lambda_ * (torch.trace(zzt) / zzt.size(0)).clamp_min(1e-6)
        else:
            lambda_scaled = lambda_

        A = zzt + lambda_scaled * torch.eye(zzt.size(0), device=dev, dtype=accum_dtype)  # add regularization for numerical stability

        dU = torch.linalg.solve(A, target).T  # [r, out]
        return dU


    @torch.no_grad()
    def solve_dW(self, state: dict, lambda_: float, scale_lambda_by_trace=True, adapter_name: str = None):
        """
        Solve for optimal U given V and the accumulated stats.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        Returns dW of shape [r, out] which can be merged into base weights as base_w += (V @ dW).T
        """
        if self.is_linear.get(adapter_name, False):
            U = self.nlora_U[adapter_name].weight.data  # [r, out]
            V = self.nlora_V[adapter_name].weight.data  # [in, r]
            alpha = self.scaling[adapter_name]
            dW = alpha * V.T @ U # [out, in]
            return dW
        else:
            samples = state["count"]
            xxt = state["xxt"] / samples  # [d, d]
            xzt = state["xzt"] / samples  # [d, out]
            rank = self.r[adapter_name]
            d = xxt.size(0)

            I = torch.eye(d, device=xxt.device, dtype=xxt.dtype)

            if scale_lambda_by_trace:
                # your stabilization heuristic, but now correctly applied
                lam = lambda_ * (torch.trace(xxt) * d * rank / samples).clamp_min(1e-6)
            else:
                lam = lambda_

            A = xxt + lam * I  # add scaled identity for numerical stability (and to prevent overfitting when data is limited)

            dW = torch.linalg.solve(A, xzt)  # [d, out]

            return dW

    @torch.no_grad()
    def solve_and_merge(self, state: dict, lr_: float, lambda_w: float, scale_by_lambda_: bool,
                        adapter_name:str, inplace_disable_adapter=False, zeroshift=False):
        """
        Solve for optimal U given V and the accumulated stats, then merge into base layer.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        """
        dW = self.solve_dW(state, lambda_=lambda_w, scale_lambda_by_trace=scale_by_lambda_, adapter_name=adapter_name)  # [d, out]

        base_w = self.base_layer.weight.data  # [out, in]

        if zeroshift:
            dU = self.solve_dU(adapter_name, lr_ * dW, state, lambda_=lambda_w,
                               scale_lambda_by_trace=scale_by_lambda_)  # [r, out]
            dU = dU.to(self.nlora_U[adapter_name].weight.data.device,
                        dtype=self.nlora_U[adapter_name].weight.data.dtype)

            with torch.no_grad():
                self.nlora_U[adapter_name].weight.data += dU

        if inplace_disable_adapter:
            self._disable_adapters = True
        
        dW = dW.to(base_w.device, dtype=base_w.dtype)
        base_w.data.add_(dW.t() * lr_)
