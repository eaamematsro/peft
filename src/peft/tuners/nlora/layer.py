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

        # Initialize adapter weights so to have unit variance
        nn.init.normal_(self.nlora_V[adapter_name].weight, std=(r)**(-1/2))
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
            state["batch_count"] = 0
            
        state["xxt"].add_(xA.t() @ xA)
        state["xzt"].add_(xA.t() @ rA)
        state["count"] += xA.size(0)
        state["batch_count"] += 1

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
            state["zzt"].add_(zA.T @ zA)
            state["zx_dW"].add_(zA.T @ xA)

    @torch.no_grad()
    def solve_dU(self, adapter_name: str, dW, state: dict, lambda_: float, scale_lambda_by_trace=True,
                 lambda_schedule=None):
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

        d = zzt.size(0)

        if scale_lambda_by_trace:
            if lambda_schedule is None:
                # default is to scale by trace of xxt
                lam = lambda_ * (torch.trace(zzt) / d ).clamp_min(1e-6)
            elif lambda_schedule == "mean_eigenvalue":
                lam = lambda_ * (torch.trace(zzt) / d ).clamp_min(1e-6)
            elif lambda_schedule == "max_eigenvalue":
                # power iteration to estimate max eigenvalue for better scaling
                x = torch.randn((d, 1), device=zzt.device, dtype=zzt.dtype)
                for _ in range(10):
                    x = zzt @ x
                    x = x / x.norm()
                max_eig = (x.t() @ zzt @ x) / (x.t() @ x)
                lam = lambda_ * max_eig.item()
            elif lambda_schedule == 'rank_adaptive':
                # adaptive scaling based on the rank of the data
                rank = self.r[adapter_name]
                lam = lambda_ * (torch.trace(zzt) * rank / d ).clamp_min(1e-6)
            elif lambda_schedule == 'sample_adaptive':
                # adaptive scaling based on the number of samples seen
                lam = lambda_ * (torch.trace(zzt) / (d  * math.sqrt(state["count"]))).clamp_min(1e-6)

            elif lambda_schedule == 'rank_and_sample_adaptive':
                rank = self.r[adapter_name]
                lam = lambda_ * (torch.trace(zzt) * rank / (d  * math.sqrt(state["count"]))).clamp_min(1e-6)
            else:
                lam = lambda_
        else:
            lam = lambda_

        A = zzt + lam * torch.eye(zzt.size(0), device=dev, dtype=accum_dtype)  # add regularization for numerical stability

        dU = torch.linalg.solve(A, target).T  # [r, out]
        return dU


    @torch.no_grad()
    def solve_dW(self, state: dict, lambda_: float, scale_lambda_by_trace=True, adapter_name: str = None,
                 lambda_schedule=None):
        """
        Solve for optimal U given V and the accumulated stats.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        Returns dW of shape [r, out] which can be merged into base weights as base_w += (V @ dW).T
        """

        samples = state["count"]
        xxt = state["xxt"] / samples  # [d, d]
        xzt = state["xzt"] / samples  # [d, out]
        d = xxt.size(0)
        rank = self.r[adapter_name]

        I = torch.eye(d, device=xxt.device, dtype=xxt.dtype)

        if scale_lambda_by_trace:
            if lambda_schedule is None:
                # default is to scale by trace of xxt
                lam = lambda_ * (torch.trace(xxt) / d ).clamp_min(1e-6)
            elif lambda_schedule == "mean_eigenvalue":
                lam = lambda_ * (torch.trace(xxt) / d ).clamp_min(1e-6)
            elif lambda_schedule == "max_eigenvalue":
                # power iteration to estimate max eigenvalue for better scaling
                x = torch.randn((d, 1), device=xxt.device, dtype=xxt.dtype)
                for _ in range(10):
                    x = xxt @ x
                    x = x / x.norm()
                max_eig = (x.t() @ xxt @ x) / (x.t() @ x)
                lam = lambda_ * max_eig.item()
            elif lambda_schedule == 'rank_adaptive':
                # adaptive scaling based on the rank of the data
                rank = self.r[adapter_name]
                lam = lambda_ * (torch.trace(xxt) * rank / d ).clamp_min(1e-6)
            elif lambda_schedule == 'sample_adaptive':
                # adaptive scaling based on the number of samples seen
                lam = lambda_ * (torch.trace(xxt) / (d  * math.sqrt(state["count"]))).clamp_min(1e-6)

            elif lambda_schedule == 'rank_and_sample_adaptive':
                rank = self.r[adapter_name]
                lam = lambda_ * (torch.trace(xxt) * rank / (d  * math.sqrt(state["count"]))).clamp_min(1e-6)
            else:
                lam = lambda_
        else:
            lam = lambda_

        # eigs = torch.linalg.eigvalsh(xxt.cpu())
        # print(eigs.max() / eigs.min())
        # print("Using schedule:", lambda_schedule)
        # print((eigs.max() + lam) / (eigs.min() + lam))
        # import pdb; pdb.set_trace()
        A = xxt + lam * I  # add scaled identity for numerical stability (and to prevent overfitting when data is limited)

        dW = torch.linalg.solve(A, xzt)  # [d, out]

        return dW

    @torch.no_grad()
    def solve_and_merge(self, state: dict, lr_: float, lambda_w: float, scale_by_lambda_: bool,
                        adapter_name:str, inplace_disable_adapter=False, zeroshift=False, shift_V: bool=False,
                        lambda_schedule=None):
        """
        Solve for optimal U given V and the accumulated stats, then merge into base layer.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        """
        dW = self.solve_dW(state, lambda_=lambda_w, scale_lambda_by_trace=scale_by_lambda_, adapter_name=adapter_name,
                           lambda_schedule=lambda_schedule)  # [d, out]

        base_w = self.base_layer.weight.data  # [out, in]

        # print("Consolidation dW relative norm:", dW.norm().item() * lr_ / (base_w.norm().item() + 1e-6))

        if shift_V and self.is_linear.get(adapter_name, False):
            self.shift_V_linear(
                adapter_name=adapter_name,
                dW=dW,
                lr=lr_,
                lambda_=lambda_w,
            )
        elif zeroshift:
            print("Performing zero-shift consolidation")
            dU = self.solve_dU(
                adapter_name, lr_ * dW, state,
                lambda_=lambda_w,
                scale_lambda_by_trace=scale_by_lambda_,
                lambda_schedule=lambda_schedule
            )
            dU = dU.to(self.nlora_U[adapter_name].weight.data.device,
                    dtype=self.nlora_U[adapter_name].weight.data.dtype)
            self.nlora_U[adapter_name].weight.data += dU

        if inplace_disable_adapter:
            print("Disabling adapter after consolidation")
            self._disable_adapters = True
        
        dW = dW.to(base_w.device, dtype=base_w.dtype)
        base_w.data.add_(dW.t() * lr_)

    @torch.no_grad()
    def shift_V_linear(self, adapter_name: str, dW: torch.Tensor, lr: float, lambda_: float):
        """
        For linear adapters: after merging lr*dW into base weights, construct V' in
        the orthogonal complement of V's row space and solve for U' such that:

            x V'^T U'^T * scaling = x V^T U^T * scaling - x dW^T * lr

        i.e. the new adapter exactly reproduces the residual after the base weight merge.
        This sets gamma = 0 by construction since col(V') ⊥ col(V).

        Args:
            adapter_name: name of the adapter
            dW:           [in, out] — weight update already merged (before lr scaling)
            lr:           consolidation learning rate
            lambda_:      ridge regularization for numerical stability
        """
        assert self.is_linear.get(adapter_name, False), \
            "shift_V_linear only valid for linear adapters"

        V       = self.nlora_V[adapter_name].weight.data   # [r, in]
        U       = self.nlora_U[adapter_name].weight.data   # [out, r]
        scaling = self.scaling[adapter_name]
        r       = self.r[adapter_name]
        d_in    = V.shape[1]

        # ── target: residual the new adapter must reproduce ──────────────
        # x V'^T U'^T * scaling = x(V^T U^T * scaling - dW * lr)
        # so V'^T U'^T = (V^T U^T * scaling - dW * lr) / scaling
        target = (V.T @ U.T * scaling - dW.to(V.dtype) * lr) / scaling   # [in, out]

        # ── construct V' in orthogonal complement of V's row space ───────
        # QR on V^T gives orthonormal basis for col(V^T) = row space of V
        # Q: [in, in], first r cols span row space of V
        #              remaining in-r cols span orthogonal complement
        V_cpu = V.cpu()  # QR can be unstable on GPU, so move to CPU if needed
        Q, _ = torch.linalg.qr(
            V_cpu.T,           # [in, r]
            mode='complete'  # full [in, in] orthogonal Q
        )
        V_orth = Q[:, r:].T    # [in-r, in] — orthogonal complement basis

        # sample r rows from orthogonal complement
        # use random permutation so V' isn't always the same directions
        idx   = torch.randperm(V_orth.shape[0])[:r]
        V_new = V_orth[idx]    # [r, in]

        # normalize V' to match V's Frobenius norm for consistent scaling
        V_new = V_new.to(V.device, dtype=V.dtype)  # move back to original device if needed

        V_new = V_new * V.norm() / V_new.norm()


        # ── solve for U' ──────────────────────────────────────────────────
        # V'^T U'^T = target
        # U'^T = (V' V'^T + lambda I)^{-1} V' target
        # U'   = target^T V'^T (V' V'^T + lambda I)^{-1}

        VVt = V_new @ V_new.T                                              # [r, r]
        lam = lambda_ * (torch.trace(VVt) / r).clamp_min(1e-6)
        A   = VVt + lam * torch.eye(r, device=V.device, dtype=V.dtype)    # [r, r]
        rhs = V_new @ target                                               # [r, out]
        U_new = torch.linalg.solve(A, rhs).T                               # [out, r]

        # ── update adapter weights ────────────────────────────────────────
        self.nlora_V[adapter_name].weight.data.copy_(V_new)
        self.nlora_U[adapter_name].weight.data.copy_(U_new)
