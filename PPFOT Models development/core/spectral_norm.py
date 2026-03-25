"""
Spectral Normalisation
=======================
Constrains Lipschitz constant of neural network layers via spectral
normalisation: W_SN = W / σ(W)  [Eq. 9]

Combined with the certified robustness bound [Eq. 10]:
  ‖f(x+δ) − f(x)‖₂ ≤ L_T · L_c · ε_adv

where L_T and L_c are the Lipschitz constants of the transport map
and classifier respectively.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpectralNorm(nn.Module):
    """Power-iteration spectral normalisation wrapper.

    Applies W_SN = W / σ(W) at each forward pass via iterative
    singular value estimation.

    Parameters
    ----------
    module : nn.Module
        Layer to normalise (typically nn.Linear).
    power_iterations : int
        Number of power iteration steps (default 1).
    """

    def __init__(self, module: nn.Module, power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations

        if hasattr(module, "weight"):
            w = module.weight.data
            h = w.shape[0]
            w_flat = w.view(h, -1)

            u = torch.randn(h)
            u = u / (u.norm() + 1e-12)
            v = torch.randn(w_flat.shape[1])
            v = v / (v.norm() + 1e-12)

            self.register_buffer("_u", u)
            self.register_buffer("_v", v)

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm() + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.module, "weight"):
            w = self.module.weight
            h = w.shape[0]
            w_flat = w.view(h, -1)

            u, v = self._u, self._v
            for _ in range(self.power_iterations):
                v = self._l2norm(w_flat.t() @ u)
                u = self._l2norm(w_flat @ v)

            sigma = u @ w_flat @ v
            self.module.weight.data = w.data / sigma

            self._u.copy_(u)
            self._v.copy_(v)

        return self.module(x)


def spectral_normalize(
    module: nn.Module, power_iterations: int = 1
) -> nn.Module:
    """Convenience wrapper: apply spectral norm to all Linear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, SpectralNorm(child, power_iterations))
        else:
            spectral_normalize(child, power_iterations)
    return module
