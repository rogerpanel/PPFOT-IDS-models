"""
Transport Map Network
======================
Implements Kantorovich dual potentials f_φ and g_ψ for adversarial OT.

The dual formulation uses:
  W₁(μ,ν) = sup_{‖f‖_L ≤ 1} E_μ[f(x)] − E_ν[f(y)]

With spectral normalisation enforcing the Lipschitz constraint [Eq. 9].

The transport cost matrix is derived from the dual potentials:
  C(x,y) ≈ f_φ(x) + g_ψ(y)
"""

import torch
import torch.nn as nn
from typing import Optional

from core.spectral_norm import SpectralNorm


def _build_potential(
    input_dim: int,
    hidden_dim: int,
    use_spectral_norm: bool = True,
) -> nn.Sequential:
    """Build a Kantorovich potential network."""
    layers = []
    dims = [input_dim, hidden_dim, hidden_dim, 1]
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        if use_spectral_norm:
            layers.append(SpectralNorm(linear))
        else:
            layers.append(linear)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KantorovichDual(nn.Module):
    """Kantorovich dual potentials for adversarial OT.

    Computes the Kantorovich-Rubinstein duality:
      W₁ ≈ E_source[f_φ(x)] − E_target[g_ψ(y)]

    Parameters
    ----------
    feature_dim : int
        Dimension of the feature space.
    hidden_dim : int
        Hidden layer dimension (default 256).
    use_spectral_norm : bool
        Enforce Lipschitz via spectral norm (default True).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.f_phi = _build_potential(feature_dim, hidden_dim, use_spectral_norm)
        self.g_psi = _build_potential(feature_dim, hidden_dim, use_spectral_norm)

    def forward(
        self, x_source: torch.Tensor, x_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute adversarial OT loss (Kantorovich-Rubinstein)."""
        f_x = self.f_phi(x_source)   # [n_source, 1]
        g_y = self.g_psi(x_target)   # [n_target, 1]
        return torch.mean(f_x) - torch.mean(g_y)

    def get_transport_cost(
        self, x_source: torch.Tensor, x_target: torch.Tensor
    ) -> torch.Tensor:
        """Approximate cost matrix from dual potentials."""
        f_x = self.f_phi(x_source)   # [n_s, 1]
        g_y = self.g_psi(x_target)   # [n_t, 1]
        return f_x + g_y.t()          # [n_s, n_t]


class TransportMapNetwork(nn.Module):
    """Learnable transport map T: source → target domain.

    Architecture: input → 256 → 128 → 64 → output
    with batch normalisation and spectral normalisation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: int = 256,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        layers = []
        dims = [input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, output_dim]
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                layers.append(SpectralNorm(linear))
            else:
                layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
