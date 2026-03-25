"""
Shared Feature Extractor
=========================
Architecture from Section III-C of the manuscript:
  Input → 256 → 128 → 64
  BatchNorm + ReLU + Dropout(0.2) after each layer
  Optional spectral normalisation [Eq. 9]
"""

import torch.nn as nn
from typing import List, Optional

from core.spectral_norm import spectral_normalize


class FeatureExtractor(nn.Module):
    """Shared feature extractor across cloud domains.

    Parameters
    ----------
    input_dim : int
        Raw feature dimension.
    hidden_dims : list[int]
        Hidden layer dimensions (default [256, 128, 64]).
    dropout : float
        Dropout rate (default 0.2).
    batch_norm : bool
        Whether to apply batch normalisation (default True).
    use_spectral_norm : bool
        Apply spectral normalisation to all layers (default True).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        batch_norm: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

        if use_spectral_norm:
            spectral_normalize(self.network)

    def forward(self, x):
        return self.network(x)
