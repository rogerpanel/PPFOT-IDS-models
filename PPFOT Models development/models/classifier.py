"""
Classifier Network
===================
Architecture from Section III-C:
  Feature_dim → 128 → 64 → num_classes
  With dropout regularisation.
"""

import torch.nn as nn
from typing import List, Optional

from core.spectral_norm import spectral_normalize


class Classifier(nn.Module):
    """Classification head for intrusion detection.

    Parameters
    ----------
    input_dim : int
        Feature dimension from the extractor.
    num_classes : int
        Number of attack categories.
    hidden_dims : list[int]
        Hidden layer sizes (default [128, 64]).
    dropout : float
        Dropout rate (default 0.2).
    use_spectral_norm : bool
        Apply spectral normalisation (default True).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        if use_spectral_norm:
            spectral_normalize(self.network)

    def forward(self, x):
        return self.network(x)
