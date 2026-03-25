"""
PPFOT-IDS: Full Model
======================
Privacy-Preserving Federated Optimal Transport for Intrusion Detection Systems.

Integrates:
  - Shared feature extractor (Section III-C)
  - Cloud-specific adaptation layers
  - Classifier head
  - Kantorovich dual potentials for adversarial OT
  - Privacy engine (Gaussian mechanism + RDP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .feature_extractor import FeatureExtractor
from .classifier import Classifier
from .transport_map import KantorovichDual


class CloudAdapter(nn.Module):
    """Cloud-specific adaptation layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.adapter(x)


class PPFOT_IDS(nn.Module):
    """Privacy-Preserving Federated OT Intrusion Detection System.

    Parameters
    ----------
    input_dim : int
        Raw feature dimension.
    num_classes : int
        Number of attack categories (7 in ICS3D: benign + 6 attack types).
    num_clouds : int
        Number of cloud domains (default 3: Container, IoT, Enterprise).
    hidden_dims : list[int]
        Feature extractor hidden dimensions (default [256, 128, 64]).
    classifier_dims : list[int]
        Classifier hidden dimensions (default [128, 64]).
    dropout : float
        Dropout rate (default 0.2).
    use_spectral_norm : bool
        Apply spectral normalisation (default True).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_clouds: int = 3,
        hidden_dims: Optional[List[int]] = None,
        classifier_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if classifier_dims is None:
            classifier_dims = [128, 64]

        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=True,
            use_spectral_norm=use_spectral_norm,
        )

        feat_dim = self.feature_extractor.output_dim

        # Cloud-specific adapters
        self.cloud_adapters = nn.ModuleList(
            [CloudAdapter(feat_dim) for _ in range(num_clouds)]
        )

        # Classifier
        self.classifier = Classifier(
            input_dim=feat_dim,
            num_classes=num_classes,
            hidden_dims=classifier_dims,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
        )

        # Adversarial OT via Kantorovich duality
        self.kantorovich = KantorovichDual(
            feature_dim=feat_dim,
            hidden_dim=hidden_dims[0],
            use_spectral_norm=use_spectral_norm,
        )

        self.num_clouds = num_clouds
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def forward(
        self,
        x: torch.Tensor,
        cloud_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor [batch, input_dim]
        cloud_id : int, optional
            Cloud domain index for cloud-specific adaptation.

        Returns
        -------
        logits : Tensor [batch, num_classes]
        features : Tensor [batch, feat_dim]
        """
        features = self.feature_extractor(x)

        if cloud_id is not None and 0 <= cloud_id < self.num_clouds:
            features = self.cloud_adapters[cloud_id](features)

        logits = self.classifier(features)
        return logits, features

    def compute_ot_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Adversarial OT loss between source and target features."""
        return self.kantorovich(source_features, target_features)

    def get_all_parameters(self) -> Dict[str, torch.Tensor]:
        """Return a flat dict of all parameters (for federated aggregation)."""
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def load_parameters(self, params: Dict[str, torch.Tensor]):
        """Load aggregated parameters from server."""
        state = self.state_dict()
        for name, value in params.items():
            if name in state:
                state[name] = value
        self.load_state_dict(state)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
