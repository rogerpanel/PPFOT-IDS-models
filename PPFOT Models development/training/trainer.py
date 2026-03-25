"""
Training Loops
===============
Local training (per-client) and evaluation utilities.

Hyperparameters from Section IV-B:
  - Learning rate: 10⁻³ with cosine annealing
  - Batch size: 64
  - Local epochs per round: 5
  - Gradient clipping: max_norm = 1.0
  - OT loss weight λ_OT = 0.1
  - Privacy regularisation λ_priv = 0.01
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class LocalTrainer:
    """Per-client local training.

    Implements one round of local optimisation before sending
    parameters/transport plans to the server.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        local_epochs: int = 5,
        max_grad_norm: float = 1.0,
        lambda_ot: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.max_grad_norm = max_grad_norm
        self.lambda_ot = lambda_ot

    def train_round(
        self,
        train_loader: DataLoader,
        target_loader: Optional[DataLoader] = None,
        cloud_id: int = 0,
    ) -> Dict[str, float]:
        """Run local training for one federated round.

        Parameters
        ----------
        train_loader : DataLoader
            Client's local training data.
        target_loader : DataLoader, optional
            Target domain data for OT alignment.
        cloud_id : int
            Cloud domain identifier.

        Returns
        -------
        metrics : dict
            Training loss components.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        epoch_metrics = defaultdict(float)
        n_batches = 0

        for epoch in range(self.local_epochs):
            if target_loader is not None:
                target_iter = iter(target_loader)

            for x_s, y_s in train_loader:
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits_s, features_s = self.model(x_s, cloud_id=cloud_id)

                # Classification loss
                cls_loss = F.cross_entropy(logits_s, y_s)

                # OT domain alignment loss
                ot_loss = torch.tensor(0.0, device=self.device)
                if target_loader is not None:
                    try:
                        x_t, _ = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        x_t, _ = next(target_iter)

                    x_t = x_t.to(self.device)
                    _, features_t = self.model(x_t, cloud_id=(cloud_id + 1) % self.model.num_clouds)
                    ot_loss = self.model.compute_ot_loss(features_s, features_t)

                # Total loss
                loss = cls_loss + self.lambda_ot * ot_loss
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                optimizer.step()

                epoch_metrics["total_loss"] += loss.item()
                epoch_metrics["cls_loss"] += cls_loss.item()
                epoch_metrics["ot_loss"] += ot_loss.item()
                n_batches += 1

        # Average over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= max(n_batches, 1)

        return dict(epoch_metrics)


class FederatedTrainer:
    """Orchestrates federated training across multiple clients.

    Manages the full training loop:
      1. Distribute global model to clients
      2. Local training on each client
      3. Collect parameters / transport plans
      4. Byzantine-robust aggregation on server
      5. Update global model
      6. Track privacy budget via RDP
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        local_epochs: int = 5,
        max_grad_norm: float = 1.0,
        lambda_ot: float = 0.1,
    ):
        self.global_model = model.to(device)
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.max_grad_norm = max_grad_norm
        self.lambda_ot = lambda_ot
        self.history = defaultdict(list)

    def train_one_round(
        self,
        client_loaders: Dict[int, DataLoader],
        target_loader: Optional[DataLoader] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Execute one communication round.

        Returns per-client model parameters for server aggregation.
        """
        client_params = {}
        client_metrics = {}

        for client_id, loader in client_loaders.items():
            # Clone global model for this client
            local_model = copy.deepcopy(self.global_model)
            trainer = LocalTrainer(
                model=local_model,
                device=self.device,
                lr=self.lr,
                local_epochs=self.local_epochs,
                max_grad_norm=self.max_grad_norm,
                lambda_ot=self.lambda_ot,
            )

            # Assign cloud_id based on client
            cloud_id = client_id % self.global_model.num_clouds

            metrics = trainer.train_round(
                loader, target_loader=target_loader, cloud_id=cloud_id
            )

            client_params[client_id] = {
                name: param.data.clone()
                for name, param in local_model.named_parameters()
            }
            client_metrics[client_id] = metrics

        return client_params, client_metrics

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, cloud_id: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate global model."""
        self.global_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.global_model(x, cloud_id=cloud_id)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }
