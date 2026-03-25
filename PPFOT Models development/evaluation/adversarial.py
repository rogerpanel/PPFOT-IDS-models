"""
Adversarial Robustness Evaluation
==================================
Implements adversarial attacks for robustness testing (Section IV-E):
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - Certified robustness bound [Eq. 10]:
      ‖f(x+δ) − f(x)‖₂ ≤ L_T · L_c · ε_adv
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AdversarialEvaluator:
    """Evaluate model robustness under adversarial perturbations.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    device : torch.device
    max_samples : int
        Maximum samples to evaluate (default 2000 for speed).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_samples: int = 2000,
    ):
        self.model = model
        self.device = device
        self.max_samples = max_samples

    # ----------------------------------------------------------------
    # FGSM Attack
    # ----------------------------------------------------------------

    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """FGSM: x_adv = x + ε · sign(∇_x L(θ, x, y))."""
        x_adv = x.clone().detach().requires_grad_(True)
        logits, _ = self.model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation
        return x_adv.detach()

    # ----------------------------------------------------------------
    # PGD Attack
    # ----------------------------------------------------------------

    def pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        step_size: float = 0.01,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """PGD: iterative projected gradient descent attack."""
        x_adv = x.clone().detach()
        x_adv += torch.empty_like(x_adv).uniform_(-epsilon, epsilon)

        for _ in range(n_steps):
            x_adv.requires_grad_(True)
            logits, _ = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + step_size * x_adv.grad.sign()
                # Project back to ε-ball
                delta = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = x + delta

        return x_adv.detach()

    # ----------------------------------------------------------------
    # Full evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        epsilons: Optional[List[float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Full adversarial robustness evaluation.

        Returns accuracy under clean, FGSM, and PGD attacks
        at each perturbation budget ε.
        """
        if epsilons is None:
            epsilons = [0.01, 0.05, 0.1, 0.2]

        self.model.eval()

        # Collect samples
        all_x, all_y = [], []
        count = 0
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
            count += len(y)
            if count >= self.max_samples:
                break
        X = torch.cat(all_x)[: self.max_samples].to(self.device)
        Y = torch.cat(all_y)[: self.max_samples].to(self.device)

        # Clean accuracy
        with torch.no_grad():
            logits_clean, _ = self.model(X)
            clean_acc = (logits_clean.argmax(1) == Y).float().mean().item()

        results = {"clean": {"accuracy": clean_acc}}

        for eps in epsilons:
            # FGSM
            x_fgsm = self.fgsm_attack(X, Y, eps)
            with torch.no_grad():
                logits_fgsm, _ = self.model(x_fgsm)
                fgsm_acc = (logits_fgsm.argmax(1) == Y).float().mean().item()

            # PGD
            x_pgd = self.pgd_attack(X, Y, eps)
            with torch.no_grad():
                logits_pgd, _ = self.model(x_pgd)
                pgd_acc = (logits_pgd.argmax(1) == Y).float().mean().item()

            results[f"fgsm_eps={eps}"] = {
                "accuracy": fgsm_acc,
                "robustness_gap": clean_acc - fgsm_acc,
            }
            results[f"pgd_eps={eps}"] = {
                "accuracy": pgd_acc,
                "robustness_gap": clean_acc - pgd_acc,
            }

        return results

    # ----------------------------------------------------------------
    # Certified robustness bound  [Eq. 10]
    # ----------------------------------------------------------------

    @staticmethod
    def certified_robustness_bound(
        lipschitz_transport: float,
        lipschitz_classifier: float,
        epsilon_adv: float,
    ) -> float:
        """Compute certified robustness bound:
        ‖f(x+δ) − f(x)‖₂ ≤ L_T · L_c · ε_adv  [Eq. 10]
        """
        return lipschitz_transport * lipschitz_classifier * epsilon_adv
