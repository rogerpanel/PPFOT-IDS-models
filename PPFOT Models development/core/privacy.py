"""
Differential Privacy Components
================================
Implements the Rényi-DP (RDP) framework from the manuscript:

  - Gaussian mechanism with calibrated noise  [Eq. 5]:
      σ² = 2Δ² log(1.25/δ) / ε²
  - Moments accountant / RDP composition     [Eq. 6]
  - RDP-to-(ε,δ)-DP conversion with optimised α* ≈ 15.1
  - Noisy histogram construction              [Eq. 11]:
      h̃ᵏ = hᵏ + N(0, σ²I)

Privacy parameters from the manuscript (T=50 rounds):
  Global ε = 0.85,  δ = 5×10⁻⁴,  per-round δ₀ = 10⁻⁵
  Noise variance σ² = 5.127 × 10⁻⁸
  Sensitivity Δ = √2/n
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Gaussian Mechanism  [Eq. 5, 11]
# ---------------------------------------------------------------------------

class GaussianMechanism:
    """Calibrated Gaussian mechanism for differential privacy.

    Parameters
    ----------
    epsilon : float   – per-query privacy budget
    delta : float     – failure probability
    sensitivity : float or "auto"
        L2 sensitivity Δ.  If "auto", computed as √2/n.
    """

    def __init__(
        self,
        epsilon: float = 0.85,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        """Compute noise scale: σ² = 2Δ² log(1.25/δ) / ε²  [Eq. 5]."""
        sigma_sq = (
            2.0
            * self.sensitivity ** 2
            * math.log(1.25 / self.delta)
            / self.epsilon ** 2
        )
        return math.sqrt(sigma_sq)

    @classmethod
    def from_dataset(
        cls, n_samples: int, epsilon: float = 0.85, delta: float = 1e-5
    ) -> "GaussianMechanism":
        """Factory with sensitivity Δ = √2/n (histogram sensitivity)."""
        sensitivity = math.sqrt(2.0) / n_samples
        return cls(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise: x̃ = x + N(0, σ²I)  [Eq. 11]."""
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def add_noise_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy variant for histogram perturbation."""
        noise = np.random.randn(*x.shape) * self.sigma
        return x + noise

    def privatise_histogram(self, h: np.ndarray) -> np.ndarray:
        """Privatise a normalised histogram: h̃ = h + N(0,σ²I), then re-normalise."""
        h_noisy = self.add_noise_numpy(h)
        h_noisy = np.clip(h_noisy, 0, None)
        total = h_noisy.sum()
        if total > 0:
            h_noisy /= total
        return h_noisy


# ---------------------------------------------------------------------------
# Rényi Differential Privacy Accountant  [Eq. 6]
# ---------------------------------------------------------------------------

class RDPAccountant:
    """Track cumulative privacy loss via Rényi Divergence composition.

    Per-round RDP: ρ₁(α) = α/(2σ²)  for Gaussian mechanism.
    After T rounds: ρ_T(α) = T · ρ₁(α)
    Conversion: ε(α) = ρ_T(α) + log(1/δ)/(α-1)

    The manuscript reports optimised α* ≈ 15.1 yielding global ε ≈ 0.85
    for T=50, δ=5×10⁻⁴, n=1.6×10⁵, B=100.
    """

    def __init__(
        self,
        sigma: float,
        delta: float = 5e-4,
        alpha_range: Optional[List[float]] = None,
    ):
        self.sigma = sigma
        self.delta = delta
        self.alpha_range = alpha_range or np.linspace(2, 100, 500).tolist()
        self.n_rounds = 0

    def step(self):
        """Record one composition step (communication round)."""
        self.n_rounds += 1

    def per_round_rdp(self, alpha: float) -> float:
        """Per-round RDP: ρ₁(α) = α / (2σ²)."""
        return alpha / (2.0 * self.sigma ** 2)

    def total_rdp(self, alpha: float) -> float:
        """Composed RDP after T rounds: ρ_T(α) = T · ρ₁(α)."""
        return self.n_rounds * self.per_round_rdp(alpha)

    def rdp_to_epsilon(self, alpha: float) -> float:
        """Convert RDP to (ε,δ)-DP: ε(α) = ρ_T(α) + log(1/δ)/(α−1)."""
        rho = self.total_rdp(alpha)
        return rho + math.log(1.0 / self.delta) / (alpha - 1.0)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Compute tightest (ε, δ) by optimising over α.

        Returns the minimum ε across the α grid (α* ≈ 15.1 in practice).
        """
        best_eps = float("inf")
        best_alpha = 0.0
        for alpha in self.alpha_range:
            eps = self.rdp_to_epsilon(alpha)
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha
        return best_eps, best_alpha

    def report(self) -> dict:
        """Full privacy report."""
        eps, alpha_star = self.get_privacy_spent()
        return {
            "epsilon": eps,
            "delta": self.delta,
            "alpha_star": alpha_star,
            "rounds": self.n_rounds,
            "sigma": self.sigma,
        }


# ---------------------------------------------------------------------------
# Privacy Engine  (wraps mechanism + accountant for training)
# ---------------------------------------------------------------------------

class PrivacyEngine:
    """End-to-end privacy engine for federated training.

    Combines Gaussian mechanism (noise addition) with RDP accounting
    to track privacy budget across communication rounds.
    """

    def __init__(
        self,
        n_samples: int,
        epsilon_target: float = 0.85,
        delta: float = 5e-4,
        delta_per_round: float = 1e-5,
        n_bins: int = 100,
    ):
        self.n_samples = n_samples
        self.epsilon_target = epsilon_target
        self.delta = delta

        # Sensitivity for histogram queries: Δ = √2/n
        self.sensitivity = math.sqrt(2.0) / n_samples

        # Calibrate noise
        self.mechanism = GaussianMechanism(
            epsilon=epsilon_target,
            delta=delta_per_round,
            sensitivity=self.sensitivity,
        )

        # Accountant
        self.accountant = RDPAccountant(
            sigma=self.mechanism.sigma, delta=delta
        )

    def privatise_transport_plan(self, gamma: torch.Tensor) -> torch.Tensor:
        """Add DP noise to a transport plan and re-project to simplex."""
        gamma_noisy = self.mechanism.add_noise(gamma)
        gamma_noisy = torch.clamp(gamma_noisy, min=0)
        total = gamma_noisy.sum()
        if total > 0:
            gamma_noisy = gamma_noisy / total
        return gamma_noisy

    def privatise_gradients(
        self, gradients: List[torch.Tensor], max_grad_norm: float = 1.0
    ) -> List[torch.Tensor]:
        """Clip and add noise to gradients (DP-SGD style)."""
        # Global gradient clipping
        total_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in gradients)
        )
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)

        noisy_grads = []
        for g in gradients:
            clipped = g * clip_coef
            noisy_grads.append(self.mechanism.add_noise(clipped))
        return noisy_grads

    def step(self):
        """Record a composition step."""
        self.accountant.step()

    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.accountant.get_privacy_spent()

    def budget_remaining(self) -> float:
        eps, _ = self.get_privacy_spent()
        return max(0.0, self.epsilon_target - eps)
