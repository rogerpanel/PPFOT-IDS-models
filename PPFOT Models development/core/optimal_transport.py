"""
Optimal Transport Components
=============================
Implements the entropic-regularised OT with adaptive Sinkhorn scheduling
as described in Algorithm 2 of the manuscript.

Key equations:
  (1)  W_p(μ,ν) = inf_γ (∫ d(x,y)^p dγ)^(1/p)
  (2)  Discrete OT:  min_γ Σ_{ij} γ_{ij} C_{ij}
  (3)  Entropic OT:  min_γ Σ_{ij} γ_{ij} C_{ij} - ε Σ_{ij} γ_{ij} log γ_{ij}
  (4)  Sinkhorn form: γ* = diag(u) K diag(v),  K = exp(-C/ε)
"""

import numpy as np
import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Cost matrix computation
# ---------------------------------------------------------------------------

def compute_cost_matrix(
    X: torch.Tensor, Y: torch.Tensor, p: int = 2
) -> torch.Tensor:
    """Pairwise L_p cost matrix between rows of X and Y."""
    # Squared Euclidean via expansion
    x_sq = (X ** 2).sum(dim=1, keepdim=True)
    y_sq = (Y ** 2).sum(dim=1, keepdim=True)
    C = x_sq + y_sq.t() - 2.0 * X @ Y.t()
    C = torch.clamp(C, min=0.0)
    if p == 2:
        return C
    return C.sqrt() if p == 1 else C ** (p / 2)


# ---------------------------------------------------------------------------
# Sinkhorn algorithm with adaptive ε-scheduling  (Algorithm 2)
# ---------------------------------------------------------------------------

class AdaptiveSinkhorn:
    """Adaptive Sinkhorn with regularisation scheduling.

    Implements Algorithm 2:  ε-annealing from ε₀ → ε_min with decay ρ,
    achieving O(log(1/ε)) complexity versus O(ε⁻³) for fixed ε.

    Parameters
    ----------
    epsilon_init : float
        Initial regularisation ε₀ (default 0.5).
    epsilon_min : float
        Minimum regularisation ε_min (default 0.01).
    decay_rate : float
        Multiplicative decay ρ (default 0.9).
    tolerance : float
        Marginal constraint tolerance τ (default 1e-6).
    max_iter : int
        Maximum Sinkhorn iterations per ε level (default 1000).
    """

    def __init__(
        self,
        epsilon_init: float = 0.5,
        epsilon_min: float = 0.01,
        decay_rate: float = 0.9,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute regularised OT plan and cost.

        Parameters
        ----------
        a : Tensor [n]   – source marginal (sums to 1)
        b : Tensor [m]   – target marginal (sums to 1)
        C : Tensor [n,m] – cost matrix

        Returns
        -------
        gamma : Tensor [n,m]  – transport plan
        cost  : Tensor scalar – ⟨γ, C⟩
        info  : dict          – convergence diagnostics
        """
        n, m = C.shape
        device = C.device

        u = torch.ones(n, device=device)
        v = torch.ones(m, device=device)

        eps = self.epsilon_init
        total_iters = 0
        eps_schedule = []

        while eps >= self.epsilon_min:
            K = torch.exp(-C / eps)

            for _ in range(self.max_iter):
                Kv = K @ v
                u_new = a / (Kv + 1e-12)
                Ktu = K.t() @ u_new
                v_new = b / (Ktu + 1e-12)

                # Check marginal constraint
                err = torch.norm(a - u_new * (K @ v_new), p=1).item()
                total_iters += 1

                u, v = u_new, v_new

                if err < self.tolerance:
                    break

            eps_schedule.append(eps)
            eps = max(self.epsilon_min, self.decay_rate * eps)
            if eps <= self.epsilon_min:
                break

        # Final transport plan: γ* = diag(u) K diag(v)   [Eq. 4]
        gamma = torch.diag(u) @ K @ torch.diag(v)

        # Transport cost
        cost = (gamma * C).sum()

        info = {
            "total_iterations": total_iters,
            "final_epsilon": eps,
            "eps_schedule": eps_schedule,
            "marginal_error": err,
        }
        return gamma, cost, info


# ---------------------------------------------------------------------------
# Sinkhorn divergence (debiased)
# ---------------------------------------------------------------------------

def sinkhorn_divergence(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence: S_ε(X,Y) = OT_ε(X,Y) - ½OT_ε(X,X) - ½OT_ε(Y,Y).

    Used for gradient-based training where the divergence needs to be
    differentiable and unbiased.
    """
    n = X.shape[0]
    m = Y.shape[0]

    a = torch.ones(n, device=X.device) / n
    b = torch.ones(m, device=Y.device) / m

    solver = AdaptiveSinkhorn(
        epsilon_init=epsilon,
        epsilon_min=epsilon,
        decay_rate=1.0,
        max_iter=max_iter,
    )

    C_xy = compute_cost_matrix(X, Y)
    _, cost_xy, _ = solver(a, b, C_xy)

    C_xx = compute_cost_matrix(X, X)
    _, cost_xx, _ = solver(a, a, C_xx)

    C_yy = compute_cost_matrix(Y, Y)
    _, cost_yy, _ = solver(b, b, C_yy)

    return cost_xy - 0.5 * cost_xx - 0.5 * cost_yy


# ---------------------------------------------------------------------------
# POT-based Wasserstein (exact, for small-scale / evaluation)
# ---------------------------------------------------------------------------

def exact_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Exact OT via network simplex (POT library)."""
    import ot

    gamma = ot.emd(a, b, C)
    cost = np.sum(gamma * C)
    return gamma, cost
