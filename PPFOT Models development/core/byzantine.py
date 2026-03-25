"""
Byzantine-Robust Aggregation
==============================
Implements Algorithm 1 from the manuscript:

Byzantine-Robust Transport Plan Aggregation in Wasserstein space.

Key bound [Eq. 8]:
  ‖γ_global − γ_true‖_F ≤ O(√(q / K(1−q))) + O(1/√n)

Parameters from manuscript:
  - Byzantine tolerance: q = 0.4  (40% malicious)
  - Threshold multiplier: α = 2.0
  - Aggregation: geometric median in W₂ space with trimmed-mean refinement
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pairwise Wasserstein distance (for transport plans)
# ---------------------------------------------------------------------------

def pairwise_wasserstein_l2(
    plans: List[torch.Tensor],
) -> torch.Tensor:
    """Compute pairwise Frobenius (≈ W₂ proxy) distances between transport plans."""
    K = len(plans)
    D = torch.zeros(K, K)
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.norm(plans[i] - plans[j], p="fro").item()
            D[i, j] = dist
            D[j, i] = dist
    return D


# ---------------------------------------------------------------------------
# Algorithm 1: Byzantine-Robust Transport Plan Aggregation
# ---------------------------------------------------------------------------

class ByzantineDetector:
    """Detect and filter Byzantine (malicious) clients.

    Implements Algorithm 1:
      1. Compute pairwise W₂ distances between client transport plans
      2. For each client, compute median distance to others
      3. Flag outliers: median_k > τ = α · median{median_k}
      4. Remove flagged clients + trim ⌊qK⌋ from each tail
      5. Weighted aggregation of surviving clients

    Parameters
    ----------
    tolerance_q : float
        Maximum Byzantine fraction (default 0.4).
    threshold_multiplier : float
        α for outlier threshold (default 2.0).
    """

    def __init__(
        self,
        tolerance_q: float = 0.4,
        threshold_multiplier: float = 2.0,
    ):
        self.tolerance_q = tolerance_q
        self.threshold_multiplier = threshold_multiplier

    def detect_and_aggregate(
        self,
        plans: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, List[int], dict]:
        """Run Algorithm 1.

        Parameters
        ----------
        plans : list of Tensor
            Local transport plans {γₖ} from K clients.
        weights : list of float, optional
            Per-client weights (default: uniform).

        Returns
        -------
        gamma_global : Tensor   – aggregated transport plan
        flagged      : list[int] – indices of flagged Byzantine clients
        info         : dict      – diagnostics
        """
        K = len(plans)
        if weights is None:
            weights = [1.0 / K] * K

        # Step 1: Pairwise distances
        D = pairwise_wasserstein_l2(plans)

        # Step 2: Per-client median distance
        medians = torch.zeros(K)
        for k in range(K):
            others = [D[k, l].item() for l in range(K) if l != k]
            medians[k] = float(np.median(others))

        # Step 3: Outlier threshold  τ = α · median{median_k}
        global_median = torch.median(medians).item()
        tau = self.threshold_multiplier * global_median

        # Flag outliers
        flagged = [k for k in range(K) if medians[k] > tau]

        # Step 4: Remove flagged, then trim tails
        surviving = [k for k in range(K) if k not in flagged]
        n_trim = int(np.floor(self.tolerance_q * K))

        if len(surviving) > 2 * n_trim:
            # Sort by median distance and trim both tails
            surviving_sorted = sorted(surviving, key=lambda k: medians[k])
            surviving = surviving_sorted[n_trim : len(surviving_sorted) - n_trim]

        # Step 5: Weighted aggregation
        if not surviving:
            # Fallback: use all non-flagged
            surviving = [k for k in range(K) if k not in flagged]
            if not surviving:
                surviving = list(range(K))  # absolute fallback

        total_weight = sum(weights[k] for k in surviving)
        gamma_global = torch.zeros_like(plans[0])
        for k in surviving:
            gamma_global += (weights[k] / total_weight) * plans[k]

        info = {
            "n_clients": K,
            "n_flagged": len(flagged),
            "n_surviving": len(surviving),
            "threshold_tau": tau,
            "median_distances": medians.tolist(),
            "flagged_indices": flagged,
            "surviving_indices": surviving,
        }
        return gamma_global, flagged, info


# ---------------------------------------------------------------------------
# Wasserstein Geometric Median (for model parameter aggregation)
# ---------------------------------------------------------------------------

def wasserstein_geometric_median(
    client_params: List[Dict[str, torch.Tensor]],
    max_iter: int = 50,
    tolerance: float = 1e-5,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute geometric median of client model parameters in W₂ space.

    Weiszfeld's algorithm adapted to parameter dictionaries.
    Used as robust aggregation alternative to FedAvg.
    """
    K = len(client_params)
    if weights is None:
        weights = [1.0 / K] * K

    keys = list(client_params[0].keys())

    # Initialise with weighted mean
    median = {}
    for key in keys:
        median[key] = sum(
            w * client_params[k][key] for k, w in enumerate(weights)
        )

    for _ in range(max_iter):
        # Compute distances
        distances = []
        for k in range(K):
            d = sum(
                torch.norm(median[key] - client_params[k][key]).item() ** 2
                for key in keys
            )
            distances.append(max(np.sqrt(d), 1e-8))

        # Weiszfeld update
        new_median = {}
        denom = sum(weights[k] / distances[k] for k in range(K))
        for key in keys:
            new_median[key] = sum(
                (weights[k] / distances[k]) * client_params[k][key]
                for k in range(K)
            ) / denom

        # Check convergence
        shift = sum(
            torch.norm(new_median[key] - median[key]).item() ** 2
            for key in keys
        )
        median = new_median
        if np.sqrt(shift) < tolerance:
            break

    return median


# ---------------------------------------------------------------------------
# Byzantine attack simulators (for evaluation)
# ---------------------------------------------------------------------------

def simulate_label_flip(
    plans: List[torch.Tensor],
    n_byzantine: int,
    attack_type: str = "random",
    seed: int = 42,
) -> List[torch.Tensor]:
    """Simulate Byzantine attacks on transport plans.

    Parameters
    ----------
    plans : list of Tensor
        Honest transport plans.
    n_byzantine : int
        Number of clients to corrupt.
    attack_type : str
        ``"random"`` – replace with random noise.
        ``"sign_flip"`` – negate the plan.
        ``"scale"`` – amplify by 10×.
    """
    rng = np.random.default_rng(seed)
    byzantine_idx = rng.choice(len(plans), size=n_byzantine, replace=False)
    corrupted = [p.clone() for p in plans]

    for idx in byzantine_idx:
        if attack_type == "random":
            corrupted[idx] = torch.rand_like(plans[idx])
            corrupted[idx] /= corrupted[idx].sum()
        elif attack_type == "sign_flip":
            corrupted[idx] = -plans[idx]
            corrupted[idx] = torch.clamp(corrupted[idx], min=0)
            total = corrupted[idx].sum()
            if total > 0:
                corrupted[idx] /= total
        elif attack_type == "scale":
            corrupted[idx] = plans[idx] * 10.0
            corrupted[idx] /= corrupted[idx].sum()

    return corrupted, list(byzantine_idx)
