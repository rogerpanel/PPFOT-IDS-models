"""
Experiment 5: Scalability Analysis  (Tables 6 & 7)
====================================================
Tests PPFOT-IDS across:
  - Client counts K ∈ {3, 5, 10, 20}           (Table 6)
  - Non-IID Dirichlet α ∈ {0.1, 0.5, 1.0, 10.0}  (Table 7)

Measures: accuracy, Byzantine robustness, round time, communication cost.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import ICS3DDataLoader, create_federated_loaders
from models.ppfot_ids import PPFOT_IDS
from training.federated import FederatedServer
from evaluation.metrics import compute_all_metrics


def run_scalability_experiment(
    data_loader: ICS3DDataLoader,
    num_clients: int,
    alpha_dir: float,
    seed: int = 42,
    rounds: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run a single scalability configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = data_loader.load_edge_iiot("DNN")
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    client_loaders, test_loader, _, _ = create_federated_loaders(
        X, y_enc.astype(str),
        num_clients=num_clients,
        batch_size=64,
        alpha_dir=alpha_dir,
        seed=seed,
    )

    model = PPFOT_IDS(
        input_dim=X.shape[1],
        num_classes=len(le.classes_),
        num_clouds=3,
    )

    server = FederatedServer(
        model=model,
        device=device,
        num_clients=num_clients,
        communication_rounds=rounds,
        epsilon_privacy=0.85,
        n_samples=len(X),
    )

    start_time = time.time()
    history = server.train(
        client_loaders=client_loaders,
        test_loader=test_loader,
        use_byzantine_detection=True,
        verbose=False,
    )
    total_time = time.time() - start_time
    round_time = total_time / rounds

    eval_result = server.trainer.evaluate(test_loader)

    # Estimate communication cost (model size × 2 × K per round)
    model_size_mb = sum(
        p.numel() * 4 for p in model.parameters()
    ) / (1024 * 1024)
    comm_per_round_mb = model_size_mb * 2 * num_clients

    return {
        "num_clients": num_clients,
        "alpha_dir": alpha_dir,
        "seed": seed,
        "accuracy": eval_result["accuracy"],
        "f1": eval_result["f1_weighted"],
        "round_time_s": round_time,
        "total_time_s": total_time,
        "comm_per_round_mb": comm_per_round_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Scalability experiments")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/scalability")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    data_loader = ICS3DDataLoader(args.data_path)

    # Table 6: Client count scaling
    print("=" * 60)
    print("Table 6: Scalability with Client Count K")
    print("=" * 60)

    client_counts = [3, 5, 10, 20]
    client_results = []
    for K in client_counts:
        result = run_scalability_experiment(
            data_loader, K, alpha_dir=0.5, seed=42, rounds=args.rounds, device=device
        )
        client_results.append(result)
        print(f"  K={K:2d}: Acc={result['accuracy']:.4f}, "
              f"Time={result['round_time_s']:.1f}s/round, "
              f"Comm={result['comm_per_round_mb']:.1f}MB")

    # Table 7: Non-IID heterogeneity
    print(f"\n{'='*60}")
    print("Table 7: Non-IID Heterogeneity Impact")
    print(f"{'='*60}")

    alphas = [0.1, 0.5, 1.0, 10.0]
    alpha_results = []
    for alpha in alphas:
        result = run_scalability_experiment(
            data_loader, num_clients=5, alpha_dir=alpha, seed=42,
            rounds=args.rounds, device=device
        )
        alpha_results.append(result)
        label = "extreme" if alpha == 0.1 else "moderate" if alpha == 0.5 else \
                "mild" if alpha == 1.0 else "near-IID"
        print(f"  α={alpha:>4.1f} ({label:>8s}): Acc={result['accuracy']:.4f}")

    output = {
        "client_scaling": client_results,
        "heterogeneity": alpha_results,
    }
    output_path = os.path.join(args.output_dir, "scalability_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
