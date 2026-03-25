"""
Experiment 3: Privacy Analysis  (Tables 1, 5, and Supplementary Table F)
=========================================================================
Evaluates:
  1. Privacy budget under different composition methods (Table 1)
  2. Accuracy vs ε ∈ {0.1, 0.3, 0.5, 0.85, 1.0, 2.0, ∞} (Table 5)
  3. δ sensitivity: δ₀ ∈ {10⁻³, 10⁻⁵, 10⁻⁷} (Table F)
  4. Membership inference attack success rate
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import ICS3DDataLoader, align_feature_dims, create_federated_loaders
from models.ppfot_ids import PPFOT_IDS
from training.federated import FederatedServer
from evaluation.privacy_audit import PrivacyAuditor
from evaluation.metrics import compute_all_metrics


def run_privacy_sensitivity(
    data_loader: ICS3DDataLoader,
    epsilon: float,
    seed: int = 42,
    rounds: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train with a specific privacy budget and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = data_loader.load_edge_iiot("DNN")
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    client_loaders, test_loader, _, _ = create_federated_loaders(
        X, y_enc.astype(str), num_clients=5, batch_size=64, seed=seed,
    )

    model = PPFOT_IDS(
        input_dim=X.shape[1],
        num_classes=len(le.classes_),
        num_clouds=3,
    )

    server = FederatedServer(
        model=model,
        device=device,
        num_clients=5,
        communication_rounds=rounds,
        epsilon_privacy=epsilon,
        delta_privacy=5e-4,
        n_samples=len(X),
    )

    history = server.train(
        client_loaders=client_loaders,
        test_loader=test_loader,
        use_byzantine_detection=True,
        verbose=False,
    )

    eval_result = server.trainer.evaluate(test_loader)

    # MIA
    auditor = PrivacyAuditor(
        model=server.global_model,
        device=device,
        epsilon=epsilon,
        n_samples=len(X),
    )

    # Use first client loader as "train" proxy
    first_loader = list(client_loaders.values())[0]
    mia_result = auditor.membership_inference_attack(first_loader, test_loader)

    return {
        "epsilon": epsilon,
        "seed": seed,
        "accuracy": eval_result["accuracy"],
        "f1": eval_result["f1_weighted"],
        "mia_accuracy": mia_result["attack_accuracy"],
        "confidence_gap": mia_result["confidence_gap"],
    }


def main():
    parser = argparse.ArgumentParser(description="Privacy sensitivity experiments")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/privacy")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    data_loader = ICS3DDataLoader(args.data_path)

    # Table 1: Composition methods comparison
    print("=" * 60)
    print("Table 1: Privacy Budget Composition Methods")
    print("=" * 60)

    model_dummy = PPFOT_IDS(input_dim=61, num_classes=7, num_clouds=3)
    auditor = PrivacyAuditor(model_dummy, device, n_samples=160000)
    composition_results = auditor.compute_privacy_budget(n_rounds=50)
    for method, result in composition_results.items():
        print(f"  {method:>12s}: ε = {result['epsilon']:.2f} ({result['reference']})")

    # Table 5: ε sensitivity
    print(f"\n{'='*60}")
    print("Table 5: Privacy Parameter Sensitivity")
    print(f"{'='*60}")

    epsilons = [0.1, 0.3, 0.5, 0.85, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1024]
    sensitivity_results = []

    for eps in epsilons:
        print(f"\nε = {eps}:")
        for seed in seeds:
            result = run_privacy_sensitivity(
                data_loader, eps, seed, args.rounds, device
            )
            sensitivity_results.append(result)
            print(f"  Seed {seed}: Acc={result['accuracy']:.4f}, MIA={result['mia_accuracy']:.4f}")

    # Table F: δ sensitivity
    print(f"\n{'='*60}")
    print("Table F: δ Sensitivity Analysis")
    print(f"{'='*60}")

    delta_results = auditor.delta_sensitivity_report()
    for r in delta_results:
        print(f"  δ₀={r['delta_per_round']:.0e} → δ_total={r['delta_total']:.0e}, σ²={r['sigma_squared']:.3e}")

    # Save all results
    output = {
        "composition_methods": composition_results,
        "epsilon_sensitivity": sensitivity_results,
        "delta_sensitivity": delta_results,
    }
    output_path = os.path.join(args.output_dir, "privacy_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
