"""
Experiment 4: Ablation Study  (Table 8)
========================================
Systematically ablates each component:
  1. Full PPFOT-IDS (baseline)
  2. w/o Adaptive Sinkhorn → fixed ε regularisation
  3. w/o Byzantine Detection → standard FedAvg aggregation
  4. w/o Differential Privacy → no noise addition
  5. w/o Spectral Normalisation → unconstrained Lipschitz
  6. Entropic OT only → no federated learning
  7. Federated Learning only → no OT alignment
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import ICS3DDataLoader, create_federated_loaders
from models.ppfot_ids import PPFOT_IDS
from training.federated import FederatedServer
from evaluation.metrics import compute_all_metrics


ABLATION_CONFIGS = {
    "full_ppfot_ids": {
        "use_byzantine": True,
        "epsilon_privacy": 0.85,
        "lambda_ot": 0.1,
        "use_spectral_norm": True,
    },
    "wo_byzantine_detection": {
        "use_byzantine": False,
        "epsilon_privacy": 0.85,
        "lambda_ot": 0.1,
        "use_spectral_norm": True,
    },
    "wo_differential_privacy": {
        "use_byzantine": True,
        "epsilon_privacy": 1000.0,  # effectively no DP
        "lambda_ot": 0.1,
        "use_spectral_norm": True,
    },
    "wo_spectral_norm": {
        "use_byzantine": True,
        "epsilon_privacy": 0.85,
        "lambda_ot": 0.1,
        "use_spectral_norm": False,
    },
    "wo_ot_alignment": {
        "use_byzantine": True,
        "epsilon_privacy": 0.85,
        "lambda_ot": 0.0,  # no OT loss
        "use_spectral_norm": True,
    },
    "federated_only": {
        "use_byzantine": False,
        "epsilon_privacy": 1000.0,
        "lambda_ot": 0.0,
        "use_spectral_norm": False,
    },
}


def run_ablation(
    config_name: str,
    config: dict,
    data_loader: ICS3DDataLoader,
    seed: int = 42,
    rounds: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run a single ablation configuration."""
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
        use_spectral_norm=config["use_spectral_norm"],
    )

    server = FederatedServer(
        model=model,
        device=device,
        num_clients=5,
        communication_rounds=rounds,
        epsilon_privacy=config["epsilon_privacy"],
        lambda_ot=config["lambda_ot"],
        n_samples=len(X),
    )

    history = server.train(
        client_loaders=client_loaders,
        test_loader=test_loader,
        use_byzantine_detection=config["use_byzantine"],
        verbose=False,
    )

    eval_result = server.trainer.evaluate(test_loader)

    return {
        "config": config_name,
        "seed": seed,
        "accuracy": eval_result["accuracy"],
        "f1": eval_result["f1_weighted"],
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study experiments")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    data_loader = ICS3DDataLoader(args.data_path)
    seeds = [42, 123, 456, 789, 1024]

    all_results = {}
    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {config_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            result = run_ablation(config_name, config, data_loader, seed, args.rounds, device)
            seed_results.append(result)
            print(f"  Seed {seed}: Acc={result['accuracy']:.4f}")

        accs = [r["accuracy"] for r in seed_results]
        print(f"  Mean: {np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}%")

        all_results[config_name] = seed_results

    output_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
