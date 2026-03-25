"""
Experiment 1: Cross-Domain Detection  (Table 2)
=================================================
Reproduces the three transfer scenarios from Section IV-C:
  Scenario 1: Container → IoT          (92.4%)
  Scenario 2: IoT → Enterprise         (89.7%)
  Scenario 3: Multi-Source → Container  (94.2%)

Each scenario is run 5× with different seeds; results are mean ± std.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import ICS3DDataLoader, align_feature_dims, create_federated_loaders
from models.ppfot_ids import PPFOT_IDS
from training.federated import FederatedServer
from evaluation.metrics import compute_all_metrics, multi_seed_summary


def run_scenario(
    scenario: str,
    data_loader: ICS3DDataLoader,
    seed: int = 42,
    rounds: int = 100,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run a single cross-domain scenario."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data per scenario
    if scenario == "container_to_iot":
        X_source, y_source = data_loader.load_containers()
        X_target, y_target = data_loader.load_edge_iiot("DNN")
    elif scenario == "iot_to_enterprise":
        X_source, y_source = data_loader.load_edge_iiot("ML")
        X_target, y_target = data_loader.load_microsoft_guide("train")
    elif scenario == "multi_to_container":
        X_iot, y_iot = data_loader.load_edge_iiot("DNN")
        X_ml, y_ml = data_loader.load_edge_iiot("ML")
        X_guide, y_guide = data_loader.load_microsoft_guide("train")
        X_target, y_target = data_loader.load_containers()

        # Align and merge sources
        X_iot, X_ml, X_guide, X_target = align_feature_dims(
            X_iot, X_ml, X_guide, X_target
        )
        X_source = np.concatenate([X_iot, X_ml, X_guide[:50000]], axis=0)
        y_source = np.concatenate([y_iot, y_ml, y_guide[:50000]])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if scenario != "multi_to_container":
        X_source, X_target = align_feature_dims(X_source, X_target)

    # Unify label encoding
    le = LabelEncoder()
    all_labels = np.concatenate([y_source, y_target]).astype(str)
    le.fit(all_labels)
    y_source_enc = le.transform(y_source.astype(str))
    y_target_enc = le.transform(y_target.astype(str))

    # Scale features
    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source).astype(np.float32)
    X_target = scaler.transform(X_target).astype(np.float32)

    input_dim = X_source.shape[1]
    num_classes = len(le.classes_)

    # Create federated loaders from source
    client_loaders, test_loader, _, _ = create_federated_loaders(
        X_source, y_source_enc.astype(str),
        num_clients=5, batch_size=64, alpha_dir=0.5, seed=seed,
    )

    # Target test loader
    from torch.utils.data import DataLoader, TensorDataset
    target_test = DataLoader(
        TensorDataset(
            torch.from_numpy(X_target[:20000]),
            torch.from_numpy(y_target_enc[:20000]).long(),
        ),
        batch_size=64, shuffle=False,
    )

    # Model
    model = PPFOT_IDS(
        input_dim=input_dim,
        num_classes=num_classes,
        num_clouds=3,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )

    # Federated server
    server = FederatedServer(
        model=model,
        device=device,
        num_clients=5,
        communication_rounds=rounds,
        local_epochs=5,
        lr=1e-3,
        lambda_ot=0.1,
        epsilon_privacy=0.85,
        delta_privacy=5e-4,
        n_samples=len(X_source),
    )

    # Train
    history = server.train(
        client_loaders=client_loaders,
        test_loader=target_test,
        target_loader=target_test,
        use_byzantine_detection=True,
        verbose=True,
    )

    # Final evaluation
    eval_results = server.trainer.evaluate(target_test)
    metrics = compute_all_metrics(
        eval_results["labels"],
        eval_results["predictions"],
        eval_results["probabilities"],
        class_names=list(le.classes_),
    )

    return {
        "scenario": scenario,
        "seed": seed,
        **{k: v for k, v in metrics.items() if k != "per_class"},
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-domain detection experiments")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/cross_domain")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto" else "cpu"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    data_loader = ICS3DDataLoader(args.data_path)

    scenarios = ["container_to_iot", "iot_to_enterprise", "multi_to_container"]
    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        seed_results = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            result = run_scenario(scenario, data_loader, seed, args.rounds, device)
            seed_results.append(result)
            print(f"  Accuracy: {result['accuracy']:.4f}, F1: {result['f1_weighted']:.4f}")

        summary = multi_seed_summary(seed_results, "accuracy")
        f1_summary = multi_seed_summary(seed_results, "f1_weighted")

        all_results[scenario] = {
            "accuracy": summary,
            "f1_weighted": f1_summary,
            "per_seed": seed_results,
        }

        print(f"\n  Summary: {summary['mean']*100:.1f} ± {summary['std']*100:.1f}%")

    # Save results
    output_path = os.path.join(args.output_dir, "cross_domain_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
