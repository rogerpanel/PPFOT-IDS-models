"""
Run All Experiments
====================
Master script to reproduce all results from the manuscript:

  Table 1:  Privacy budget composition methods
  Table 2:  Cross-domain detection accuracy
  Table 3:  Byzantine robustness
  Table 4:  Robust aggregation baselines
  Table 5:  Privacy parameter sensitivity
  Table 6:  Scalability with client count
  Table 7:  Non-IID heterogeneity impact
  Table 8:  Ablation study
  Table 9:  Zero-day attack detection
  Table 10: Privacy-utility-communication trade-off
  Table D:  Per-class F1 (supplementary)
  Table F:  δ sensitivity (supplementary)

Usage:
  python experiments/run_all.py --data-path /path/to/ics3d --device auto
"""

import argparse
import os
import subprocess
import sys
import time


EXPERIMENTS = [
    ("Cross-Domain Detection (Table 2)", "experiments/run_cross_domain.py"),
    ("Byzantine Robustness (Tables 3 & 4)", "experiments/run_byzantine.py"),
    ("Privacy Analysis (Tables 1, 5, F)", "experiments/run_privacy.py"),
    ("Ablation Study (Table 8)", "experiments/run_ablation.py"),
    ("Scalability (Tables 6 & 7)", "experiments/run_scalability.py"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all PPFOT-IDS experiments")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to ICS3D dataset")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Communication rounds (50 for quick, 100 for full)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("PPFOT-IDS: Full Experiment Reproduction Suite")
    print("Byzantine-Robust Federated IDS via Rényi-Private Optimal Transport")
    print("Anaedevha, Trofimov & Borodachev — IEEE Access 2026")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    total_start = time.time()

    for i, (name, script) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(EXPERIMENTS)}] {name}")
        print(f"{'='*70}")

        cmd = [
            sys.executable, script,
            "--rounds", str(args.rounds),
            "--device", args.device,
        ]
        if args.data_path:
            cmd.extend(["--data-path", args.data_path])

        start = time.time()
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        elapsed = time.time() - start

        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"\n  [{status}] {name} — {elapsed:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All experiments completed in {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
