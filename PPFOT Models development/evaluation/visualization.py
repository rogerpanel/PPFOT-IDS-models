"""
Visualisation Suite
====================
Generates all figures from the manuscript:
  - Training convergence curves (loss, accuracy, F1)
  - Domain adaptation comparison (bar charts)
  - Privacy-utility trade-off (line plot)
  - Adversarial robustness (grouped bars)
  - Byzantine resilience (degradation curves)
  - Per-class F1 heatmap
  - Privacy budget consumption over rounds
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ResultsPlotter:
    """Generate publication-quality figures.

    Parameters
    ----------
    save_dir : str
        Directory for saving figures (default "results/figures").
    dpi : int
        Figure resolution (default 300).
    """

    def __init__(self, save_dir: str = "results/figures", dpi: int = 300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)

        # Use a clean style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.dpi": dpi,
        })

    def _save(self, fig, name: str):
        path = os.path.join(self.save_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ----------------------------------------------------------------
    # Training convergence
    # ----------------------------------------------------------------

    def plot_training_curves(self, history: dict, filename: str = "training_convergence"):
        """Plot training loss, test accuracy, F1, and ε over rounds."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        rounds = history.get("round", list(range(len(history.get("train_loss", [])))))

        # Loss
        if "train_loss" in history:
            axes[0, 0].plot(rounds, history["train_loss"], "b-", linewidth=1.5)
            axes[0, 0].set_xlabel("Communication Round")
            axes[0, 0].set_ylabel("Training Loss")
            axes[0, 0].set_title("Training Loss")

        # Accuracy
        if "test_accuracy" in history:
            axes[0, 1].plot(rounds, history["test_accuracy"], "g-", linewidth=1.5)
            axes[0, 1].set_xlabel("Communication Round")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Test Accuracy")

        # F1
        if "test_f1" in history:
            axes[1, 0].plot(rounds, history["test_f1"], "r-", linewidth=1.5)
            axes[1, 0].set_xlabel("Communication Round")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].set_title("Weighted F1 Score")

        # Privacy budget
        if "epsilon_spent" in history:
            axes[1, 1].plot(rounds, history["epsilon_spent"], "m-", linewidth=1.5)
            axes[1, 1].axhline(y=0.85, color="k", linestyle="--", label="Target ε=0.85")
            axes[1, 1].set_xlabel("Communication Round")
            axes[1, 1].set_ylabel("ε spent")
            axes[1, 1].set_title("Privacy Budget Consumption")
            axes[1, 1].legend()

        fig.suptitle("PPFOT-IDS Training Convergence", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Cross-domain comparison  (Table 2)
    # ----------------------------------------------------------------

    def plot_cross_domain_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str = "cross_domain_comparison",
    ):
        """Bar chart comparing methods across domain transfer scenarios."""
        methods = list(results.keys())
        scenarios = list(results[methods[0]].keys())

        x = np.arange(len(scenarios))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, method in enumerate(methods):
            values = [results[method][s] for s in scenarios]
            ax.bar(x + i * width, values, width, label=method)

        ax.set_xlabel("Transfer Scenario")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Cross-Domain Detection Accuracy (Table 2)")
        ax.set_xticks(x + width * len(methods) / 2)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim([50, 100])

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Byzantine robustness  (Table 3)
    # ----------------------------------------------------------------

    def plot_byzantine_robustness(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str = "byzantine_robustness",
    ):
        """Line plot: accuracy vs Byzantine fraction for each method."""
        fig, ax = plt.subplots(figsize=(8, 5))

        byz_fractions = sorted(
            [k for k in list(results.values())[0].keys()], key=float
        )

        for method, values in results.items():
            accs = [values[f] for f in byz_fractions]
            marker = "o" if "PPFOT" in method else "s"
            linewidth = 2.5 if "PPFOT" in method else 1.5
            ax.plot(
                [float(f) for f in byz_fractions],
                accs,
                marker=marker,
                linewidth=linewidth,
                label=method,
            )

        ax.set_xlabel("Byzantine Fraction")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Byzantine Robustness (Table 3)")
        ax.legend()

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Privacy-utility trade-off  (Table 5)
    # ----------------------------------------------------------------

    def plot_privacy_utility(
        self,
        epsilons: List[float],
        accuracies: List[float],
        f1_scores: Optional[List[float]] = None,
        filename: str = "privacy_utility_tradeoff",
    ):
        """Line plot of accuracy/F1 vs privacy budget ε."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(epsilons, accuracies, "bo-", linewidth=2, markersize=8, label="Accuracy")
        if f1_scores:
            ax.plot(epsilons, f1_scores, "rs--", linewidth=2, markersize=8, label="Macro-F1")

        ax.axvline(x=0.85, color="green", linestyle=":", alpha=0.7, label="ε*=0.85 (ours)")
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Performance (%)")
        ax.set_title("Privacy-Utility Trade-off (Table 5)")
        ax.legend()

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Per-class F1 heatmap  (Supplementary Table D)
    # ----------------------------------------------------------------

    def plot_per_class_heatmap(
        self,
        per_class_results: Dict[str, Dict[str, float]],
        filename: str = "per_class_f1_heatmap",
    ):
        """Heatmap of per-class F1 across methods."""
        methods = list(per_class_results.keys())
        classes = list(per_class_results[methods[0]].keys())

        data = np.array([
            [per_class_results[m][c] for c in classes] for m in methods
        ])

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            data,
            annot=True,
            fmt=".1f",
            xticklabels=classes,
            yticklabels=methods,
            cmap="YlOrRd",
            vmin=30,
            vmax=100,
            ax=ax,
        )
        ax.set_title("Per-Class F1 Scores (%) — Supplementary Table D")

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Adversarial robustness  (Section IV-E)
    # ----------------------------------------------------------------

    def plot_adversarial_robustness(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str = "adversarial_robustness",
    ):
        """Grouped bar chart for clean vs adversarial accuracy."""
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(results.keys())
        accuracies = [results[k]["accuracy"] for k in labels]

        colors = ["#2ecc71" if "clean" in k else "#e74c3c" if "pgd" in k else "#3498db"
                   for k in labels]

        ax.bar(range(len(labels)), accuracies, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Adversarial Robustness Evaluation")
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------------
    # Scalability  (Table 6)
    # ----------------------------------------------------------------

    def plot_scalability(
        self,
        client_counts: List[int],
        accuracies: List[float],
        round_times: List[float],
        filename: str = "scalability",
    ):
        """Dual-axis plot: accuracy and round time vs number of clients."""
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(client_counts, accuracies, "bo-", linewidth=2, label="Accuracy")
        ax1.set_xlabel("Number of Clients (K)")
        ax1.set_ylabel("Accuracy (%)", color="b")

        ax2 = ax1.twinx()
        ax2.plot(client_counts, round_times, "rs--", linewidth=2, label="Round Time")
        ax2.set_ylabel("Round Time (s)", color="r")

        fig.legend(loc="upper center", ncol=2)
        ax1.set_title("Scalability Analysis (Table 6)")

        plt.tight_layout()
        self._save(fig, filename)
