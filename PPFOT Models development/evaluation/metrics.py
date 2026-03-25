"""
Evaluation Metrics
===================
Comprehensive metrics matching all tables in the manuscript:
  - Accuracy, Weighted F1, Macro-F1
  - Per-class F1 (Table D in supplementary)
  - ROC-AUC (one-vs-rest for multi-class)
  - Precision-Recall curves

Statistical testing: paired t-test with Bonferroni correction (p < 0.05).
All experiments repeated over 5 random seeds with mean ± std reported.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy import stats


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all metrics reported in Tables 2-10 of the manuscript.

    Returns
    -------
    dict with keys: accuracy, f1_weighted, f1_macro, per_class_f1,
                    precision_macro, recall_macro, roc_auc (if probabilities given).
    """
    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    results["precision_macro"] = float(np.mean(precision))
    results["recall_macro"] = float(np.mean(recall))

    # Per-class F1 (Table D)
    n_classes = len(np.unique(y_true))
    per_class = {}
    for i in range(len(f1_per)):
        label = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per[i]),
            "support": int(support[i]),
        }
    results["per_class"] = per_class

    # ROC-AUC (one-vs-rest)
    if y_prob is not None and y_prob.ndim == 2:
        try:
            results["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        except ValueError:
            results["roc_auc_ovr"] = 0.0

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return results


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Per-class F1 scores (Supplementary Table D)."""
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(f1_per))]
    return {name: float(f) for name, f in zip(class_names, f1_per)}


def statistical_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
    bonferroni_n: int = 1,
) -> Dict[str, float]:
    """Paired t-test with Bonferroni correction.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Metric scores across random seeds.
    alpha : float
        Significance level (default 0.05).
    bonferroni_n : int
        Number of comparisons for Bonferroni correction.

    Returns
    -------
    dict with t_statistic, p_value, significant, corrected_alpha.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    corrected_alpha = alpha / bonferroni_n
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < corrected_alpha,
        "corrected_alpha": corrected_alpha,
        "mean_a": float(np.mean(scores_a)),
        "std_a": float(np.std(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "std_b": float(np.std(scores_b)),
    }


def multi_seed_summary(
    seed_results: List[Dict[str, float]],
    metric_key: str = "accuracy",
) -> Dict[str, float]:
    """Summarise results across multiple seeds (mean ± std)."""
    values = [r[metric_key] for r in seed_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": len(values),
    }
