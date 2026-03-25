"""
ICS3D Dataset Loader and Preprocessor
=====================================
Integrated Cloud Security 3-Datasets (ICS3D) from Kaggle:
  - Edge-IIoTset (IoT/IIoT security): 236,748 samples (DNN) / 187,562 (ML)
  - Containers Dataset (Kubernetes CVE flows): 157,329 samples
  - Microsoft GUIDE (enterprise SOC incidents): 589,437 train / 147,359 test

Preprocessing follows Section IV-A of the manuscript:
  1. Feature extraction (flow statistics, protocol features, behavioural patterns)
  2. Histogram discretisation into B=100 bins
  3. L1-normalisation to empirical distributions
  4. Winsorisation of outliers at [1%, 99%] quantiles
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import kagglehub
except ImportError:
    kagglehub = None


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class ICS3DDataLoader:
    """Loader for the Integrated Cloud Security 3-Datasets (ICS3D)."""

    KAGGLE_SLUG = "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d"

    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is not None:
            self.path = dataset_path
        elif kagglehub is not None:
            self.path = kagglehub.dataset_download(self.KAGGLE_SLUG)
        else:
            raise RuntimeError(
                "No dataset_path provided and kagglehub is not installed. "
                "Install with: pip install kagglehub"
            )
        print(f"[ICS3D] Dataset root: {self.path}")

    # ---- Edge-IIoTset ----

    def load_edge_iiot(self, variant: str = "DNN") -> Tuple[np.ndarray, np.ndarray]:
        """Load Edge-IIoTset dataset.

        Parameters
        ----------
        variant : str
            ``"DNN"`` (61 features, 236 748 samples) or
            ``"ML"`` (48 features, 187 562 samples).
        """
        filename = (
            "DNN-EdgeIIoT-dataset.csv"
            if variant.upper() == "DNN"
            else "ML-EdgeIIoT-dataset.csv"
        )
        df = pd.read_csv(os.path.join(self.path, filename), low_memory=False)
        return self._preprocess_edge_iiot(df)

    def _preprocess_edge_iiot(self, df: pd.DataFrame):
        df = df.replace([np.inf, -np.inf], np.nan)

        # Extract labels
        if "Attack_type" in df.columns:
            labels = df["Attack_type"].values
            df = df.drop(columns=["Attack_type"])
        elif "Label" in df.columns:
            labels = df["Label"].values
            df = df.drop(columns=["Label"])
        else:
            labels = np.zeros(len(df), dtype=str)

        # Numeric features only
        df = df.select_dtypes(include=[np.number])
        df = df.fillna(df.median())

        # Winsorise outliers at 1%/99%
        for col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

        return df.values.astype(np.float32), labels

    # ---- Containers / Kubernetes ----

    def load_containers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Containers Dataset (157 329 network flows, 78 features)."""
        df = pd.read_csv(
            os.path.join(self.path, "Containers_Dataset.csv"), low_memory=False
        )
        return self._preprocess_containers(df)

    def _preprocess_containers(self, df: pd.DataFrame):
        df = df.replace([np.inf, -np.inf], np.nan)

        if "Label" in df.columns:
            labels = df["Label"].values
            df = df.drop(columns=["Label"])
        else:
            labels = np.zeros(len(df), dtype=str)

        df = df.select_dtypes(include=[np.number])
        df = df.fillna(df.median())

        for col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

        return df.values.astype(np.float32), labels

    # ---- Microsoft GUIDE ----

    def load_microsoft_guide(
        self, split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load Microsoft GUIDE dataset.

        Parameters
        ----------
        split : str
            ``"train"`` (589 437 incidents) or ``"test"`` (147 359 incidents).
        """
        filename = (
            "Microsoft_GUIDE_Train.csv"
            if split == "train"
            else "Microsoft_GUIDE_Test.csv"
        )
        df = pd.read_csv(os.path.join(self.path, filename), low_memory=False)
        return self._preprocess_guide(df)

    def _preprocess_guide(self, df: pd.DataFrame):
        # Drop high-cardinality ID columns
        drop_cols = ["Id", "OrgId", "IncidentId", "AlertId", "DeviceId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        if "IncidentGrade" in df.columns:
            labels = df["IncidentGrade"].values
            df = df.drop(columns=["IncidentGrade"])
        else:
            labels = np.zeros(len(df), dtype=str)

        df = df.select_dtypes(include=[np.number])
        df = df.fillna(0)

        return df.values.astype(np.float32), labels


# ---------------------------------------------------------------------------
# Histogram discretisation  (Section IV-A, step 2)
# ---------------------------------------------------------------------------

def features_to_histogram(
    X: np.ndarray, n_bins: int = 100
) -> np.ndarray:
    """Discretise feature matrix into per-sample normalised histograms.

    Each feature column is binned independently; the concatenated bin counts
    are L1-normalised so they sum to 1 (empirical distribution).
    """
    histograms = []
    for i in range(X.shape[1]):
        col = X[:, i]
        bin_edges = np.linspace(col.min(), col.max(), n_bins + 1)
        indices = np.digitize(col, bin_edges[1:-1])  # 0 .. n_bins-1
        one_hot = np.eye(n_bins)[indices]
        histograms.append(one_hot)
    H = np.concatenate(histograms, axis=1).astype(np.float32)
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return H / row_sums


# ---------------------------------------------------------------------------
# Align feature dimensions across heterogeneous clouds
# ---------------------------------------------------------------------------

def align_feature_dims(
    *arrays: np.ndarray, target_dim: Optional[int] = None
) -> list:
    """Pad or truncate arrays so all share the same feature dimension."""
    if target_dim is None:
        target_dim = max(a.shape[1] for a in arrays)

    aligned = []
    for a in arrays:
        if a.shape[1] < target_dim:
            pad = np.zeros((a.shape[0], target_dim - a.shape[1]), dtype=a.dtype)
            aligned.append(np.concatenate([a, pad], axis=1))
        else:
            aligned.append(a[:, :target_dim])
    return aligned


# ---------------------------------------------------------------------------
# Federated non-IID partitioning  (Dirichlet distribution, Section IV-C)
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Partition sample indices across clients using Dirichlet(α).

    Lower α → more heterogeneous (non-IID) partitions.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    client_indices: Dict[int, list] = {k: [] for k in range(num_clients)}

    for c in classes:
        idx_c = np.where(labels == c)[0]
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idx_c)).astype(int)
        # Distribute remainder
        proportions[-1] = len(idx_c) - proportions[:-1].sum()
        splits = np.split(idx_c, np.cumsum(proportions)[:-1])
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    return {k: np.array(v) for k, v in client_indices.items()}


# ---------------------------------------------------------------------------
# Build DataLoaders for federated scenarios
# ---------------------------------------------------------------------------

def create_federated_loaders(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 5,
    batch_size: int = 64,
    alpha_dir: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """Create per-client train loaders and a shared test loader.

    Returns
    -------
    client_loaders : dict[int, DataLoader]
    test_loader : DataLoader
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Partition training data across clients
    partition = dirichlet_partition(y_train, num_clients, alpha=alpha_dir, seed=seed)

    client_loaders = {}
    for k, idx in partition.items():
        ds = TensorDataset(
            torch.from_numpy(X_train[idx]),
            torch.from_numpy(y_train[idx]).long(),
        )
        client_loaders[k] = DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=False
        )

    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader, le, scaler
