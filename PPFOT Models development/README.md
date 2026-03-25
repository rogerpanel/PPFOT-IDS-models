# PPFOT-IDS: Byzantine-Robust Federated Intrusion Detection via Rényi-Private Optimal Transport

**Authors:** Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev
**Affiliation:** National Research Nuclear University MEPhI, Moscow
**Venue:** IEEE Access (under review)

---

## Overview

This repository contains the complete implementation of **PPFOT-IDS** — a privacy-preserving federated intrusion detection system that integrates:

1. **Optimal Transport** — Entropic-regularised Sinkhorn with adaptive ε-scheduling for cross-cloud domain alignment (O(n²) vs O(n³) complexity, 15–23× speedup)
2. **Rényi Differential Privacy** — Calibrated Gaussian mechanism with RDP composition achieving (ε=0.85, δ=5×10⁻⁴) — 5× tighter than basic composition
3. **Byzantine-Robust Aggregation** — Wasserstein-space outlier detection tolerating up to 40% malicious participants

### Key Results

| Metric | Value |
|--------|-------|
| Cross-cloud detection accuracy | **94.2%** (vs 78.3% FedAvg) |
| Byzantine resilience (40% adversaries) | **87.1%** (7.1 pt drop vs 26.8 pt FedAvg) |
| Privacy guarantee | **ε=0.85**, δ=5×10⁻⁴ (Rényi DP) |
| Inference latency | **2.9 ms/sample** |
| Communication cost | **12.1 MB/round** (3.7× reduction) |

---

## Repository Structure

```
PPFOT-IDS/
├── configs/
│   └── default.yaml              # All hyperparameters (Tables matching manuscript)
├── core/                          # Core algorithmic components
│   ├── optimal_transport.py       # Adaptive Sinkhorn (Algorithm 2), Eqs. 1–4
│   ├── privacy.py                 # Gaussian mechanism, RDP accountant, Eqs. 5–7, 11
│   ├── byzantine.py               # Byzantine detection (Algorithm 1), Eq. 8
│   └── spectral_norm.py           # Spectral normalisation, Eqs. 9–10
├── models/                        # Neural network architectures
│   ├── feature_extractor.py       # Shared: Input→256→128→64 + BN + SN
│   ├── classifier.py              # Head: 128→64→num_classes
│   ├── transport_map.py           # Kantorovich dual potentials f_φ, g_ψ
│   └── ppfot_ids.py               # Full PPFOT-IDS model with cloud adapters
├── training/                      # Training infrastructure
│   ├── trainer.py                 # Local training loop + evaluation
│   └── federated.py               # Federated server with robust aggregation
├── evaluation/                    # Evaluation suite
│   ├── metrics.py                 # Accuracy, F1, per-class, statistical tests
│   ├── privacy_audit.py           # MIA, budget tracking, ε/δ sensitivity
│   ├── adversarial.py             # FGSM, PGD attacks, certified bounds
│   └── visualization.py           # Publication-quality figures
├── experiments/                   # Reproducible experiment scripts
│   ├── run_cross_domain.py        # Table 2: Cross-domain scenarios
│   ├── run_byzantine.py           # Tables 3 & 4: Byzantine robustness
│   ├── run_privacy.py             # Tables 1, 5, F: Privacy analysis
│   ├── run_ablation.py            # Table 8: Ablation study
│   ├── run_scalability.py         # Tables 6 & 7: Client count & non-IID
│   └── run_all.py                 # Master script for full reproduction
├── scripts/
│   ├── download_data.sh           # Dataset download
│   └── reproduce_results.sh       # One-command full reproduction
├── configs/default.yaml           # All hyperparameters
├── requirements.txt
├── setup.py
└── LICENSE                        # MIT
```

---

## Dataset

**Integrated Cloud Security 3Datasets (ICS3D)**
[Kaggle: rogernickanaedevha/integrated-cloud-security-3datasets-ics3d](https://kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d)
License: CC BY-NC-SA 4.0

| Dataset | Samples | Features | Domain |
|---------|---------|----------|--------|
| Containers Dataset | 157,329 flows | 78 | Kubernetes CVE security (10 CVE types + benign) |
| Edge-IIoTset (DNN) | 236,748 | 61 | IoT/IIoT 7-layer architecture |
| Edge-IIoTset (ML) | 187,562 | 48 | IoT/IIoT (ML variant) |
| Microsoft GUIDE (train) | 589,437 incidents | 33 entity types | Enterprise SOC, 441 MITRE ATT&CK techniques |
| Microsoft GUIDE (test) | 147,359 incidents | — | Enterprise SOC |

**Attack categories:** Benign, DoS/DDoS, Reconnaissance, Injection, MitM, Malware, CVE-specific

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
bash scripts/download_data.sh
```

### 3. Run All Experiments

```bash
# Full reproduction (all tables in the paper)
bash scripts/reproduce_results.sh --device cuda --rounds 100

# Or run individual experiments:
python experiments/run_cross_domain.py --data-path /path/to/ics3d --device cuda
python experiments/run_byzantine.py --data-path /path/to/ics3d --device cuda
python experiments/run_privacy.py --data-path /path/to/ics3d --device cuda
python experiments/run_ablation.py --data-path /path/to/ics3d --device cuda
python experiments/run_scalability.py --data-path /path/to/ics3d --device cuda
```

---

## Mapping to Manuscript

### Algorithms
| Algorithm | Implementation | Description |
|-----------|---------------|-------------|
| Algorithm 1 | `core/byzantine.py:ByzantineDetector` | Byzantine-robust transport plan aggregation |
| Algorithm 2 | `core/optimal_transport.py:AdaptiveSinkhorn` | Adaptive Sinkhorn with ε-scheduling |

### Key Equations
| Equation | File | Description |
|----------|------|-------------|
| Eq. 1–4 | `core/optimal_transport.py` | Wasserstein distance, entropic OT, Sinkhorn |
| Eq. 5 | `core/privacy.py:GaussianMechanism` | Gaussian noise calibration σ² = 2Δ²log(1.25/δ)/ε² |
| Eq. 6 | `core/privacy.py:RDPAccountant` | Moments accountant / RDP composition |
| Eq. 7 | `core/privacy.py` | Utility bound ‖γ*−γ̃*‖_F ≤ O(1/(√n·ε)) |
| Eq. 8 | `core/byzantine.py` | Byzantine convergence bound |
| Eq. 9 | `core/spectral_norm.py` | Spectral normalisation W_SN = W/σ(W) |
| Eq. 10 | `evaluation/adversarial.py` | Certified robustness bound |
| Eq. 11 | `core/privacy.py:privatise_histogram` | Noisy histogram h̃ = h + N(0,σ²I) |

### Experiment → Table Mapping
| Script | Tables Reproduced |
|--------|-------------------|
| `run_cross_domain.py` | Table 2, Table 9, Table D |
| `run_byzantine.py` | Table 3, Table 4 |
| `run_privacy.py` | Table 1, Table 5, Table 10, Table F |
| `run_ablation.py` | Table 8 |
| `run_scalability.py` | Table 6, Table 7 |

---

## Hyperparameters

All hyperparameters are specified in `configs/default.yaml` and match Section IV-B:

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Learning rate | 10⁻³ (cosine annealing) | Section IV-B |
| Batch size | 64 | Section IV-B |
| Local epochs | 5 | Section IV-B |
| Communication rounds | 100 | Section IV-B |
| Dropout | 0.2 | Section III-C |
| Gradient clipping | 1.0 | Section IV-B |
| Sinkhorn ε₀ → ε_min | 0.5 → 0.01 (ρ=0.9) | Algorithm 2 |
| Privacy ε (global) | 0.85 | Table 1 |
| Privacy δ (global) | 5×10⁻⁴ | Section IV-D |
| Byzantine tolerance q | 0.4 | Section IV-E |
| Threshold multiplier α | 2.0 | Algorithm 1 |

---

## Hardware Requirements

Experiments in the paper were run on:
- **GPU:** NVIDIA A100 (40 GB VRAM)
- **CPU:** 64-core AMD EPYC
- **RAM:** 512 GB
- **Software:** PyTorch 2.0, Python 3.10, POT 0.9

The codebase also supports CPU-only execution (slower).

---

## Statistical Rigour

- 5 random seeds per experiment (seeds: 42, 123, 456, 789, 1024)
- Results reported as mean ± standard deviation
- Significance: paired t-test, p < 0.05 with Bonferroni correction

---

## Citation

```bibtex
@article{anaedevha2026ppfot,
  title={Byzantine-Robust Federated Intrusion Detection via
         R\'{e}nyi-Private Optimal Transport},
  author={Anaedevha, Roger Nick and Trofimov, Alexander G.
          and Borodachev, Yuri V.},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

Dataset (ICS3D) is licensed under CC BY-NC-SA 4.0.
