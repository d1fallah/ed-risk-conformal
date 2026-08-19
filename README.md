# 🏥 Conformal Risk Prediction for Emergency Deterioration — GRU-D + ECI on MIMIC-IV-ED

> **Part of the SihhaDZ research project** — An AI-powered smart emergency platform for the Algerian healthcare system.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![ICLR 2025](https://img.shields.io/badge/ECI-ICLR%202025-brightgreen)](https://arxiv.org/abs/2502.00818)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: MIMIC-IV-ED](https://img.shields.io/badge/Dataset-MIMIC--IV--ED-blueviolet)](https://physionet.org/content/mimic-iv-ed/2.2/)

---

## 🎯 What This Is

This module implements a **real-time clinical deterioration risk prediction pipeline** for emergency department patients, built on top of the [`mimic4ed-benchmark`](https://github.com/nliulab/mimic4ed-benchmark) baseline and extended with two key contributions:

1. **GRU-D** (Gated Recurrent Unit with Decay) — handles irregular, missing-value-rich vital-sign streams natively, matching real ED monitoring conditions.
2. **Error-Quantified Conformal Inference (ECI, ICLR 2025)** — replaces standard split-conformal calibration to produce adaptive, coverage-guaranteed prediction intervals that account for temporal dependence in time-series data.

The result: a system that predicts whether a patient will deteriorate within a chosen horizon (1h / 6h / 12h) and communicates **how uncertain** that prediction is — critical for clinical decision support.

---

## 🔬 Research Contribution

| Component | Description |
|-----------|-------------|
| **Baseline** | `mimic4ed-benchmark` — NEWS2 & standard ML on MIMIC-IV-ED |
| **Core model** | GRU-D on vital-sign time series |
| **Uncertainty** | ECI (ICLR 2025) vs. MAPIE comparison |
| **Ablations** | ECI vs. MAPIE · Horizon 1h/6h/12h · Logit vs. clipped calibration |
| **Metric** | Empirical coverage + interval width on held-out test set |

### Why ECI over standard conformal?

Standard split-conformal calibration assumes exchangeability — a condition violated in temporal clinical data (patient state evolves, confounders shift over time). ECI uses error-quantified bounds that remain valid under this temporal dependence, yielding tighter intervals in stable periods and appropriately wider ones during deterioration episodes.

---

## 📁 Repository Structure

```
module2/
├── mimic4ed-benchmark-main/    # Vendored baseline (submodule / external reference)
├── CAREBench/                  # Reference benchmark (submodule)
├── MDS-ED/                     # Reference multi-dataset ED benchmark (submodule)
│   └── src/                    # Preprocessing utilities adapted for this work
├── Error-quantified-Conformal-Inference/   # ECI implementation (submodule)
│   └── core/
│       ├── methods.py          # Core ECI calibration methods
│       ├── ar.py               # Autoregressive error modeling
│       └── quantile.py         # Quantile regression utilities
└── README.md
```

> **Note:** Baseline repositories are vendored/referenced but not modified. All novel code lives in this module's own scripts and integrations.

---

## 🏗️ Architecture Overview

```
Raw MIMIC-IV-ED Vitals & Labs
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  ← mimic4ed-benchmark pipeline
│  (imputation,     │
│   normalization)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     GRU-D         │  ← Handles irregular intervals + missingness
│  (decay gates on  │     natively; no mean-imputation approximation
│   missing values) │
└────────┬──────────┘
         │  risk score ∈ [0,1]
         ▼
┌────────────────────────────┐
│  Conformal Calibration     │
│  ┌──────────┐ ┌─────────┐  │
│  │  MAPIE   │ │   ECI   │  │  ← Primary contribution (ICLR 2025)
│  │ baseline │ │ (ICLR25)│  │     adaptive intervals under temporal
│  └──────────┘ └─────────┘  │     dependence
└────────────────────────────┘
         │
         ▼
  Risk score + prediction interval
  (guaranteed empirical coverage)
         │
         ▼
  ┌──────────────┐
  │  NEWS2       │  ← Heuristic baseline comparison
  │  comparison  │
  └──────────────┘
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- MIMIC-IV-ED access ([PhysioNet credentialed access required](https://physionet.org/content/mimic-iv-ed/2.2/))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mimic4ed-critical-outcome-prediction
cd mimic4ed-critical-outcome-prediction

# Install dependencies
pip install -r mimic4ed-benchmark-main/requirements.txt
pip install -r Error-quantified-Conformal-Inference/requirements.txt
pip install torch torchvision  # GRU-D backbone
```

### Data

This repository does **not** include any patient data. You must obtain MIMIC-IV-ED access through [PhysioNet](https://physionet.org/) (credentialed, requires training completion).

Place the downloaded data in `data/mimic-iv-ed/` (gitignored).

---

## 🚀 Reproducing Results

```bash
# Step 1 — Run mimic4ed-benchmark baseline (NEWS2 + standard ML)
python mimic4ed-benchmark-main/reproduce_baseline.py

# Step 2 — Train GRU-D model
python scripts/train_grud.py --horizon 6h --seed 42

# Step 3 — Calibrate with MAPIE (baseline uncertainty)
python scripts/calibrate_mapie.py --model outputs/grud_6h.pt

# Step 4 — Calibrate with ECI (primary contribution)
python scripts/calibrate_eci.py --model outputs/grud_6h.pt

# Step 5 — Run ablations (all horizons, both calibrators)
python scripts/run_ablations.py
```

All scripts use fixed seeds for reproducibility. Results are written to `results/`.

---

## 📊 Outcome Definition (Gate-Approved)

The deterioration event, prediction origin, horizon h, observation window, exclusion criteria, and cohort inclusion criteria are formally defined and supervisor-approved in:

📄 **[`M2_Outcome_Definition_Gate.md`](M2_Outcome_Definition_Gate.md)**

This gate document must be read before interpreting any results — all metrics are computed identically across NEWS2, GRU-D, MAPIE, and ECI per its specification.

---

## 📚 References

- **mimic4ed-benchmark:** Xie et al., *Benchmarking Emergency Department Prediction Models with Machine Learning and Large Language Models*, Nature Medicine 2024.
- **GRU-D:** Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 2018.
- **ECI (ICLR 2025):** *Error-Quantified Conformal Inference for Time Series*, arXiv:2502.00818.
- **MAPIE:** Taquet et al., *MAPIE: an open-source library for distribution-free uncertainty quantification*, arXiv:2207.12274.
- **MDS-ED:** Reyna et al., *Rethinking Clinical Predictions*, 2023.
- **CAREBench:** Choi et al., *CAREBench*, 2024.
- **MIMIC-IV-ED:** Johnson et al., PhysioNet 2023.

---

## ⚠️ Data & Ethics Notice

This work uses the MIMIC-IV-ED dataset, which contains de-identified patient data from Beth Israel Deaconess Medical Center. Access requires:
- Completion of CITI "Data or Specimens Only Research" training
- Signed PhysioNet credentialed data use agreement

**No patient data is included in this repository.**

---

## 🔗 Related Modules

This is Module 2 of the SihhaDZ research stack:

| Module | Topic |
|--------|-------|
| M1 | Secure Homomorphic Patient Profiling (CKKS) |
| **M2** | **← You are here** — Conformal Risk Prediction (GRU-D + ECI) |
| M3 | Federated Anomaly Detection for IoT (DP + FLARE) |
| M4 | Hospital Resource Forecasting (N-BEATSx + NCC) |
| M5 | Multi-Agent Ambulance Routing (MAPPO + GAT) |
