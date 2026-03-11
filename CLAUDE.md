# MonkeyNeuralForecasting — Agent Instructions

## Project Overview

**Task**: Multivariate time-series forecasting of μECoG neural signals from rhesus macaques during reaching tasks.
**Metric**: Mean Squared Error (MSE, lower is better) on LMP (feature 0) predictions.
**Goal**: Research-grade investigation of neural forecasting architectures, with AMAG (NeurIPS 2023) as the baseline.

---

## Data Format

All data in `dataset/` as `.npz` files. Load with `np.load(path, allow_pickle=True)['data']`.

**Shape**: `(N_samples, 20, N_channels, 9_features)`

| Dimension | Meaning |
|-----------|---------|
| N | Number of trials |
| 20 | Timesteps at 30ms each = 600ms total |
| C | Channels: 239 (Affi) or 89 (Beignet) |
| 9 | Features: LMP + 8 power bands |

**Input/Target split**: Timesteps 0–9 are model input. Timesteps 10–19 are the prediction target.

```
Input:  data[:, 0:10, :, :]   → (N, 10, C, 9)
Target: data[:, 10:20, :, 0]  → (N, 10, C), LMP only
```

**Important**: Timesteps 10–19 contain **real recorded data** for all 9 features. They are NOT masked or repeats. However, they must NOT be used as model input at inference — only as auxiliary training targets.

**Features**:
- `feature[0]` = LMP (Local Motor Potential) — prediction target, signed voltage, range ~±15,000 μV
- `feature[1–8]` = Power bands (0.5–400 Hz) — always non-negative

**Monkeys**:
- Affi: 239 channels, 985 train / 162 test trials, spans M1/PM/FEF/SMA/DLPFC
- Beignet: 89 channels, 700 train / 158 test trials, M1 only

---

## Directory Structure

```
MonkeyNeuralForecastingClaude/
├── CLAUDE.md                           # This file
├── requirements.txt                    # Python dependencies
├── NeurIPS-2023-amag-*.pdf             # AMAG reference paper
├── .gitignore
│
├── dataset/                            # Raw .npz data files
├── human_notes/                        # Human's personal notes (do not modify)
│
├── utils/                              # Shared utilities (2 files)
│   ├── data.py                         # Data loading, normalization, dataloaders
│   └── metrics.py                      # MSE, R², evaluation, print/save results
│
├── replications/                       # Method replications (self-contained)
│   └── amag/                           # AMAG baseline replication
│       ├── model.py, components.py     # Model code
│       ├── train.py, evaluate.py       # Scripts
│       ├── ablation.py                 # Ablation study
│       ├── config.yaml                 # Hyperparameters
│       ├── checkpoints/                # Trained weights
│       └── results/                    # Metrics, logs, plots
│
├── experiments/                        # Research experiments
│   └── study_001_amag_arch_improvements/  # Completed: multi-hop SI, delta TR, interleaved
│
├── dashboard/                          # Streamlit interactive tools
│   ├── app.py                          # Main entry (streamlit run dashboard/app.py)
│   └── pages/                          # Training, Results, Diagnosis pages
│
└── notebooks/                          # Jupyter notebooks (exploration)
```

---

## Quick Start

```python
# Load data
from utils.data import get_dataloaders
train_loader, val_loader, stats = get_dataloaders('affi', batch_size=32)

# Build and evaluate a model
from replications.amag.model import build_model
from utils.metrics import evaluate_model
model = build_model('affi', {'hidden_size': 64, 'use_adaptor': True, 'compute_init_corr': True})
results = evaluate_model(model, val_loader)
```

---

## Current Best Results

| Monkey | Val MSE | Paper MSE | Improvement |
|--------|---------|-----------|-------------|
| Affi   | 0.01410 | 0.0144    | −2.1%       |
| Beignet| 0.01497 | 0.0192    | −21.9%      |

---

## Key Rules

- **Never use test data for training/validation** — private .npz files are test-only
- **Never use timesteps 10–19 as model input at inference** — they are prediction targets
- **Normalize using training set statistics only** — no test-set leakage
- **Do not modify `human_notes/`** — human's personal notes
- **Each replication is self-contained** — model code, training, evaluation all within its directory
- **`utils/` has only shared essentials** — data loading and metrics that must be consistent across methods
