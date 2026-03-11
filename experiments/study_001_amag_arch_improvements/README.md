# Study 001: AMAG Architecture Improvements

## Research Question

Can we improve AMAG's forecasting accuracy by addressing three structural limitations in the standard TE → SI → TR pipeline?

The AMAG paper (NeurIPS 2023) proposes a clean three-stage architecture, but our analysis identified specific architectural bottlenecks:
1. Single-hop spatial propagation limits information flow between distant brain regions
2. The TR module receives redundant temporal information already present in SI output
3. Temporal and spatial processing are completely decoupled, missing cross-domain interactions

## Background

AMAG (Additive, Multiplicative and Adaptive Graph Neural Network) is the prior state-of-the-art for this competition. It achieves strong results through its three-module design:
- **Temporal Encoder (TE)**: Per-channel GRU encodes temporal dynamics independently
- **Spatial Interaction (SI)**: Graph message passing (additive + multiplicative + adaptive) mixes channel information
- **Temporal Readout (TR)**: Per-channel GRU decodes spatial embeddings into future predictions

Our replication (see `replication/amag/`) achieved:
- Affi: MSE = 0.01410 (R² = 0.763)
- Beignet: MSE = 0.01497 (R² = 0.665)

## Shared Code

All experiments in this study share a single Colab notebook:
**`notebooks/colab_amag_experiments.ipynb`**

The notebook runs all experiments sequentially on Google Cloud GPUs, saving checkpoints and results to Google Drive. It includes OOM fallback logic and checkpoint resume support for session recovery.

## Experiments

| # | Name | Architectural Change | Affi MSE | Beignet MSE | vs Baseline (Affi) | vs Baseline (Beignet) |
|---|------|---------------------|----------|-------------|-------------------|---------------------|
| baseline | Control | Standard AMAG (TE → SI → TR) | 0.014137 | 0.015402 | — | — |
| 001 | Multi-hop SI | 2-hop SI with learned gating | 0.014017 | 0.015012 | **-0.85%** | **-2.53%** |
| 002 | Delta TR | Feed (Z-H) to TR, init h0 from TE | 0.014267 | 0.015761 | +0.92% | +2.33% |
| 003 | Interleaved + Aux | GRUCell+SI per step, aux next-step loss | **0.013880** | 0.016167 | **-1.82%** | +4.97% |

## Shared Configuration

See [config.yaml](config.yaml) for full hyperparameters. Key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 64 | Matches AMAG paper |
| lr | 5e-4 | Adam optimizer |
| weight_decay | 1e-5 | |
| n_epochs | 500 | Max epochs |
| patience | 100 | Early stopping |
| batch_size | 32 | (16 for OOM fallback) |
| val_fraction | 0.15 | Train/val split |
| seed | 42 | Reproducibility |
| lr_decay | 0.95 | StepLR gamma |
| lr_decay_every | 50 | StepLR step size |
| grad_clip | 1.0 | Max gradient norm |

## Conclusions

1. **Multi-hop SI (Exp 1) is the safest improvement** — consistent gains on both monkeys (-0.85% Affi, -2.53% Beignet). Should be incorporated into the standard architecture.

2. **Delta TR (Exp 2) is a failure** — removing temporal context from TR input hurts performance. The redundancy in the TE → SI → TR pipeline acts as a useful residual connection.

3. **Interleaved + Aux (Exp 3) is polarizing** — best single-monkey result on Affi (-1.82%) but worst on Beignet (+4.97%). Promising for high-channel-count subjects but needs tuning for smaller networks.

4. **Key insight**: The AMAG pipeline benefits from information redundancy between modules. Approaches that preserve or enhance redundancy (multi-hop) work better than those that remove it (delta TR).

### Recommended next experiments:
- Combine multi-hop (Exp 1) with standard AMAG as the new baseline
- Try interleaved without aux loss (isolate the contribution of interleaving vs aux loss)
- Try lower aux loss weights (0.01, 0.05) for Beignet
