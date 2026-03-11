# Baseline: Standard AMAG

**Study**: study_001_amag_arch_improvements
**Code**: `notebooks/colab_amag_experiments.ipynb` (Section 3: Baseline)
**Date**: 2026-03-02
**Status**: Running on Colab

## Problem
N/A — this is the control run for comparison.

## Change
None. Standard AMAG architecture: TE (per-channel GRU) → SI (additive + multiplicative + adaptive graph) → TR (per-channel GRU → linear).

Adjacency matrices initialized from LMP correlation matrix. All ablation flags at defaults (use_add=True, use_mul=True, use_adaptor=True, learnable_adj=True).

## Hypothesis
Establishes the performance floor for this study. Expected to match our prior replication results (~0.0141 Affi, ~0.0150 Beignet) within noise from different random splits and training dynamics.

## Results

| Monkey | Val MSE | Epochs | Notes |
|--------|---------|--------|-------|
| Affi | 0.014137 | 140 | Early stopped; matches prior replication |
| Beignet | 0.015402 | 146 | Slightly higher than prior (0.01497), within noise |

## Analysis
Results match expectations — close to our prior replication numbers (Affi 0.01410, Beignet 0.01497). Small differences are expected from different random val splits and training dynamics.

## Key Takeaways
- Baseline is a valid control for comparison with the three experiments.
- Both monkeys converged in ~140-146 epochs, well within the 500 epoch budget.
