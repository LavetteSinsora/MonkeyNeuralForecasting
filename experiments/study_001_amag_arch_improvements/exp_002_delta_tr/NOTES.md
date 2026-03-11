# Experiment 002: Delta Temporal Readout

**Study**: study_001_amag_arch_improvements
**Code**: `notebooks/colab_amag_experiments.ipynb` (Section 5: Experiment 2)
**Date**: 2026-03-02
**Status**: Complete

## Problem
The standard AMAG pipeline feeds Z (SI output) directly to the TR GRU. But Z already contains temporal context from TE (Z = β₁·H + β₂·FC(add) + β₃·FC(mul)), so the TR GRU receives redundant temporal information. This forces TR to first separate "what SI added" from "what TE already encoded" before it can decode future predictions.

## Change
Two modifications to the TR input:

1. **Delta input**: Feed `Z - H` (spatial correction only) instead of `Z` to the TR GRU. This isolates what SI learned, removing redundant temporal information.

2. **h₀ initialization**: Initialize TR's GRU hidden state from TE's final hidden state (`H[:, -1, :, :]`), giving TR the temporal context it needs without mixing it into the input signal.

```python
Delta = Z - H                    # spatial correction only
h_last = H[:, -1, :, :]          # TE's final state
pred = self.tr(Delta, h0=h_last) # TR starts from temporal context
```

## Hypothesis
By disentangling spatial corrections from temporal context, the TR GRU can focus purely on temporal extrapolation from a well-initialized starting point. This should:
- Reduce learning difficulty for TR (cleaner input signal)
- Improve gradient flow to SI (SI's contribution is directly visible as the input)
- Potentially speed up convergence

## Results

| Monkey | Val MSE | Epochs | vs Baseline |
|--------|---------|--------|-------------|
| Affi | 0.014267 | 153 | +0.92% (worse) |
| Beignet | 0.015761 | 163 | +2.33% (worse) |

## Analysis
Delta TR regressed on both monkeys. The hypothesis that removing redundant temporal info from the TR input would help was wrong. The TR GRU apparently *benefits* from receiving the full Z embedding (including temporal context from TE), not just the spatial correction.

Possible explanations:
- The TR GRU may use the overlap between Z and H as a form of skip connection / identity shortcut, which aids gradient flow.
- The "spatial correction only" signal (Z - H) may be too weak or noisy by itself to drive accurate predictions. The temporal backbone in Z provides an anchor.
- h₀ initialization from TE's last state may conflict with the delta input signal.

## Key Takeaways
- **Do not strip temporal context from TR input.** The TR GRU needs the full signal.
- The redundancy in the TE → SI → TR pipeline may actually serve as a useful inductive bias (residual-like connection).
- This experiment should be logged in failure_analysis.md.

## Next Steps
- Do NOT pursue this direction further.
- The finding suggests that explicit residual connections (Z = H + correction) may be more effective than delta-only input.
