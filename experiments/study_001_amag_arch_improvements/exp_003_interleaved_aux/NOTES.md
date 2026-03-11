# Experiment 003: Interleaved TE-SI with Auxiliary Loss

**Study**: study_001_amag_arch_improvements
**Code**: `notebooks/colab_amag_experiments.ipynb` (Section 6: Experiment 3)
**Date**: 2026-03-02
**Status**: Complete

## Problem
In standard AMAG, temporal encoding (TE) processes each channel in complete isolation — the GRU hidden state at timestep t contains zero spatial information from other channels. Spatial mixing only happens after all temporal processing is complete. This means the temporal encoder cannot adapt its representations based on what neighboring channels are doing, missing potentially useful cross-channel temporal patterns.

Additionally, the SI module only receives gradient signal from the final prediction loss (10 steps into the future), which is a weak and indirect training signal for learning spatial interactions.

## Change
Two coupled modifications:

1. **Interleaved TE-SI**: Replace the sequential TE→SI pipeline with a step-by-step GRUCell that interleaves temporal and spatial processing at every timestep:
   ```python
   for t in range(T):
       h_t = GRUCell(x_t, h_{t-1})      # temporal update
       z_t = SI(h_t)                      # spatial mixing
       h_t = h_t + gate * tanh(W(z_t - h_t))  # gated residual feedback
   ```
   The gated residual allows spatial information to influence the GRU hidden state without destroying temporal patterns.

2. **Auxiliary next-step prediction loss**: Add a per-timestep training signal for SI:
   ```python
   L_aux = Σ_{t=0}^{8} ||Decoder(z_t) - x_{t+1}||²
   L_total = L_forecast + 0.1 * L_aux
   ```
   This gives SI a direct, local training signal: "can your spatially-mixed embedding predict the next input?" Motivated by predictive coding theory in neuroscience.

## Hypothesis
Interleaving should allow the temporal encoder to build spatially-informed representations from the start. The auxiliary loss should:
- Provide per-timestep gradient signal to SI (instead of only from final prediction)
- Act as regularization (encouraging generally useful spatial representations)
- Speed up SI learning (more frequent, local feedback)

This is the most ambitious experiment — it changes the core computation graph. We expect either a clear improvement or useful failure modes to analyze.

## Results

| Monkey | Val MSE | Epochs | vs Baseline |
|--------|---------|--------|-------------|
| Affi | 0.013880 | 157 | **-1.82%** (best overall) |
| Beignet | 0.016167 | 138 | +4.97% (worst overall) |

## Analysis
The most ambitious experiment produced the most polarized results. On Affi (239 channels), interleaving with auxiliary loss achieved the **best MSE of any model** (0.01388). On Beignet (89 channels), it was the **worst performer** (0.01617).

Possible explanations for the divergence:
- **Affi benefits from richer spatial context**: With 239 channels, the interleaved GRUCell receives useful spatial corrections at each timestep. The auxiliary loss provides meaningful per-step gradients to SI.
- **Beignet's smaller graph may not need per-step spatial mixing**: With only 89 channels, the single-pass SI may already capture sufficient spatial structure. The additional complexity of interleaving may cause overfitting or interfere with temporal encoding.
- **Aux loss weight (0.1) may need monkey-specific tuning**: The auxiliary loss may be too strong for Beignet, pulling the model toward short-horizon prediction at the expense of the full 10-step forecast.

The early stopping at epoch 138 on Beignet (vs 157 on Affi) suggests the model may have started overfitting earlier.

## Key Takeaways
- Interleaved TE-SI is **highly promising for high-channel-count subjects** (Affi).
- The approach may need architectural or hyperparameter adjustments for smaller channel counts.
- The auxiliary loss weight should be treated as a hyperparameter, possibly monkey-specific.

## Next Steps
- Try interleaved without aux loss (ablation: is interleaving alone helpful?)
- Try aux loss with lower weight (0.01, 0.05) for Beignet
- Try combining multi-hop (Exp 1) with interleaving (Exp 3)
- Examine the learned gate parameter (alpha) — how much spatial feedback was incorporated?
