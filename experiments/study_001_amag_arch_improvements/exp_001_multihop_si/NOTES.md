# Experiment 001: Multi-hop Spatial Interaction

**Study**: study_001_amag_arch_improvements
**Code**: `notebooks/colab_amag_experiments.ipynb` (Section 4: Experiment 1)
**Date**: 2026-03-02
**Status**: Running on Colab

## Problem
AMAG's SI module is single-hop: each channel aggregates messages only from its direct neighbors. In cortical motor circuits, information flows through relay paths (e.g., DLPFC → PM → M1), requiring multi-hop propagation to capture indirect dependencies. Single-hop SI cannot represent these indirect pathways.

## Change
Added a second SI pass (weight-tied with the first) and a learned gating parameter:

```python
Z1 = self.si(H)              # first hop — direct neighbors
Z2 = self.si(Z1)             # second hop — 2-hop paths (weight-tied)
alpha = sigmoid(self.hop2_alpha)  # learned blend weight
Z = (1 - alpha) * Z1 + alpha * Z2
```

The gating parameter `hop2_alpha` is initialized to 0 (sigmoid → 0.5), allowing the model to learn whether the second hop helps. Weight tying prevents parameter explosion.

## Hypothesis
Multi-hop propagation should improve predictions for channels that depend on distant brain regions (2+ edges away in the functional connectivity graph). We expect:
- Modest improvement on Affi (239 channels, more distant pairs)
- Smaller improvement on Beignet (89 channels, denser connectivity)
- The learned alpha should converge above 0.3 if 2-hop paths are useful

## Results

| Monkey | Val MSE | Epochs | vs Baseline |
|--------|---------|--------|-------------|
| Affi | 0.014017 | 140 | **-0.85%** |
| Beignet | 0.015012 | 163 | **-2.53%** |

## Analysis
Multi-hop SI improved both monkeys consistently. The gain is larger on Beignet (-2.53%) than Affi (-0.85%), which is opposite to our hypothesis (we expected Affi's larger channel count to benefit more). This suggests that Beignet's denser connectivity graph may have more useful 2-hop paths proportionally.

The model took similar or slightly more epochs to converge, suggesting the additional capacity is being utilized rather than causing instability.

## Key Takeaways
- 2-hop propagation provides consistent improvement on both monkeys.
- Beignet benefits more than Affi — surprising, worth investigating the learned alpha value.
- This is the only experiment that improved both monkeys.

## Next Steps
- If alpha > 0.3: try 3-hop or hierarchical pooling
- If alpha ≈ 0: single-hop may be sufficient; focus effort elsewhere
- Examine per-channel improvements — do distant channel pairs benefit most?
