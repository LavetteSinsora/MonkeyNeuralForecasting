# Study 002: Faithful AMAG Replication — Multi-Step Architecture

## Research Question

Can faithfully implementing the paper's **multi-step Transformer architecture** (instead of
the one-step GRU variant) match or exceed the paper's reported R² (0.763 Affi, 0.665 Beignet)?

Our current replication (study baseline) uses GRU for TE and TR, which corresponds to
**AMAG-G** (GRU one-step-like), not the **AMAG-T** (Transformer multi-step) used for
the paper's Table 2 results.

---

## Key Finding from Paper

> **Section 4.2**: "For one-step prediction, we use GRU architecture for TE and TR, and
> for multi-step prediction, **Transformers are used for TE and TR**."

The paper's Table 2 results (Affi MSE=0.0144, Beignet MSE=0.0192) use AMAG-T, not AMAG-G.

---

## Architectural Discrepancies Fixed in This Study

| Aspect | Paper (multi-step) | Our Previous Code | Fixed Here |
|--------|--------------------|-------------------|------------|
| TE module | Transformer decoder with PE + LN | GRU | ✅ AMAG-T variant |
| TR module | Transformer (simultaneous) | GRU single-shot | ✅ AMAG-T variant |
| GRU TR (AMAG-G) | Autoregressive (feed-back) | Single FC projection | ✅ AMAG-G-AR variant |
| Adaptor MLP | 4-layer [128, 128, 256, 64] → 1 | 2-layer [128, 64] → 1 | ✅ `adaptor_depth=4` |
| Context steps | 5 (paper exact) | 10 | ✅ AMAG-T-5 variant |

---

## Experiment Variants

### exp_gru_ar — AMAG-G Autoregressive
- Model: AMAGReplica with `readout_mode=autoregressive`
- Data: 10 context steps (same as baseline)
- Fix: autoregressive decoding vs single-shot FC projection
- Expected: moderate R² improvement, better waveform dynamics

### exp_transformer_5ctx — AMAG-T (Paper Exact)
- Model: AMAGTransformer
- Data: 5 context steps (paper's exact setting, full 20-step input with 15 masked)
- Fix: full Transformer TE+TR architecture
- Expected: approach paper R² values (0.763 / 0.665)

### exp_transformer_10ctx — AMAG-T (Competition Setting)
- Model: AMAGTransformer
- Data: 10 context steps (our competition setting, full 20-step input with 10 masked)
- Fix: Transformer architecture with more context
- Expected: potentially exceed paper MSE (more context helps)

---

## Results

| Experiment | Architecture | Affi Val MSE | Beignet Val MSE | Affi R² | Beignet R² | Notes |
|------------|--------------|-------------|-----------------|---------|------------|-------|
| baseline (study_001) | AMAG-GRU direct, 10ctx | 0.014137 | 0.015402 | 0.714 | ~0.64 | Previous best |
| exp_gru_ar | AMAG-G autoregressive | TBD | TBD | TBD | TBD | |
| exp_transformer_5ctx | AMAG-T, 5ctx | TBD | TBD | TBD | TBD | Paper target: R²=0.763/0.665 |
| exp_transformer_10ctx | AMAG-T, 10ctx | TBD | TBD | TBD | TBD | |
| **Paper AMAG-T** | Transformer | **0.0144** | **0.0192** | **0.763** | **0.665** | Target |

---

## How to Run

```bash
# From repo root:

# Train GRU autoregressive variant
python experiments/study_002_faithful_replication/train.py --variant gru_ar

# Train Transformer 5-context-step variant (paper exact)
python experiments/study_002_faithful_replication/train.py --variant transformer_5ctx

# Train Transformer 10-context-step variant (competition setting)
python experiments/study_002_faithful_replication/train.py --variant transformer_10ctx

# Train all variants
python experiments/study_002_faithful_replication/train.py --variant all

# Smoke test (2 epochs)
python experiments/study_002_faithful_replication/train.py --variant gru_ar --epochs 2
```

---

## Architecture Details

### TransformerTemporalEncoder
```
Input: (B, T=20, C, 9) → mask future positions to 0
FC: 9 → 64
+ SinusoidalPE (max_len=32)
2× TransformerEncoderLayer:
  - MultiHeadAttention(d=64, heads=4)
  - FFN(64 → 256 → 64)
  - LayerNorm (pre-norm)
Output: (B, T=20, C, 64)
```

### TransformerTemporalReadout
```
Same structure as TransformerTE
Input: (B, T=20, C, 64) from SI
Extract positions [context_steps : context_steps+pred_steps]
Linear(64, 1) → LMP scalar
Output: (B, T_pred=10, C)
```

### Adaptor MLP (4-layer, paper spec)
```
Linear(128, 128) + ReLU
Linear(128, 256) + ReLU
Linear(256, 64) + ReLU
Linear(64, 1)     → scalar weight S_{uv}
```

---

## Parameter Count Verification

Paper reports ~0.27M parameters. Expected breakdown:
- TransformerTE: ~0.08M
- SI (adaptor_depth=4): ~0.12M (Affi, 239 channels)
- TransformerTR: ~0.08M
- Total: ~0.28M (close to paper's 0.27M)
