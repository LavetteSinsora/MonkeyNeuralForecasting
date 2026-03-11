# AMAG Replication

**Paper**: "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network for Forecasting Neuron Activity" — NeurIPS 2023

## Architecture

```
Input: X (B, 10, C, 9)  — normalized LMP + 8 power bands

1. Temporal Encoding (TE): GRU per channel (weight-tied) → H (B, 10, C, 64)
2. Spatial Interaction (SI): Add + Mul graph message passing → Z (B, 10, C, 64)
   - Add: a_t^v = Σ_u S(u,v) * tanh(A_a) * h_t^u
   - Mul: m_t^v = Σ_u tanh(A_m) * (h_t^u ⊙ h_t^v)
   - S(u,v) = sigmoid(MLP([H^u, H^v]))  — sample-dependent adaptor
3. Temporal Readout (TR): GRU → linear → pred (B, 10, C)
```

## Results

| Monkey | Val MSE | Paper MSE | Improvement | R² | Corr | Best Epoch |
|--------|---------|-----------|-------------|-----|------|------------|
| Affi   | **0.01410** | 0.0144 | −2.1% | 0.714 | 0.844 | 44 |
| Beignet | **0.01497** | 0.0192 | −21.9% | 0.641 | 0.802 | 63 |

**Note**: Beignet's large improvement is likely from using 10 context steps vs paper's 5.

## Key Findings

- **Correlation-init is critical**: Random adjacency init drops R² by ~0.08
- **Affi channels are bimodal**: 49/239 channels have R² > 0.85 (likely M1/PM); 16 have R² < 0.50
- **Beignet overfits**: train MSE 0.0099 vs val 0.0179 by epoch 213; weight_decay 1e-5 is too low
- **Non-monotonic timestep MSE for Affi**: peaks at t=6, then drops — possibly beta oscillation artifact

## Usage

```bash
# Train both monkeys
python replications/amag/train.py

# Train single monkey (smoke test)
python replications/amag/train.py --monkey affi --epochs 2

# Evaluate
python replications/amag/evaluate.py --monkey both

# Ablation study
python replications/amag/ablation.py --monkey beignet
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | AMAG model with ablation flags |
| `components.py` | TE, SI, TR submodules |
| `train.py` | Training script with embedded Trainer |
| `evaluate.py` | Evaluation script |
| `ablation.py` | Ablation study runner |
| `config.yaml` | All hyperparameters |
| `checkpoints/` | Trained model weights |
| `results/` | Metrics, logs, plots |
