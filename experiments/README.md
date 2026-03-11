# Experiments

Each experiment lives in its own subdirectory and is self-contained.

## Convention

```
experiments/
  exp_NNN_description/
    README.md       — hypothesis, method, results
    config.yaml     — hyperparameters
    model.py        — model code (if custom)
    train.py        — training script
    checkpoints/    — saved weights
    results/        — metrics, logs, plots
```

## Workflow

1. Create `experiments/exp_NNN_description/`
2. Write hypothesis in README.md
3. Run experiment, save checkpoints and metrics
4. Document results and analysis in README.md
