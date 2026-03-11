"""
Evaluation metrics and model evaluation for MonkeyNeuralForecasting.

Primary competition metric: MSE on LMP (feature 0) predictions.
All metrics operate on arrays of shape (N, T, C).
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, Dict, Optional

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(pred: ArrayLike, target: ArrayLike) -> float:
    """Mean Squared Error — primary competition metric."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.mean((p - t) ** 2))


def rmse(pred: ArrayLike, target: ArrayLike) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(pred, target)))


def r2_score(pred: ArrayLike, target: ArrayLike) -> Dict[str, float]:
    """R² coefficient of determination. Returns dict with 'mean' and 'per_channel'."""
    p, t = _to_numpy(pred), _to_numpy(target)
    p_flat = p.reshape(-1, p.shape[-1])
    t_flat = t.reshape(-1, t.shape[-1])
    ss_res = np.sum((t_flat - p_flat) ** 2, axis=0)
    ss_tot = np.sum((t_flat - t_flat.mean(axis=0)) ** 2, axis=0)
    per_channel = 1.0 - ss_res / (ss_tot + 1e-12)
    return {'mean': float(per_channel.mean()), 'per_channel': per_channel}


def pearson_correlation(pred: ArrayLike, target: ArrayLike) -> Dict[str, float]:
    """Pearson correlation per channel. Returns dict with 'mean' and 'per_channel'."""
    p, t = _to_numpy(pred), _to_numpy(target)
    p_flat = p.reshape(-1, p.shape[-1])
    t_flat = t.reshape(-1, t.shape[-1])
    p_centered = p_flat - p_flat.mean(axis=0)
    t_centered = t_flat - t_flat.mean(axis=0)
    num = np.sum(p_centered * t_centered, axis=0)
    denom = np.sqrt(np.sum(p_centered ** 2, axis=0) * np.sum(t_centered ** 2, axis=0)) + 1e-12
    per_channel = num / denom
    return {'mean': float(per_channel.mean()), 'per_channel': per_channel}


def per_timestep_mse(pred: ArrayLike, target: ArrayLike) -> np.ndarray:
    """MSE at each prediction timestep, averaged over N and C. Returns (T,)."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return np.mean((p - t) ** 2, axis=(0, 2))


def per_channel_mse(pred: ArrayLike, target: ArrayLike) -> np.ndarray:
    """MSE per channel, averaged over N and T. Returns (C,)."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return np.mean((p - t) ** 2, axis=(0, 1))


def compute_all_metrics(pred: ArrayLike, target: ArrayLike) -> Dict:
    """Compute the full suite of metrics."""
    return {
        'mse': mse(pred, target),
        'rmse': rmse(pred, target),
        'r2': r2_score(pred, target),
        'correlation': pearson_correlation(pred, target),
        'per_timestep_mse': per_timestep_mse(pred, target),
        'per_channel_mse': per_channel_mse(pred, target),
    }


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: Optional[str] = None,
    norm_stats: Optional[Dict] = None,
) -> Dict:
    """Evaluate a model on a DataLoader. Reports normalized + raw-scale metrics."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    is_nn = isinstance(model, torch.nn.Module)
    if is_nn:
        model.eval()
        model = model.to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            if is_nn:
                X = X.to(device)
                pred = model(X)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
            else:
                pred = model.predict_batch(X.numpy())
            all_preds.append(np.asarray(pred))
            all_targets.append(Y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    results = compute_all_metrics(preds, targets)

    if norm_stats is not None:
        from .data import denormalize_lmp
        preds_raw = denormalize_lmp(preds, norm_stats)
        targets_raw = denormalize_lmp(targets, norm_stats)
        raw_mse_val = mse(preds_raw, targets_raw)
        raw_rmse_val = float(np.sqrt(raw_mse_val))
        raw_pts = np.sqrt(per_timestep_mse(preds_raw, targets_raw))
        results['raw_mse_uv2'] = raw_mse_val
        results['raw_rmse_uv'] = raw_rmse_val
        results['raw_per_timestep_rmse_uv'] = raw_pts

    return results


def print_results(results: Dict, prefix: str = ''):
    """Pretty-print evaluation results."""
    indent = f"[{prefix}] " if prefix else ''
    print(f"{indent}-- Normalized scale (competition metric) --")
    print(f"{indent}  MSE (norm):       {results['mse']:.6f}")
    print(f"{indent}  RMSE (norm):      {results['rmse']:.6f}")
    print(f"{indent}  R² (mean):        {results['r2']['mean']:.4f}")
    print(f"{indent}  Corr (mean):      {results['correlation']['mean']:.4f}")
    ts_mse = results['per_timestep_mse']
    print(f"{indent}  MSE@t=0: {ts_mse[0]:.6f}   MSE@t=9: {ts_mse[-1]:.6f}")
    if 'raw_rmse_uv' in results:
        pts = results['raw_per_timestep_rmse_uv']
        print(f"{indent}-- Raw scale (uV) --")
        print(f"{indent}  RMSE (uV):        {results['raw_rmse_uv']:.1f} uV")
        print(f"{indent}  MSE (uV²):        {results['raw_mse_uv2']:.0f} uV²")
        print(f"{indent}  RMSE@t=0: {pts[0]:.1f} uV   RMSE@t=9: {pts[-1]:.1f} uV")


def save_results(results: Dict, path: str):
    """Save evaluation results to JSON (handles numpy arrays)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj
    with open(path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
