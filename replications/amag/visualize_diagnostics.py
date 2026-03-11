#!/usr/bin/env python3
"""
AMAG Replication — Diagnostic Visualization

Loads existing checkpoints, runs predictions on validation data, and plots
predicted vs ground truth waveforms for channels with high, medium, and low R².

Usage:
    python replications/amag/visualize_diagnostics.py
    python replications/amag/visualize_diagnostics.py --monkey affi --n-channels 15
    python replications/amag/visualize_diagnostics.py --monkey beignet --n-channels 10
"""

import sys
import os
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data import get_dataloaders
from utils.metrics import evaluate_model, r2_score
from replications.amag.model import build_model


def load_checkpoint(monkey: str, config: dict):
    """Load best checkpoint for a given monkey."""
    ckpt_path = os.path.join(_here, 'checkpoints', monkey, 'best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = build_model(monkey, config['model'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    norm_stats = state.get('norm_stats', None)
    return model, norm_stats


def get_predictions_and_targets(model, val_loader, device):
    """Collect all predictions and targets from validation loader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in val_loader:
            X = X.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(Y.numpy())
    preds = np.concatenate(all_preds, axis=0)     # (N, T_pred, C)
    targets = np.concatenate(all_targets, axis=0)  # (N, T_pred, C)
    return preds, targets


def plot_waveform_grid(preds, targets, channel_indices, channel_labels, title, save_path,
                       n_cols=3, timestep_ms=30):
    """Plot predicted vs ground truth waveforms for selected channels."""
    n_channels = len(channel_indices)
    n_rows = (n_channels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    T_pred = preds.shape[1]
    t = np.arange(T_pred) * timestep_ms  # milliseconds

    for plot_idx, ch in enumerate(channel_indices):
        ax = axes[plot_idx]
        pred_ch = preds[:, :, ch]    # (N, T_pred)
        tgt_ch = targets[:, :, ch]  # (N, T_pred)

        # Plot mean ± std across trials
        pred_mean = pred_ch.mean(0)
        pred_std = pred_ch.std(0)
        tgt_mean = tgt_ch.mean(0)
        tgt_std = tgt_ch.std(0)

        ax.fill_between(t, pred_mean - pred_std, pred_mean + pred_std,
                        alpha=0.2, color='steelblue')
        ax.fill_between(t, tgt_mean - tgt_std, tgt_mean + tgt_std,
                        alpha=0.2, color='firebrick')
        ax.plot(t, pred_mean, color='steelblue', linewidth=1.5, label='Predicted')
        ax.plot(t, tgt_mean, color='firebrick', linewidth=1.5, linestyle='--', label='Ground Truth')

        label = channel_labels[plot_idx]
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel('Norm LMP', fontsize=8)
        ax.tick_params(labelsize=7)
        if plot_idx == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_r2_distribution(per_channel_r2, monkey, save_path):
    """Plot histogram of per-channel R² values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_channel_r2, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(per_channel_r2), color='firebrick', linestyle='--',
               linewidth=2, label=f'Mean R²={np.mean(per_channel_r2):.3f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('R²', fontsize=11)
    ax.set_ylabel('# Channels', fontsize=11)
    ax.set_title(f'{monkey.capitalize()} — Per-Channel R² Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_timestep_mse(preds, targets, monkey, save_path, timestep_ms=30):
    """Plot MSE at each prediction timestep."""
    T_pred = preds.shape[1]
    per_t_mse = np.mean((preds - targets) ** 2, axis=(0, 2))  # (T_pred,)
    t = np.arange(1, T_pred + 1) * timestep_ms

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, per_t_mse, marker='o', color='steelblue', linewidth=2, markersize=5)
    ax.fill_between(t, 0, per_t_mse, alpha=0.15, color='steelblue')
    ax.set_xlabel('Prediction Horizon (ms)', fontsize=11)
    ax.set_ylabel('MSE (normalized)', fontsize=11)
    ax.set_title(f'{monkey.capitalize()} — MSE by Prediction Step', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scatter_pred_vs_true(preds, targets, monkey, save_path, max_samples=2000):
    """Scatter plot of predicted vs ground truth values (random sample)."""
    flat_pred = preds.reshape(-1)
    flat_true = targets.reshape(-1)

    if len(flat_pred) > max_samples:
        idx = np.random.default_rng(42).choice(len(flat_pred), max_samples, replace=False)
        flat_pred = flat_pred[idx]
        flat_true = flat_true[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(flat_true, flat_pred, alpha=0.2, s=4, color='steelblue', rasterized=True)
    lim = max(abs(flat_true).max(), abs(flat_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='Perfect prediction')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Ground Truth (normalized)', fontsize=11)
    ax.set_ylabel('Predicted (normalized)', fontsize=11)
    ax.set_title(f'{monkey.capitalize()} — Pred vs True (scatter)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def diagnose_monkey(monkey: str, config: dict, n_display_channels: int = 15):
    """Run full diagnostics for a single monkey."""
    print(f"\n{'='*60}")
    print(f"  Diagnostics — {monkey.upper()}")
    print('='*60)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}")

    # Load model
    model, norm_stats = load_checkpoint(monkey, config)
    model = model.to(device)
    info = model.get_model_info()
    print(f"  Model: {info['name']} | {info['n_params_M']}M params")

    # Load validation data
    data_cfg = config['data']
    _, val_loader, stats = get_dataloaders(
        monkey, batch_size=64,
        val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
    )
    if norm_stats is None:
        norm_stats = stats
    n_val = len(val_loader.dataset)
    print(f"  Validation samples: {n_val}")

    # Get predictions
    preds, targets = get_predictions_and_targets(model, val_loader, device)
    print(f"  Predictions shape: {preds.shape} | Targets shape: {targets.shape}")

    # Compute per-channel R²
    N, T_pred, C = preds.shape
    per_ch_r2 = r2_score(
        preds.reshape(N * T_pred, C),
        targets.reshape(N * T_pred, C)
    )['per_channel']  # (C,)
    per_ch_mse = np.mean((preds - targets) ** 2, axis=(0, 1))  # (C,)

    mean_r2 = float(np.mean(per_ch_r2))
    mean_mse = float(np.mean(per_ch_mse))
    overall_mse = float(np.mean((preds - targets) ** 2))
    print(f"  Val MSE: {overall_mse:.5f} | Mean R²: {mean_r2:.4f}")
    print(f"  R² range: [{per_ch_r2.min():.3f}, {per_ch_r2.max():.3f}]")
    print(f"  Channels with R²>0.8: {(per_ch_r2 > 0.8).sum()}/{C}")
    print(f"  Channels with R²<0.3: {(per_ch_r2 < 0.3).sum()}/{C}")

    # Output directory
    diag_dir = os.path.join(_here, 'results', 'diagnostics', monkey)
    os.makedirs(diag_dir, exist_ok=True)

    # 1. R² distribution histogram
    plot_r2_distribution(
        per_ch_r2, monkey,
        save_path=os.path.join(diag_dir, 'r2_distribution.png')
    )

    # 2. Per-timestep MSE
    plot_per_timestep_mse(
        preds, targets, monkey,
        save_path=os.path.join(diag_dir, 'per_timestep_mse.png')
    )

    # 3. Scatter pred vs true
    plot_scatter_pred_vs_true(
        preds, targets, monkey,
        save_path=os.path.join(diag_dir, 'scatter_pred_vs_true.png')
    )

    # 4. Waveform plots for high/medium/low R² channels
    sorted_ch = np.argsort(per_ch_r2)[::-1]  # descending

    n_each = n_display_channels // 3
    high_r2_channels = sorted_ch[:n_each].tolist()
    low_r2_channels = sorted_ch[-n_each:].tolist()
    # Medium: pick from middle
    mid_start = C // 2 - n_each // 2
    mid_channels = sorted_ch[mid_start:mid_start + n_each].tolist()

    for group_name, ch_list in [('high_r2', high_r2_channels),
                                 ('medium_r2', mid_channels),
                                 ('low_r2', low_r2_channels)]:
        labels = [f"ch{ch} R²={per_ch_r2[ch]:.3f}" for ch in ch_list]
        plot_waveform_grid(
            preds, targets,
            channel_indices=ch_list,
            channel_labels=labels,
            title=f"{monkey.capitalize()} — {group_name.replace('_', ' ').title()} Channels",
            save_path=os.path.join(diag_dir, f'waveforms_{group_name}.png'),
        )

    # 5. Print summary table
    print(f"\n  Top-5 channels by R²:")
    for ch in sorted_ch[:5]:
        print(f"    ch{ch:3d}: R²={per_ch_r2[ch]:.4f}, MSE={per_ch_mse[ch]:.5f}")
    print(f"\n  Bottom-5 channels by R²:")
    for ch in sorted_ch[-5:]:
        print(f"    ch{ch:3d}: R²={per_ch_r2[ch]:.4f}, MSE={per_ch_mse[ch]:.5f}")

    return {
        'monkey': monkey,
        'val_mse': overall_mse,
        'mean_r2': mean_r2,
        'per_channel_r2': per_ch_r2.tolist(),
        'per_channel_mse': per_ch_mse.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='AMAG Diagnostic Visualization')
    parser.add_argument('--config', default=os.path.join(_here, 'config.yaml'))
    parser.add_argument('--monkey', choices=['affi', 'beignet', 'both'], default='both')
    parser.add_argument('--n-channels', type=int, default=15,
                        help='Total channels to display (split equally into high/med/low R²)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]
    for monkey in monkeys:
        try:
            diagnose_monkey(monkey, config, n_display_channels=args.n_channels)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")
            print(f"  Run training first: python replications/amag/train.py --monkey {monkey}")

    print(f"\n  Diagnostics saved to: {os.path.join(_here, 'results', 'diagnostics')}/")


if __name__ == '__main__':
    main()
