#!/usr/bin/env python3
"""
Study 002: Faithful AMAG Replication — Training Script

Trains and evaluates three architectural variants:
  gru_ar:           AMAG-G autoregressive GRU decoder (fixes single-shot readout)
  transformer_5ctx: AMAG-T Transformer TE+TR, 5 context steps (paper exact)
  transformer_10ctx: AMAG-T Transformer TE+TR, 10 context steps (competition setting)

Usage:
    python experiments/study_002_faithful_replication/train.py --variant gru_ar
    python experiments/study_002_faithful_replication/train.py --variant transformer_5ctx
    python experiments/study_002_faithful_replication/train.py --variant transformer_10ctx
    python experiments/study_002_faithful_replication/train.py --variant all
    python experiments/study_002_faithful_replication/train.py --variant gru_ar --epochs 2  # smoke test
    python experiments/study_002_faithful_replication/train.py --variant all --monkey affi
"""

import sys
import os
import argparse
import json
import time
import yaml

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import get_dataloaders, get_dataloaders_fullseq
from utils.metrics import evaluate_model, print_results, save_results
from replications.amag.model import build_model
from replications.amag.train import Trainer, detect_device


VARIANTS = ['gru_ar', 'transformer_5ctx', 'transformer_10ctx']


def train_variant(variant_name: str, monkey: str, config: dict,
                  override_epochs: int = None) -> dict:
    """Train a single variant for a single monkey."""
    var_cfg = config[variant_name]
    exp_cfg = config['experiment']

    data_cfg = var_cfg['data']
    model_cfg = var_cfg['model']
    train_cfg = var_cfg['training'].copy()

    temporal_module = model_cfg.get('temporal_module', 'gru')
    readout_mode = model_cfg.get('readout_mode', 'direct')
    context_steps = data_cfg.get('context_steps', 10)

    print(f"\n{'='*70}")
    print(f"  Study 002 | Variant: {variant_name} | Monkey: {monkey.upper()}")
    print(f"  temporal_module={temporal_module}, readout_mode={readout_mode}, "
          f"context_steps={context_steps}")
    print(f"  Target: MSE={exp_cfg[f'paper_mse_{monkey}']}, R²={exp_cfg[f'paper_r2_{monkey}']}")
    print('='*70)

    # Load data
    if temporal_module == 'transformer':
        train_loader, val_loader, stats = get_dataloaders_fullseq(
            monkey, context_steps=context_steps,
            batch_size=data_cfg['batch_size'],
            val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
        )
    else:
        train_loader, val_loader, stats = get_dataloaders(
            monkey, batch_size=data_cfg['batch_size'],
            val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
        )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Build model
    model = build_model(monkey, model_cfg)
    info = model.get_model_info()
    print(f"  Model: {info['name']} | {info['n_params_M']}M params")

    # Directories
    results_dir = os.path.join(_here, 'results', variant_name)
    ckpt_dir = os.path.join(_here, 'checkpoints', variant_name, monkey)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if override_epochs is not None:
        train_cfg['n_epochs'] = override_epochs
        train_cfg['patience'] = override_epochs + 1

    if train_cfg.get('device', 'auto') == 'auto':
        train_cfg['device'] = detect_device()
    print(f"  Device: {train_cfg['device']}")

    train_cfg['checkpoint_dir'] = ckpt_dir
    train_cfg['log_path'] = os.path.join(results_dir, f'training_log_{monkey}.json')
    train_cfg['norm_stats'] = stats

    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    train_result = trainer.train()

    trainer.load_best_checkpoint()
    val_results = evaluate_model(model, val_loader)

    print(f"\n--- Val Results: {variant_name} | {monkey.upper()} ---")
    print_results(val_results, prefix=f"{variant_name}/{monkey}")

    save_results(val_results, os.path.join(results_dir, f'final_{monkey}_val.json'))

    paper_mse = exp_cfg[f'paper_mse_{monkey}']
    paper_r2 = exp_cfg[f'paper_r2_{monkey}']
    mse_diff = val_results['mse'] - paper_mse
    r2_diff = val_results['r2']['mean'] - paper_r2
    print(f"\n  Paper target: MSE={paper_mse:.5f}, R²={paper_r2:.3f}")
    print(f"  Ours:         MSE={val_results['mse']:.5f}, R²={val_results['r2']['mean']:.3f}")
    print(f"  Delta:        MSE{mse_diff:+.5f}, R²{r2_diff:+.3f}")

    return {
        'variant': variant_name,
        'monkey': monkey,
        'val_mse': val_results['mse'],
        'val_r2': val_results['r2']['mean'],
        'val_corr': val_results['correlation']['mean'],
        'paper_mse': paper_mse,
        'paper_r2': paper_r2,
        'mse_diff': mse_diff,
        'r2_diff': r2_diff,
        'n_params_M': info['n_params_M'],
        'best_epoch_mse': train_result['best_val_mse'],
        'n_epochs_trained': len(train_result['history']),
    }


def print_summary_table(all_results: list):
    """Print a formatted comparison table across all variants and monkeys."""
    print(f"\n{'='*90}")
    print("  STUDY 002 — FINAL COMPARISON TABLE")
    print('='*90)
    print(f"  {'Variant':<22} {'Monkey':<10} {'Val MSE':<12} {'Paper MSE':<12} "
          f"{'MSE Δ':<10} {'R²':<8} {'Paper R²':<10}")
    print(f"  {'-'*85}")
    for r in all_results:
        pct = r['mse_diff'] / r['paper_mse'] * 100
        print(f"  {r['variant']:<22} {r['monkey']:<10} {r['val_mse']:<12.5f} "
              f"{r['paper_mse']:<12.5f} {r['mse_diff']:+.5f}({pct:+.1f}%)  "
              f"{r['val_r2']:<8.4f} {r['paper_r2']:<10.3f}")


def main():
    parser = argparse.ArgumentParser(description='Study 002: Faithful AMAG Replication')
    parser.add_argument('--config', default=os.path.join(_here, 'config.yaml'))
    parser.add_argument('--variant', choices=VARIANTS + ['all'], default='all',
                        help='Which variant to train')
    parser.add_argument('--monkey', choices=['affi', 'beignet', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override n_epochs (use small value for smoke test)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    variants_to_run = VARIANTS if args.variant == 'all' else [args.variant]
    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]

    all_results = []
    t_start = time.time()

    for variant_name in variants_to_run:
        for monkey in monkeys:
            try:
                result = train_variant(variant_name, monkey, config,
                                       override_epochs=args.epochs)
                all_results.append(result)
            except Exception as e:
                print(f"\n  [ERROR] {variant_name}/{monkey}: {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        print_summary_table(all_results)

        summary_path = os.path.join(_here, 'results', 'study_002_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Summary saved to: {summary_path}")

    total_min = (time.time() - t_start) / 60
    print(f"\n  Total time: {total_min:.1f} minutes")


if __name__ == '__main__':
    main()
