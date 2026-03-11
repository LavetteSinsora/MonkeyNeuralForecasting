#!/usr/bin/env python3
"""
AMAG Replication — Ablation Study

Usage:
    python replications/amag/ablation.py
    python replications/amag/ablation.py --epochs 5 --monkey beignet
"""

import sys
import os
import argparse
import json
import yaml
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import get_dataloaders
from utils.metrics import evaluate_model, save_results
from replications.amag.model import build_model
from replications.amag.train import Trainer, set_seed


def run_ablation_variant(monkey, variant, ablation_train_cfg, data_cfg, model_base_cfg, results_dir, override_epochs=None):
    name = variant['name']
    print(f"\n  [{name}] {variant.get('description', name)}")

    train_loader, val_loader, stats = get_dataloaders(
        monkey,
        batch_size=ablation_train_cfg.get('batch_size', 32),
        val_fraction=ablation_train_cfg.get('val_fraction', 0.15),
        seed=ablation_train_cfg.get('seed', 42),
    )

    model = build_model(monkey, model_base_cfg, variant=variant)
    info = model.get_model_info()
    print(f"    params={info['n_params_M']}M", end='')

    cfg = ablation_train_cfg.copy()
    if cfg.get('device', 'auto') == 'auto':
        if torch.cuda.is_available():
            cfg['device'] = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cfg['device'] = 'mps'
        else:
            cfg['device'] = 'cpu'

    ckpt_dir = os.path.join(_here, 'checkpoints', f'ablation_{monkey}_{name}')
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg['checkpoint_dir'] = ckpt_dir
    cfg['log_path'] = os.path.join(results_dir, f'ablation_{monkey}_{name}_log.json')
    cfg['norm_stats'] = stats
    if override_epochs is not None:
        cfg['n_epochs'] = override_epochs
        cfg['patience'] = override_epochs + 1

    trainer = Trainer(model, train_loader, val_loader, cfg)
    train_result = trainer.train(verbose=False)
    trainer.load_best_checkpoint()
    val_results = evaluate_model(model, val_loader)

    mse_val = val_results['mse']
    r2 = val_results['r2']['mean']
    corr = val_results['correlation']['mean']
    print(f" -> MSE={mse_val:.5f}  R2={r2:.4f}  Corr={corr:.4f}")

    save_results(val_results, os.path.join(results_dir, f'ablation_{monkey}_{name}_eval.json'))
    return {'name': name, 'monkey': monkey, 'mse': mse_val, 'r2': r2, 'correlation': corr,
            'best_val_mse': train_result['best_val_mse'], 'n_epochs': len(train_result['history'])}


def main():
    parser = argparse.ArgumentParser(description='AMAG Ablation Study')
    parser.add_argument('--config', default=os.path.join(_here, 'config.yaml'))
    parser.add_argument('--monkey', choices=['affi', 'beignet'], default='beignet')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--variants', nargs='+', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    variants = config.get('ablations', [])
    if args.variants:
        variants = [v for v in variants if v['name'] in args.variants]

    ablation_train_cfg = config.get('ablation_training', config['training'].copy())
    results_dir = os.path.join(_here, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  AMAG Ablation Study — {args.monkey.upper()} ({len(variants)} variants)")
    print('='*65)

    all_results = []
    for variant in variants:
        result = run_ablation_variant(
            args.monkey, variant, ablation_train_cfg, config['data'],
            config['model'], results_dir, override_epochs=args.epochs,
        )
        all_results.append(result)

    baseline = next((r for r in all_results if r['name'] == 'full_amag'), None)
    print(f"\n{'='*65}")
    print(f"  ABLATION SUMMARY — {args.monkey.upper()}")
    print('='*65)
    print(f"  {'Variant':<20} {'MSE':<10} {'R²':<8} {'ΔR² vs full'}")
    print(f"  {'-'*45}")
    for r in all_results:
        delta = (r['r2'] - baseline['r2']) if baseline else float('nan')
        print(f"  {r['name']:<20} {r['mse']:<10.5f} {r['r2']:<8.4f} {delta:+.4f}")

    summary_path = os.path.join(results_dir, f'ablation_{args.monkey}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
