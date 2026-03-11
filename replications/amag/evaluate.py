#!/usr/bin/env python3
"""
AMAG Replication — Evaluation Script

Usage:
    python replications/amag/evaluate.py --monkey affi
    python replications/amag/evaluate.py --monkey both --plots
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import get_dataloaders, get_test_loader
from utils.metrics import evaluate_model, print_results, save_results
from replications.amag.model import build_model


def load_checkpoint(monkey, config):
    ckpt_path = os.path.join(_here, 'checkpoints', monkey, 'best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    state = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = build_model(monkey, config['model'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    norm_stats = state.get('norm_stats', None)
    return model, norm_stats


def evaluate_monkey(monkey, config, split='val'):
    print(f"\n{'='*65}")
    print(f"  Evaluating AMAG — {monkey.upper()} ({split} set)")
    print('='*65)

    model, norm_stats = load_checkpoint(monkey, config)
    data_cfg = config['data']

    if split == 'val' or norm_stats is None:
        _, val_loader, stats = get_dataloaders(
            monkey, batch_size=64,
            val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
        )
        if norm_stats is None:
            norm_stats = stats
        loader = val_loader
    else:
        _, _, stats = get_dataloaders(
            monkey, batch_size=64,
            val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
        )
        loader = get_test_loader(monkey, stats, batch_size=64)
        norm_stats = stats

    device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    results = evaluate_model(model, loader, device=device, norm_stats=norm_stats)

    print(f"\n--- {monkey.upper()} {split.capitalize()} Results ---")
    print_results(results, prefix=monkey)

    paper_mse = config['experiment'].get(f'paper_mse_{monkey}')
    if paper_mse:
        diff = results['mse'] - paper_mse
        pct = diff / paper_mse * 100
        print(f"\n  Paper MSE: {paper_mse:.5f} | Ours: {results['mse']:.5f} | Diff: {diff:+.5f} ({pct:+.1f}%)")

    results_dir = os.path.join(_here, 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, os.path.join(results_dir, f'eval_{monkey}_{split}.json'))
    return results


def main():
    parser = argparse.ArgumentParser(description='AMAG Evaluation')
    parser.add_argument('--config', default=os.path.join(_here, 'config.yaml'))
    parser.add_argument('--monkey', choices=['affi', 'beignet', 'both'], default='both')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]
    for monkey in monkeys:
        try:
            evaluate_monkey(monkey, config, split=args.split)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")


if __name__ == '__main__':
    main()
