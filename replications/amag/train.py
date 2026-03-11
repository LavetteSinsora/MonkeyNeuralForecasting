#!/usr/bin/env python3
"""
AMAG Replication — Training Script

Usage:
    python replications/amag/train.py
    python replications/amag/train.py --epochs 2        # smoke test
    python replications/amag/train.py --monkey affi
"""

import sys
import os
import argparse
import json
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import get_dataloaders
from utils.metrics import evaluate_model, print_results, save_results
from replications.amag.model import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Generic trainer with MSE loss, early stopping, checkpoint saving, LR scheduling."""

    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)

        set_seed(config.get('seed', 42))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_decay_every', 50),
            gamma=config.get('lr_decay', 0.95),
        )
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self.log_path = config.get('log_path', './results/metrics.json')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.history = []
        self.best_val_mse = float('inf')
        self.patience_counter = 0

    def _train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        for X, Y in self.train_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def _val_epoch(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for X, Y in self.val_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                loss = self.criterion(self.model(X), Y)
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    def save_checkpoint(self, tag='best'):
        path = os.path.join(self.checkpoint_dir, f'{tag}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mse': self.best_val_mse,
            'norm_stats': self.config.get('norm_stats', None),
            'config': {k: v for k, v in self.config.items() if k != 'norm_stats'},
        }, path)

    def load_best_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, 'best.pth')
        state = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])

    def train(self, verbose=True):
        n_epochs = self.config.get('n_epochs', 500)
        patience = self.config.get('patience', 100)
        log_every = self.config.get('log_every', 10)

        info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        if verbose:
            print(f"Training {info.get('name', 'model')} | {info.get('n_params_M', '?')}M params | device={self.device}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_mse = self._train_epoch()
            val_mse = self._val_epoch()
            self.scheduler.step()

            improved = val_mse < self.best_val_mse
            if improved:
                self.best_val_mse = val_mse
                self.patience_counter = 0
                self.save_checkpoint('best')
            else:
                self.patience_counter += 1

            epoch_info = {
                'epoch': epoch, 'train_mse': train_mse, 'val_mse': val_mse,
                'best_val_mse': self.best_val_mse,
                'lr': self.scheduler.get_last_lr()[0],
                'time_s': round(time.time() - t0, 2),
            }
            self.history.append(epoch_info)

            if verbose and (epoch % log_every == 0 or epoch == 1):
                mark = '*' if improved else ' '
                print(f"  Epoch {epoch:4d}/{n_epochs} {mark} | train={train_mse:.6f} val={val_mse:.6f} "
                      f"best={self.best_val_mse:.6f} | lr={epoch_info['lr']:.2e} | {epoch_info['time_s']:.1f}s")

            if self.patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        with open(self.log_path, 'w') as f:
            json.dump({'history': self.history, 'best_val_mse': self.best_val_mse}, f, indent=2)

        if verbose:
            print(f"\nTraining complete. Best val MSE: {self.best_val_mse:.6f}")
        return {'best_val_mse': self.best_val_mse, 'history': self.history}


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_monkey(monkey, config, override_epochs=None):
    print(f"\n{'='*65}")
    print(f"  AMAG Replication — {monkey.upper()}")
    print(f"  Target MSE: {config['experiment'][f'paper_mse_{monkey}']}")
    print('='*65)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training'].copy()

    train_loader, val_loader, stats = get_dataloaders(
        monkey, batch_size=data_cfg['batch_size'],
        val_fraction=data_cfg['val_fraction'], seed=data_cfg['seed'],
    )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_model(monkey, model_cfg)
    info = model.get_model_info()
    print(f"  Model: {info['name']} | {info['n_params_M']}M params")

    results_dir = os.path.join(_here, 'results')
    ckpt_dir = os.path.join(_here, 'checkpoints', monkey)
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
    print(f"\n--- Val Results ({monkey.upper()}) ---")
    print_results(val_results, prefix=monkey)
    save_results(val_results, os.path.join(results_dir, f'final_{monkey}_val.json'))

    return {
        'monkey': monkey,
        'val_mse': val_results['mse'],
        'val_r2': val_results['r2']['mean'],
        'val_corr': val_results['correlation']['mean'],
        'paper_mse': config['experiment'][f'paper_mse_{monkey}'],
        'best_epoch_mse': train_result['best_val_mse'],
        'n_epochs_trained': len(train_result['history']),
    }


def main():
    parser = argparse.ArgumentParser(description='AMAG Replication Training')
    parser.add_argument('--config', default=os.path.join(_here, 'config.yaml'))
    parser.add_argument('--monkey', choices=['affi', 'beignet', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]
    all_results = []
    for monkey in monkeys:
        result = train_monkey(monkey, config, override_epochs=args.epochs)
        all_results.append(result)

    print(f"\n{'='*65}")
    print("  FINAL COMPARISON VS PAPER")
    print('='*65)
    print(f"  {'Monkey':<10} {'Val MSE':<12} {'Paper MSE':<12} {'Diff':<10} {'R²':<8}")
    print(f"  {'-'*55}")
    for r in all_results:
        diff = r['val_mse'] - r['paper_mse']
        print(f"  {r['monkey']:<10} {r['val_mse']:<12.5f} {r['paper_mse']:<12.5f} {diff:+.5f}    {r['val_r2']:<8.4f}")

    summary_path = os.path.join(_here, 'results', 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
