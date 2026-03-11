"""
Shared helper functions for the Streamlit dashboard.
"""

import sys
import os
import glob
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# Project root setup
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import MONKEY_CHANNELS, get_dataloaders


def get_available_monkeys() -> List[str]:
    """Return list of available monkey names."""
    return list(MONKEY_CHANNELS.keys())


def detect_device() -> str:
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def scan_checkpoints() -> List[Dict]:
    """
    Scan for available checkpoint files under the project root.
    Returns a list of dicts with keys: path, label, monkey (if detectable).
    """
    search_dirs = [
        os.path.join(_root, 'replication'),
        os.path.join(_root, 'replications'),
        os.path.join(_root, 'experiments'),
        os.path.join(_root, 'experiment_results'),
    ]
    checkpoints = []
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        pattern = os.path.join(base, '**', '*.pth')
        for path in sorted(glob.glob(pattern, recursive=True)):
            rel = os.path.relpath(path, _root)
            monkey = _detect_monkey_from_path(rel)
            checkpoints.append({
                'path': path,
                'label': rel,
                'monkey': monkey,
            })
    return checkpoints


def _detect_monkey_from_path(rel_path: str) -> Optional[str]:
    """Try to detect monkey name from a checkpoint path."""
    lower = rel_path.lower()
    if '/affi/' in lower or 'affi' in os.path.basename(os.path.dirname(rel_path)).lower():
        return 'affi'
    if '/beignet/' in lower or 'beignet' in os.path.basename(os.path.dirname(rel_path)).lower():
        return 'beignet'
    return None


def _sniff_model_cfg(state_dict: dict, config: dict) -> dict:
    """Infer model config from checkpoint state dict to handle architecture mismatches."""
    hidden_size = config.get('hidden_size', 64)

    # Detect hidden_size from GRU weight shape if not in config
    if 'te.gru.weight_ih_l0' in state_dict:
        hidden_size = state_dict['te.gru.weight_ih_l0'].shape[0] // 3

    # Detect use_adaptor: present only if adaptor_mlp keys exist
    use_adaptor = any('adaptor_mlp' in k for k in state_dict)

    # Detect adaptor_depth from number of Linear layers in adaptor_mlp
    adaptor_depth = 4  # default
    if use_adaptor:
        n_linear = sum(1 for k in state_dict if 'adaptor_mlp' in k and k.endswith('.weight'))
        adaptor_depth = n_linear * 2  # 1 linear → depth=2, 2 linears → depth=4

    # Detect use_mul: fc_mod present only when use_mul=True
    use_mul = any('fc_mod' in k for k in state_dict)

    return {
        'hidden_size': hidden_size,
        'use_adaptor': use_adaptor,
        'use_mul': use_mul,
        'adaptor_depth': adaptor_depth,
        'compute_init_corr': True,
    }


def load_checkpoint(path: str, monkey: str, device: str = 'cpu') -> Tuple[torch.nn.Module, Dict]:
    """
    Load a checkpoint and reconstruct the AMAG model.
    Returns (model, checkpoint_dict).
    """
    checkpoint = torch.load(path, weights_only=False, map_location=device)

    try:
        from replications.amag.model import build_model
    except ImportError:
        from replication.amag.model import build_model

    config = checkpoint.get('config', {})
    model_cfg = _sniff_model_cfg(checkpoint['model_state_dict'], config)
    model = build_model(monkey, model_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on entire dataloader and return (predictions, targets) as numpy arrays.
    Both have shape (N, 10, C).
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(Y.numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets


def collect_hidden_states(
    model: torch.nn.Module,
    X_batch: torch.Tensor,
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """
    Run a single batch through the model and capture intermediate hidden states
    using forward hooks. Returns dict with 'te_output', 'si_output', 'prediction'.
    """
    model.eval()
    X_batch = X_batch.to(device)
    states = {}

    def hook_te(module, input, output):
        states['te_output'] = output.detach().cpu().numpy()

    def hook_si(module, input, output):
        states['si_output'] = output.detach().cpu().numpy()

    h1 = model.te.register_forward_hook(hook_te)
    h2 = model.si.register_forward_hook(hook_si)

    with torch.no_grad():
        pred = model(X_batch)
        states['prediction'] = pred.detach().cpu().numpy()

    h1.remove()
    h2.remove()
    return states
