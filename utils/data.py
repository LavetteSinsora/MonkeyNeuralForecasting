"""
Data loading and preprocessing for MonkeyNeuralForecasting.

Data format: (N, 20, C, 9) npz arrays
  - N: number of trials
  - 20: timesteps at 30ms each (600ms total)
  - C: channels (239 for affi, 89 for beignet)
  - 9: features [LMP, pb0.5-4, pb4-8, pb8-12, pb12-25, pb25-50, pb50-100, pb100-200, pb200-400]

Split: timesteps 0-9 are input (all 9 features).
Timesteps 10-19 contain real recorded data for all 9 features, but must NOT be used
as model input at inference — they are the prediction target window.
Target: only feature 0 (LMP) at prediction timesteps 10-19.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
CONTEXT_STEPS = 10
PRED_STEPS = 10

MONKEY_FILES = {
    'affi': {
        'train': 'train_data_affi.npz',
        'test':  ['train_data_affi_2024-03-20_private.npz'],
    },
    'beignet': {
        'train': 'train_data_beignet.npz',
        'test':  [
            'train_data_beignet_2022-06-01_private.npz',
            'train_data_beignet_2022-06-02_private.npz',
        ],
    },
}

MONKEY_CHANNELS = {'affi': 239, 'beignet': 89}


def load_npz(path: str) -> np.ndarray:
    """Load a .npz file and return the data array of shape (N, 20, C, 9)."""
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    if 'data' in keys:
        return data['data']
    return data[keys[0]]


def compute_normalization_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-channel-per-feature normalization statistics from training data.
    Uses only the context timesteps (0:CONTEXT_STEPS) to avoid leakage."""
    ctx = data[:, :CONTEXT_STEPS, :, :]
    flat = ctx.reshape(-1, ctx.shape[2], ctx.shape[3])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return {'mean': mean, 'std': std}


def normalize(data: np.ndarray, stats: Dict[str, np.ndarray],
              clip_value: float = 4.0) -> np.ndarray:
    """Normalize data: z-score then clip to [-clip_value, clip_value] and rescale to [-1, 1]."""
    mean = stats['mean']
    std = stats['std']
    normalized = (data - mean) / std
    normalized = np.clip(normalized, -clip_value, clip_value) / clip_value
    return normalized


def denormalize_lmp(arr: np.ndarray, stats: Dict[str, np.ndarray],
                    clip_value: float = 4.0) -> np.ndarray:
    """Reverse normalization for LMP (feature 0) predictions.
    arr: (N, T, C) normalized LMP → returns raw-scale LMP in μV."""
    lmp_mean = stats['mean'][:, 0]
    lmp_std = stats['std'][:, 0]
    return arr * clip_value * lmp_std + lmp_mean


def split_context_target(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split normalized data into context (input) and target (output).
    Returns X: (N, 10, C, 9), Y: (N, 10, C) LMP only."""
    X = data[:, :CONTEXT_STEPS, :, :]
    Y = data[:, CONTEXT_STEPS:, :, 0]
    return X, Y


class MonkeyDataset(Dataset):
    """PyTorch dataset for monkey neural forecasting."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    monkey: str,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load training data, normalize, split train/val, return DataLoaders + norm stats."""
    assert monkey in MONKEY_FILES, f"Unknown monkey: {monkey}. Choose from {list(MONKEY_FILES)}"
    train_path = os.path.join(DATASET_DIR, MONKEY_FILES[monkey]['train'])
    raw = load_npz(train_path)
    stats = compute_normalization_stats(raw)
    data_norm = normalize(raw, stats)
    rng = np.random.default_rng(seed)
    n = len(data_norm)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    X_train, Y_train = split_context_target(data_norm[train_idx])
    X_val, Y_val = split_context_target(data_norm[val_idx])
    train_ds = MonkeyDataset(X_train, Y_train)
    val_ds = MonkeyDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, stats


def get_test_loader(
    monkey: str,
    stats: Dict,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Load and normalize test data using training normalization stats."""
    test_files = MONKEY_FILES[monkey]['test']
    arrays = []
    for fname in test_files:
        path = os.path.join(DATASET_DIR, fname)
        arrays.append(load_npz(path))
    raw = np.concatenate(arrays, axis=0)
    data_norm = normalize(raw, stats)
    X, Y = split_context_target(data_norm)
    ds = MonkeyDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_raw_test_data(monkey: str) -> np.ndarray:
    """Return concatenated raw (unnormalized) test data for submission generation."""
    test_files = MONKEY_FILES[monkey]['test']
    arrays = [load_npz(os.path.join(DATASET_DIR, f)) for f in test_files]
    return np.concatenate(arrays, axis=0)


def prepare_submission_array(
    context_lmp: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Combine context LMP and predictions into competition submission format.
    context_lmp: (N, 10, C), predictions: (N, 10, C) → returns (N, 20, C)."""
    return np.concatenate([context_lmp, predictions], axis=1)
