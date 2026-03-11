"""
AMAG Replication Model with ablation support.

Architecture: Input → Temporal Encoding (TE) → Spatial Interaction (SI) → Temporal Readout (TR)

Reference: "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network
for Forecasting Neuron Activity", NeurIPS 2023.
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from replications.amag.components import TemporalEncoder, TemporalReadout


class ForecastingModel(nn.Module, ABC):
    """Abstract base class for all neural forecasting models.
    forward(x): (B, 10, C, 9) → (B, 10, C) predicted LMP."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def get_model_info(self) -> Dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'name': self.__class__.__name__,
            'n_params': n_params,
            'n_params_M': round(n_params / 1e6, 3),
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            pred = self(x_t)
        return pred.cpu().numpy()


class SpatialInteractionAblatable(nn.Module):
    """AMAG Spatial Interaction module with ablation flags.

    Computes: z^(v)_t = β1*h^(v)_t + β2*FC(a^(v)_t) + β3*FC(m^(v)_t)

    Ablation flags: use_add, use_mul, use_adaptor, learnable_adj
    """

    def __init__(self, hidden_size, n_channels, init_corr=None,
                 use_add=True, use_mul=True, use_adaptor=True, learnable_adj=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.use_add = use_add
        self.use_mul = use_mul
        self.use_adaptor = use_adaptor

        if init_corr is not None:
            corr_tensor = torch.tensor(init_corr, dtype=torch.float32)
        else:
            std = 1.0 / (n_channels ** 0.5)
            corr_tensor = torch.randn(n_channels, n_channels) * std

        if learnable_adj:
            self.A_a = nn.Parameter(corr_tensor.clone())
            self.A_m = nn.Parameter(corr_tensor.clone())
        else:
            self.register_buffer('A_a', corr_tensor.clone())
            self.register_buffer('A_m', corr_tensor.clone())

        self.beta = nn.Parameter(torch.ones(3))

        if use_add:
            self.fc_add = nn.Linear(hidden_size, hidden_size, bias=False)
        if use_mul:
            self.fc_mod = nn.Linear(hidden_size, hidden_size, bias=False)

        if use_add and use_adaptor:
            self.adaptor_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, C, d = H.shape
        n_active = 1 + int(self.use_add) + int(self.use_mul)
        beta = F.softmax(self.beta[:n_active], dim=0) * n_active

        Z_list = []
        for t in range(T):
            h_t = H[:, t, :, :]
            components = [h_t]

            if self.use_add:
                A_a_norm = torch.tanh(self.A_a)
                if self.use_adaptor:
                    H_mean = H[:, :t + 1, :, :].mean(dim=1)
                    h_u = H_mean.unsqueeze(2).expand(-1, -1, C, -1)
                    h_v = H_mean.unsqueeze(1).expand(-1, C, -1, -1)
                    h_uv = torch.cat([h_u, h_v], dim=-1)
                    S = torch.sigmoid(self.adaptor_mlp(h_uv).squeeze(-1))
                    A_a_scaled = A_a_norm.unsqueeze(0) * S
                    a_t = torch.einsum('bvu, bud -> bvd', A_a_scaled, h_t)
                else:
                    a_t = torch.einsum('vu, bud -> bvd', A_a_norm, h_t)
                components.append(self.fc_add(a_t))

            if self.use_mul:
                A_m_norm = torch.tanh(self.A_m)
                h_u = h_t.unsqueeze(1).expand(-1, C, -1, -1)
                h_v = h_t.unsqueeze(2).expand(-1, -1, C, -1)
                hadamard = h_u * h_v
                m_t = torch.einsum('vu, bvud -> bvd', A_m_norm, hadamard)
                components.append(self.fc_mod(m_t))

            z_t = sum(beta[i] * components[i] for i in range(len(components)))
            Z_list.append(z_t)

        return torch.stack(Z_list, dim=1)


class AMAGReplica(ForecastingModel):
    """AMAG replication model. Pipeline: TE (GRU) → SI (add + mul graph) → TR (GRU → linear)"""

    def __init__(self, n_channels, n_features=9, hidden_size=64, n_pred_steps=10,
                 init_corr=None, use_add=True, use_mul=True, use_adaptor=True, learnable_adj=True):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.te = TemporalEncoder(n_features, hidden_size)
        self.si = SpatialInteractionAblatable(
            hidden_size=hidden_size, n_channels=n_channels, init_corr=init_corr,
            use_add=use_add, use_mul=use_mul, use_adaptor=use_adaptor, learnable_adj=learnable_adj,
        )
        self.tr = TemporalReadout(hidden_size, n_pred_steps)

    def forward(self, x):
        H = self.te(x)
        Z = self.si(H)
        return self.tr(Z)


def compute_correlation_matrix(monkey: str) -> np.ndarray:
    """Compute LMP channel-channel correlation matrix from training data."""
    from utils.data import MONKEY_CHANNELS, MONKEY_FILES, DATASET_DIR, load_npz
    n_channels = MONKEY_CHANNELS[monkey]
    train_path = os.path.join(DATASET_DIR, MONKEY_FILES[monkey]['train'])
    raw = load_npz(train_path)
    lmp = raw[:, :10, :, 0].reshape(-1, n_channels)
    lmp_centered = lmp - lmp.mean(axis=0)
    norm = np.sqrt((lmp_centered ** 2).sum(axis=0) + 1e-12)
    lmp_normed = lmp_centered / norm
    corr = (lmp_normed.T @ lmp_normed) / len(lmp)
    return corr.astype(np.float32)


def build_model(monkey, model_cfg, variant=None):
    """Factory function: build AMAGReplica from config dicts."""
    from utils.data import MONKEY_CHANNELS
    n_channels = MONKEY_CHANNELS[monkey]
    hidden_size = model_cfg.get('hidden_size', 64)

    if variant is not None:
        use_add = variant.get('use_add', True)
        use_mul = variant.get('use_mul', True)
        use_adaptor = variant.get('use_adaptor', True)
        learnable_adj = variant.get('learnable_adj', True)
        init_type = variant.get('init_type', 'correlation')
        compute_init_corr = (init_type == 'correlation')
    else:
        use_add = True
        use_mul = True
        use_adaptor = model_cfg.get('use_adaptor', True)
        learnable_adj = True
        compute_init_corr = model_cfg.get('compute_init_corr', True)

    init_corr = None
    if compute_init_corr:
        print(f"  Computing LMP correlation matrix for {monkey}...")
        init_corr = compute_correlation_matrix(monkey)

    return AMAGReplica(
        n_channels=n_channels, hidden_size=hidden_size,
        use_add=use_add, use_mul=use_mul, use_adaptor=use_adaptor,
        learnable_adj=learnable_adj, init_corr=init_corr,
    )
