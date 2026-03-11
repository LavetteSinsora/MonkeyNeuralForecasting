"""
Model class definitions for Study 001: AMAG Architecture Improvements.

Extracted from notebooks/colab_amag_experiments.ipynb so that checkpoints
can be loaded locally for evaluation and dashboard visualization.

Classes:
    AMAGBaseline     -- standard AMAG (TE -> SI -> TR)
    AMAG_MultiHop    -- 2-hop SI with learned gating
    AMAG_DeltaTR     -- delta input + h0 init for TR
    AMAG_Interleaved -- interleaved TE-SI + auxiliary loss

Usage:
    from experiments.study_001_amag_arch_improvements.models import MODEL_REGISTRY
    model_cls, kwargs = MODEL_REGISTRY['baseline']
    model = model_cls(n_channels=239, **kwargs)
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from replications.amag.model import ForecastingModel, SpatialInteractionAblatable
from replications.amag.components import TemporalEncoder, TemporalReadout


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class AMAGBaseline(ForecastingModel):
    """Standard AMAG: TE -> SI -> TR."""

    def __init__(self, n_channels, n_features=9, hidden_size=64,
                 n_pred_steps=10, init_corr=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.te = TemporalEncoder(n_features, hidden_size)
        self.si = SpatialInteractionAblatable(
            hidden_size=hidden_size, n_channels=n_channels,
            init_corr=init_corr, use_add=True, use_mul=True,
            use_adaptor=True, learnable_adj=True)
        self.tr = TemporalReadout(hidden_size, n_pred_steps)

    def forward(self, x):
        H = self.te(x)
        Z = self.si(H)
        return self.tr(Z)


# ---------------------------------------------------------------------------
# Experiment 1: Multi-hop SI
# ---------------------------------------------------------------------------

class AMAG_MultiHop(ForecastingModel):
    """AMAG with 2-hop spatial interaction and learned gating."""

    def __init__(self, n_channels, n_features=9, hidden_size=64,
                 n_pred_steps=10, init_corr=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.te = TemporalEncoder(n_features, hidden_size)
        self.si = SpatialInteractionAblatable(
            hidden_size=hidden_size, n_channels=n_channels,
            init_corr=init_corr, use_add=True, use_mul=True,
            use_adaptor=True, learnable_adj=True)
        self.tr = TemporalReadout(hidden_size, n_pred_steps)
        self.hop2_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        H = self.te(x)
        Z1 = self.si(H)
        Z2 = self.si(Z1)
        alpha = torch.sigmoid(self.hop2_alpha)
        Z = (1 - alpha) * Z1 + alpha * Z2
        return self.tr(Z)


# ---------------------------------------------------------------------------
# Experiment 2: Delta TR
# ---------------------------------------------------------------------------

class TemporalReadoutWithH0(nn.Module):
    """Modified TR that accepts an initial hidden state h0."""

    def __init__(self, hidden_size, n_pred_steps=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_pred_steps = n_pred_steps
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_pred_steps)

    def forward(self, Z, h0=None):
        B, T, C, d = Z.shape
        Z_bc = Z.permute(0, 2, 1, 3).reshape(B * C, T, d)
        if h0 is not None:
            h0_bc = h0.reshape(B * C, d).unsqueeze(0).contiguous()
            _, h_n = self.gru(Z_bc, h0_bc)
        else:
            _, h_n = self.gru(Z_bc)
        h_last = h_n.squeeze(0)
        pred_bc = self.fc(h_last)
        return pred_bc.reshape(B, C, self.n_pred_steps).permute(0, 2, 1)


class AMAG_DeltaTR(ForecastingModel):
    """AMAG with delta input for TR and h0 initialization."""

    def __init__(self, n_channels, n_features=9, hidden_size=64,
                 n_pred_steps=10, init_corr=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.te = TemporalEncoder(n_features, hidden_size)
        self.si = SpatialInteractionAblatable(
            hidden_size=hidden_size, n_channels=n_channels,
            init_corr=init_corr, use_add=True, use_mul=True,
            use_adaptor=True, learnable_adj=True)
        self.tr = TemporalReadoutWithH0(hidden_size, n_pred_steps)

    def forward(self, x):
        H = self.te(x)
        Z = self.si(H)
        Delta = Z - H
        h_last = H[:, -1, :, :]
        return self.tr(Delta, h0=h_last)


# ---------------------------------------------------------------------------
# Experiment 3: Interleaved TE-SI + Auxiliary Loss
# ---------------------------------------------------------------------------

class InterleavedTemporalEncoder(nn.Module):
    """TE that interleaves spatial interaction at each timestep."""

    def __init__(self, input_size, hidden_size, n_channels, init_corr=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.si = SpatialInteractionAblatable(
            hidden_size=hidden_size, n_channels=n_channels,
            init_corr=init_corr, use_add=True, use_mul=True,
            use_adaptor=True, learnable_adj=True)
        self.W_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, T, C, n_feat = x.shape
        device = x.device
        h = torch.zeros(B * C, self.hidden_size, device=device)
        H_list = []
        Z_list = []
        alpha = torch.sigmoid(self.alpha)

        for t in range(T):
            x_t = x[:, t, :, :]
            x_bc = x_t.reshape(B * C, n_feat)
            h = self.gru_cell(x_bc, h)
            h_shaped = h.reshape(B, C, self.hidden_size)
            h_for_si = h_shaped.unsqueeze(1)
            z_for_si = self.si(h_for_si)
            z_t = z_for_si.squeeze(1)
            Z_list.append(z_t)
            spatial_delta = z_t - h_shaped
            gate = alpha * torch.tanh(self.W_gate(spatial_delta))
            h_updated = h_shaped + gate
            H_list.append(h_updated)
            h = h_updated.reshape(B * C, self.hidden_size)

        H = torch.stack(H_list, dim=1)
        Z_all = torch.stack(Z_list, dim=1)
        return H, Z_all


class AMAG_Interleaved(ForecastingModel):
    """AMAG with interleaved TE-SI and auxiliary next-step prediction loss."""

    def __init__(self, n_channels, n_features=9, hidden_size=64,
                 n_pred_steps=10, init_corr=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.te_si = InterleavedTemporalEncoder(
            n_features, hidden_size, n_channels, init_corr)
        self.tr = TemporalReadout(hidden_size, n_pred_steps)
        self.aux_decoder = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        H, Z_all = self.te_si(x)
        pred = self.tr(H)
        if self.training:
            z_for_aux = Z_all[:, :-1, :, :]
            x_targets = x[:, 1:, :, :]
            x_pred = self.aux_decoder(z_for_aux)
            aux_loss = nn.functional.mse_loss(x_pred, x_targets)
            return pred, aux_loss
        return pred


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'baseline': AMAGBaseline,
    'exp_001_multihop_si': AMAG_MultiHop,
    'exp_002_delta_tr': AMAG_DeltaTR,
    'exp_003_interleaved_aux': AMAG_Interleaved,
}
