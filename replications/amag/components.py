"""
AMAG component modules: Temporal Encoding, Spatial Interaction, Temporal Readout.

Reference: "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network
For Forecasting Neuron Activity", NeurIPS 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder. Processes each channel independently (weight-tied).
    Input: (B, T, C, F) → Output: (B, T, C, d)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, F = x.shape
        x_bc = x.permute(0, 2, 1, 3).reshape(B * C, T, F)
        h, _ = self.gru(x_bc)
        return h.reshape(B, C, T, self.hidden_size).permute(0, 2, 1, 3)


class TemporalReadout(nn.Module):
    """GRU-based temporal readout. Maps (B, T_ctx, C, d) → (B, T_pred, C) LMP predictions."""

    def __init__(self, hidden_size: int, n_pred_steps: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_pred_steps = n_pred_steps
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_pred_steps)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, C, d = Z.shape
        Z_bc = Z.permute(0, 2, 1, 3).reshape(B * C, T, d)
        _, h_n = self.gru(Z_bc)
        h_last = h_n.squeeze(0)
        pred_bc = self.fc(h_last)
        return pred_bc.reshape(B, C, self.n_pred_steps).permute(0, 2, 1)
