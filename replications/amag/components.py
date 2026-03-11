"""
AMAG component modules: Temporal Encoding, Spatial Interaction, Temporal Readout.

Reference: "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network
For Forecasting Neuron Activity", NeurIPS 2023.

Variants implemented:
  - TemporalEncoder: original GRU-based encoder (one-step / AMAG-G)
  - TemporalReadout: original single-shot GRU readout (one-step variant)
  - AutoregressiveTemporalReadout: AMAG-G paper variant — autoregressive GRU decoding
  - TransformerTemporalEncoder: Transformer-based encoder (AMAG-T / multi-step variant)
  - TransformerTemporalReadout: Transformer-based readout for simultaneous prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# GRU-based modules (original one-step variant)
# ---------------------------------------------------------------------------

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
    """GRU-based temporal readout (single-shot variant).
    Takes final hidden state → linear projection to all T_pred steps at once.
    Input: (B, T_ctx, C, d) → Output: (B, T_pred, C) LMP predictions."""

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


# ---------------------------------------------------------------------------
# Autoregressive GRU readout (AMAG-G paper variant)
# ---------------------------------------------------------------------------

class AutoregressiveTemporalReadout(nn.Module):
    """Autoregressive GRU temporal readout (AMAG-G paper variant, Appendix A.2).

    Paper description: each prediction is fed back as input for the next step.
    The encoder GRU compresses the context sequence Z into a hidden state, then
    a GRUCell decodes autoregressively step-by-step with optional teacher forcing.

    Input:  Z (B, T_ctx, C, d)
    Output: (B, T_pred, C) LMP predictions
    """

    def __init__(self, hidden_size: int, n_pred_steps: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_pred_steps = n_pred_steps
        # Encoder: compress context sequence to hidden state
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Decoder: step-by-step autoregressive prediction
        self.decoder_gru = nn.GRUCell(1, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    @property
    def supports_teacher_forcing(self) -> bool:
        return True

    def forward(self, Z: torch.Tensor,
                teacher_target: Optional[torch.Tensor] = None,
                teacher_ratio: float = 0.0) -> torch.Tensor:
        """
        Args:
            Z: (B, T_ctx, C, d) — spatial interaction output
            teacher_target: (B, T_pred, C) — normalized ground truth for teacher forcing
            teacher_ratio: float in [0, 1] — probability of using ground truth at each step
        """
        B, T, C, d = Z.shape
        Z_bc = Z.permute(0, 2, 1, 3).reshape(B * C, T, d)

        # Encode context to get initial decoder hidden state
        _, h_n = self.encoder_gru(Z_bc)
        h = h_n.squeeze(0)  # (B*C, d)

        # First prediction from initial hidden state
        pred_0 = self.fc_out(h)  # (B*C, 1)
        preds = [pred_0]

        # Autoregressive decoding
        for t in range(1, self.n_pred_steps):
            use_teacher = (
                teacher_target is not None
                and teacher_ratio > 0.0
                and torch.rand(1).item() < teacher_ratio
            )
            if use_teacher:
                # teacher_target: (B, T_pred, C) → need (B*C, 1)
                input_t = teacher_target[:, t - 1, :].permute(1, 0).reshape(B * C, 1)
            else:
                input_t = preds[-1]  # (B*C, 1) — use own prediction
            h = self.decoder_gru(input_t, h)
            preds.append(self.fc_out(h))

        # Stack and reshape: (B*C, n_pred_steps) → (B, T_pred, C)
        pred_bc = torch.cat(preds, dim=-1)  # (B*C, n_pred_steps)
        return pred_bc.reshape(B, C, self.n_pred_steps).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Transformer-based modules (AMAG-T paper variant, multi-step)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 20, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerTemporalEncoder(nn.Module):
    """Transformer-based temporal encoder (AMAG-T, Appendix A.3).

    Per-channel processing (weight-tied across channels):
      1. FC projection: F features → hidden_size
      2. Sinusoidal positional encoding
      3. N × Transformer encoder layers (pre-norm, multi-head attention + FFN)

    Input:  (B, T_full, C, F)  — full 20-step sequence, future positions masked to constant
    Output: (B, T_full, C, d)

    Paper: "E = FC(mask(X)) + PE; ATT = Softmax(QKᵀ/√d)V; H = LN(MLP(E + ATT))"
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_heads: int = 4, n_layers: int = 2,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, F = x.shape
        # Process all channels jointly with weight-tied transformer
        x_bc = x.permute(0, 2, 1, 3).reshape(B * C, T, F)
        h = self.input_proj(x_bc)   # (B*C, T, d)
        h = self.pos_enc(h)         # + positional encoding
        h = self.transformer(h)     # (B*C, T, d)
        return h.reshape(B, C, T, self.hidden_size).permute(0, 2, 1, 3)


class TransformerTemporalReadout(nn.Module):
    """Transformer-based temporal readout for simultaneous multi-step prediction (AMAG-T).

    Processes the full T_full sequence representation from SI, then extracts
    predictions at future timestep positions via a linear d→1 head.

    Input:  Z (B, T_full, C, d)
    Output: (B, T_pred, C) LMP predictions at positions [n_context_steps : n_context_steps+T_pred]

    Paper: "Same Transformer structure as TE; output: predicted LMP at future positions via linear d→1"
    """

    def __init__(self, hidden_size: int, n_pred_steps: int = 10,
                 n_context_steps: int = 10,
                 n_heads: int = 4, n_layers: int = 2,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_pred_steps = n_pred_steps
        self.n_context_steps = n_context_steps
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, C, d = Z.shape
        Z_bc = Z.permute(0, 2, 1, 3).reshape(B * C, T, d)
        Z_bc = self.pos_enc(Z_bc)
        out = self.transformer(Z_bc)  # (B*C, T, d)

        # Extract outputs at prediction positions and project to LMP scalar
        ctx = self.n_context_steps
        pred_steps = self.n_pred_steps
        future = out[:, ctx:ctx + pred_steps, :]  # (B*C, T_pred, d)
        pred = self.fc_out(future).squeeze(-1)     # (B*C, T_pred)
        return pred.reshape(B, C, pred_steps).permute(0, 2, 1)  # (B, T_pred, C)
