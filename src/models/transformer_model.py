"""
Transformer encoder using self-attention to capture long-range
temporal dependencies in financial return sequences.

This is the dissertation's primary model — attention mechanisms can
model complex non-linear temporal patterns that GARCH and even LSTM
struggle to capture.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerVolatilityModel(nn.Module):
    """
    Transformer encoder for one-step-ahead volatility forecasting.

    Architecture:
        Input -> Linear projection -> Positional Encoding
              -> Transformer Encoder (multi-head self-attention)
              -> Global pooling -> Dense -> Softplus -> volatility (>0)

    Parameters
    ----------
    input_size : int   features per timestep (default 1)
    d_model    : int   embedding dimension (default 64)
    nhead      : int   number of attention heads (default 4)
    num_layers : int   transformer encoder layers (default 2)
    dim_ff     : int   feed-forward hidden dim (default 128)
    dropout    : float dropout probability (default 0.1)
    """

    def __init__(self, input_size=1, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
            activation      = "gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc       = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)                 # global average pooling over timesteps
        out = self.fc(x)
        return self.softplus(out)

    @property
    def name(self):
        return "Transformer"