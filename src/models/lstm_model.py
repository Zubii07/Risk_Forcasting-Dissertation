"""
LSTM Volatility Forecasting Model
Long Short-Term Memory network for capturing temporal dependencies
in financial return sequences to forecast next day volatility.
"""

import torch.nn as nn
class LSTMVolatilityModel(nn.Module):
    """
    LSTM network for one-step-ahead volatility forecasting.

    Architecture:
        Input  -> LSTM layers -> Dropout -> Dense -> Softplus -> volatility (>0)
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()    # ensures positive volatility output

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]              # take last timestep output
        last = self.dropout(last)
        out  = self.fc(last)
        return self.softplus(out)         # volatility must be positive

    @property
    def name(self):
        return "LSTM"