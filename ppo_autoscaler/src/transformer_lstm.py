"""Compact Transformer + LSTM model for CPU autoscaling (PyTorch).

This is a small, self-contained implementation intended for demonstration
and local testing. It matches the high-level architecture described in the
paper: a small Transformer encoder followed by an LSTM and a linear head.

The forward() accepts a tensor of shape (batch, seq_len, features).
It returns logits for a discrete action space (for example: 2 actions).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # last odd column stays zero for simplicity
            pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 64, nhead: int = 4, num_layers: int = 1, lstm_hidden: int = 64, num_actions: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim)

        Returns:
            logits: Tensor of shape (batch, num_actions)
        """
        # project inputs to d_model
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # transformer expects (seq_len, batch, d_model)
        t = x.transpose(0, 1)
        t = self.transformer(t)
        # back to (batch, seq_len, d_model)
        t = t.transpose(0, 1)

        # LSTM (batch, seq_len, d_model) -> take last output
        out, (h_n, c_n) = self.lstm(t)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits


def example():
    m = TransformerLSTM(input_dim=1, d_model=32, nhead=4, num_layers=1, lstm_hidden=32, num_actions=2)
    x = torch.rand(8, 10, 1)
    logits = m(x)
    print('logits', logits.shape)


if __name__ == '__main__':
    example()
