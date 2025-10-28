"""Create and save a small random TransformerLSTM checkpoint to `models/`.

This is a helper so the model server can demonstrate loading a real model.
"""
import os
from pathlib import Path
import torch

from src.transformer_lstm import TransformerLSTM


def main():
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TransformerLSTM(input_dim=1, d_model=32, nhead=4, num_layers=1, lstm_hidden=32, num_actions=2)
    path = out_dir / "transformer_lstm.pth"
    torch.save({'state_dict': model.state_dict()}, path)
    print(f"Saved dummy checkpoint to {path}")


if __name__ == '__main__':
    main()
