import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformer_lstm import TransformerLSTM
import torch


def test_forward_shape():
    m = TransformerLSTM(input_dim=1, d_model=16, nhead=4, num_layers=1, lstm_hidden=16, num_actions=2)
    x = torch.rand(4, 8, 1)
    out = m(x)
    assert out.shape == (4, 2)


if __name__ == '__main__':
    test_forward_shape()
    print('test passed')
