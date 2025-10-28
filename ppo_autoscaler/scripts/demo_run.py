"""Generate synthetic CPU traces, run the model, and print predictions.

This demo produces console output similar to what's shown in the paper: for
each synthetic sequence we print the last CPU value, average CPU and the model
prediction (action + probability) and a human-readable scaling decision.
"""
import sys
import os
import numpy as np

# ensure project package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

try:
    import model_server_real as msr
except Exception:
    msr = None


def make_sequences(seq_len=10):
    seqs = []
    # Low constant
    seqs.append(np.clip( np.ones(seq_len) * 20 + np.random.randn(seq_len)*2, 0, 100))
    # High constant
    seqs.append(np.clip( np.ones(seq_len) * 80 + np.random.randn(seq_len)*3, 0, 100))
    # Spike
    s = np.ones(seq_len) * 30
    s[-1] = 95
    seqs.append(np.clip(s + np.random.randn(seq_len)*5, 0, 100))
    # Increasing trend
    seqs.append(np.clip(np.linspace(10, 90, seq_len) + np.random.randn(seq_len)*4, 0, 100))
    # Oscillating / bursty
    t = np.linspace(0, 4*np.pi, seq_len)
    seqs.append(np.clip(50 + 30*np.sin(t) + np.random.randn(seq_len)*5, 0, 100))
    return seqs


def run_demo():
    seqs = make_sequences(seq_len=12)

    # Prefer the real model wrapper from model_server_real if available
    model_wrapper = None
    if msr is not None and hasattr(msr, 'model'):
        model_wrapper = msr.model

    print('\nDemo: synthetic CPU traces -> model predictions\n')
    print(f"{'id':>2}  {'last%':>6}  {'avg%':>6}  {'action':>6}  {'decision':>10}  {'probs':>18}")
    print('-'*64)

    for i, seq in enumerate(seqs):
        last = float(seq[-1])
        avg = float(np.mean(seq))

        # If we have the model wrapper from model_server_real, use its predict API
        if model_wrapper is not None:
            try:
                # model_server_real expects obs shaped (1, -1) or similar
                action, _ = model_wrapper.predict(np.array([last]))
                # If wrapper is TorchWrapper, it doesn't expose probs; recompute
                probs = None
                try:
                    # Try to compute softmax probs if underlying module exists
                    import torch
                    from src.transformer_lstm import TransformerLSTM
                    # create tensor from full sequence to get logits
                    m = None
                    # If model_wrapper is a TorchWrapper it has .module
                    if hasattr(model_wrapper, 'module'):
                        m = model_wrapper.module
                    if m is not None:
                        arr = torch.tensor(seq.astype(np.float32)).reshape(1, -1, 1)
                        with torch.no_grad():
                            logits = m(arr)
                            probs_t = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            probs = probs_t.tolist()
                except Exception:
                    probs = None
            except Exception:
                # fallback to naive rule
                action = 1 if last > 50 else 0
                probs = None
        else:
            # No model wrapper available, use the simple threshold rule
            action = 1 if last > 50 else 0
            probs = None

        decision = 'scale_up' if int(action) == 1 else 'scale_down'
        probs_str = f"[{probs[0]:.2f}, {probs[1]:.2f}]" if probs is not None else "n/a"
        print(f"{i:2d}  {last:6.1f}  {avg:6.1f}  {int(action):6d}  {decision:10s}  {probs_str:>18s}")


if __name__ == '__main__':
    run_demo()
