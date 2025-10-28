"""FastAPI model server that attempts to load the real TransformerLSTM model.

If a checkpoint is present at `models/transformer_lstm.pth` it will be loaded.
Otherwise the server falls back to the lightweight DummyModel used elsewhere
in the repo.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from pathlib import Path

app = FastAPI(title='CPU Autoscaler (real-model loader)')


class PredictRequest(BaseModel):
    cpu: float


class DummyModel:
    def predict(self, obs, deterministic=True):
        try:
            cpu = float(np.array(obs).flatten()[0])
        except Exception:
            cpu = 0.0
        action = 1 if cpu > 50 else 0
        return int(action), None


# Try to load real PyTorch model if available
model = DummyModel()
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'transformer_lstm.pth'
if MODEL_PATH.exists():
    try:
        import torch
        from src.transformer_lstm import TransformerLSTM

        device = torch.device('cpu')
        m = TransformerLSTM(input_dim=1, d_model=32, nhead=4, num_layers=1, lstm_hidden=32, num_actions=2)
        ckpt = torch.load(MODEL_PATH, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        m.load_state_dict(state)
        m.eval()

        class TorchWrapper:
            def __init__(self, module):
                self.module = module

            def predict(self, obs, deterministic=True):
                # obs: numpy array (1,) or (1,1)
                import torch as _t
                arr = _t.tensor(np.array(obs, dtype=np.float32)).reshape(1, -1, 1)
                with _t.no_grad():
                    logits = self.module(arr)
                    probs = _t.softmax(logits, dim=-1)
                    action = int(_t.argmax(probs, dim=-1).cpu().numpy()[0])
                return action, None

        model = TorchWrapper(m)
        print('Loaded TransformerLSTM from', MODEL_PATH)
    except Exception as e:
        print('Failed to load real model:', e)


@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'real' if MODEL_PATH.exists() else 'dummy'}


@app.post('/predict')
def predict(req: PredictRequest):
    try:
        obs = np.array([req.cpu], dtype=np.float32).reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        scaling_action = 'scale_up' if action == 1 else 'scale_down'
        return {
            'cpu_input': float(req.cpu),
            'ppo_action': int(action),
            'scaling_decision': scaling_action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
