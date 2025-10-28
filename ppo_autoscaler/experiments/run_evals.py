"""Run quick experiments to produce metrics for:
- Transformer-style regressor (LSTM/Transformer)
- PPO agent (baseline)
- Hybrid: PPO that receives LSTM/regressor forecast as extra observation

This script uses small synthetic datasets and short training runs to produce
quick, reproducible metrics for demo/paper table population.

NOTE: This is intentionally small and fast. For production/accurate results,
increase dataset size and training timesteps.
"""
import time
import math
import numpy as np
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

RESULTS = {}


def make_synthetic_sequences(n=2000, seq_len=10, seed=0):
    # allow scaling of noise for quick experiments via DATA_NOISE env var
    noise_scale = float(os.environ.get('DATA_NOISE', '1.0'))
    rng = np.random.RandomState(seed)
    X = []
    Y = []
    for _ in range(n):
        base = rng.uniform(10, 80)
        # drift is scaled by noise_scale (smaller -> easier task)
        drift = rng.normal(0, 1, size=seq_len) * noise_scale
        seq = np.clip(np.linspace(base - 10, base + 10, seq_len) + drift * 3 + rng.randn(seq_len) * 2 * noise_scale, 0, 100)
        X.append(seq.astype(np.float32))
        # target: next-step CPU (regression) with smaller noise when DATA_NOISE < 1
        next_val = np.clip(seq[-1] + rng.randn() * 3 * noise_scale + (rng.rand() - 0.5) * 5 * noise_scale, 0, 100)
        Y.append(np.float32(next_val))
    return np.stack(X), np.stack(Y)


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SmallRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=32, lstm_hidden=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)
        # expose repr dim for hybrid use
        self.repr_dim = lstm_hidden
        self._lstm_hidden = lstm_hidden

    def forward(self, x):
        # x: (batch, seq_len)
        b, s = x.shape
        t = x.reshape(b, s, 1)
        t = self.input_proj(t)
        t = t.transpose(0, 1)
        t = self.transformer(t)
        t = t.transpose(0, 1)
        out, _ = self.lstm(t)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

    def embed(self, x):
        """Return the learned representation (last LSTM hidden) for input x.

        x: numpy or torch tensor of shape (batch, seq_len) or (seq_len,) for single.
        Returns a numpy array (batch, repr_dim) or (repr_dim,) for single input.
        """
        single = False
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True
        x = x.to(next(self.parameters()).device)
        with torch.no_grad():
            b, s = x.shape
            t = x.reshape(b, s, 1)
            t = self.input_proj(t)
            t = t.transpose(0, 1)
            t = self.transformer(t)
            t = t.transpose(0, 1)
            out, _ = self.lstm(t)
            last = out[:, -1, :]
        if single:
            return last.squeeze(0).cpu().numpy()
        return last.cpu().numpy()


def train_regressor(X_train, Y_train, X_val, Y_val, device='cpu', epochs=6):
    model = SmallRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = SeqDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    loss_fn = nn.MSELoss()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - start

    # eval
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_val).to(device)
        yt = torch.from_numpy(Y_val).to(device)
        preds = model(xt).cpu().numpy()
    mse = float(np.mean((preds - Y_val) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - Y_val)))

    # latency: single prediction

    import timeit
    def single():
        with torch.no_grad():
            _ = model(torch.from_numpy(X_val[:1]).to(device))
    lat = timeit.timeit(single, number=50) / 50.0

    return dict(model=model, mse=mse, rmse=rmse, mae=mae, train_time=train_time, latency=lat)


# supervised classifier removed per user request
####################################
# PPO experiments
####################################
def make_env(seq_len=10):
    import gym
    from gym import spaces

    class CPUSimpleEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.seq_len = seq_len
            self.rng = np.random.RandomState()
            # observation: last seq_len cpu values
            self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(seq_len,), dtype=np.float32)
            # action: 0 scale_down, 1 scale_up
            self.action_space = spaces.Discrete(2)

        def reset(self):
            base = self.rng.uniform(10, 80)
            self.seq = np.clip(np.linspace(base - 10, base + 10, self.seq_len) + self.rng.randn(self.seq_len) * 2, 0, 100)
            return self.seq.astype(np.float32)

        def step(self, action):
            # generate next value
            next_val = np.clip(self.seq[-1] + self.rng.randn() * 3 + (self.rng.rand() - 0.5) * 5, 0, 100)
            # correct decision is whether next_val > 50
            correct = 1 if next_val > 50 else 0
            reward = 1.0 if action == correct else 0.0
            # shift sequence
            self.seq = np.roll(self.seq, -1)
            self.seq[-1] = next_val
            done = False
            info = {'correct': correct}
            return self.seq.astype(np.float32), reward, done, info

    return CPUSimpleEnv


def train_ppo(num_timesteps=8000, seed=0, seq_len=10, use_hybrid=False, regressor=None, episodes=300):
    """Fallback: simple REINFORCE-style policy gradient agent implemented in PyTorch.
    This avoids issues with stable-baselines3 on some macOS setups and provides
    comparable quick metrics for decision accuracy and latency.
    """
    # Simple MLP policy
    class Policy(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            # use a larger network to handle regressor features when hybrid
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        def forward(self, x):
            logits = self.net(x)
            return logits

    env_cls = make_env(seq_len=seq_len)
    env = env_cls()

    obs_dim = env.observation_space.shape[0]
    if use_hybrid:
        # rep_dim: number of features the regressor exposes
        repr_dim = getattr(regressor, 'repr_dim', 1) if regressor is not None else 0
        obs_dim += 1 + int(repr_dim)

    policy = Policy(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # If hybrid mode, do a behavioral cloning (BC) phase using regressor-augmented observations
    # This collects labeled (obs, correct) pairs from the env and trains the policy as a classifier
    # to jump-start performance before policy-gradient fine-tuning.
    if use_hybrid and regressor is not None:
        bc_samples = int(os.environ.get('HYBRID_BC_SAMPLES', '8000'))
        bc_epochs = int(os.environ.get('HYBRID_BC_EPOCHS', '40'))
        print(f'Hybrid BC: collecting {bc_samples} samples and training BC for {bc_epochs} epochs...')
        Xbc = []
        Ybc = []
        for i in range(bc_samples):
            obs0 = env.reset()
            o = obs0.astype(np.float32)
            # compute regressor forecast and representation
            with torch.no_grad():
                rp = float(regressor(torch.from_numpy(o.reshape(1, -1)).float()).cpu().numpy()[0])
                rvec = regressor.embed(o)
            # rvec may be 1d; ensure shape
            if rvec.ndim == 1:
                rvec_use = rvec
            else:
                rvec_use = rvec[0]
            o_in = np.concatenate([o, np.array([rp]), rvec_use], axis=-1)
            # step once to obtain the 'correct' label (independent of action)
            _, _, _, info = env.step(0)
            label = int(info.get('correct', 0))
            Xbc.append(o_in.astype(np.float32))
            Ybc.append(label)

        Xbc_t = torch.from_numpy(np.stack(Xbc)).float()
        Ybc_t = torch.from_numpy(np.array(Ybc, dtype=np.int64))
        from torch.utils.data import TensorDataset
        bc_ds = TensorDataset(Xbc_t, Ybc_t)
        bc_loader = DataLoader(bc_ds, batch_size=64, shuffle=True)

        bc_model = Policy(obs_dim)
        bc_opt = torch.optim.Adam(bc_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(bc_epochs):
            bc_model.train()
            for xb, yb in bc_loader:
                logits = bc_model(xb)
                loss = loss_fn(logits, yb)
                bc_opt.zero_grad()
                loss.backward()
                bc_opt.step()

        # copy BC weights into the main policy to initialize
        policy.load_state_dict(bc_model.state_dict())
        print('Hybrid BC training completed — proceeding to policy-gradient fine-tune')

    start = time.time()
    gamma = 0.99
    for ep in range(episodes):
        obs = env.reset()
        rewards = []
        logps = []
        for t in range(50):
            o = obs.astype(np.float32)
            if use_hybrid and regressor is not None:
                with torch.no_grad():
                    rp = float(regressor(torch.from_numpy(o.reshape(1, -1)).float()).cpu().numpy()[0])
                    rvec = regressor.embed(o)
                if rvec.ndim == 1:
                    rvec_use = rvec
                else:
                    rvec_use = rvec[0]
                o_in = np.concatenate([o, np.array([rp]), rvec_use], axis=-1)
            else:
                o_in = o

            o_t = torch.from_numpy(o_in.astype(np.float32)).unsqueeze(0)
            logits = policy(o_t)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            action = np.random.choice(2, p=probs)
            logp = torch.log_softmax(logits, dim=-1)[0, action]
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            logps.append(logp)
            if done:
                break

        # compute returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        # normalize
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, ret in zip(logps, returns):
            loss = loss - lp * ret
        opt.zero_grad()
        loss.backward()
        opt.step()

    train_time = time.time() - start

    # evaluate decision accuracy on 500 episodes
    n = 500
    correct = 0
    lat_times = []
    for i in range(n):
        obs = env.reset()
        o = obs.astype(np.float32)
        if use_hybrid and regressor is not None:
            with torch.no_grad():
                rp = float(regressor(torch.from_numpy(o.reshape(1, -1)).float()).cpu().numpy()[0])
                rvec = regressor.embed(o)
            if rvec.ndim == 1:
                rvec_use = rvec
            else:
                rvec_use = rvec[0]
            o_in = np.concatenate([o, np.array([rp]), rvec_use], axis=-1)
        else:
            o_in = o
        o_t = torch.from_numpy(o_in.astype(np.float32)).unsqueeze(0)
        t0 = time.time()
        logits = policy(o_t)
        act = int(torch.argmax(logits, dim=-1).item())
        lat_times.append(time.time() - t0)
        obs2, reward, done, info = env.step(act)
        corr = info.get('correct', 0) if isinstance(info, dict) else 0
        if int(act) == int(corr):
            correct += 1

    acc = correct / n * 100.0
    latency = float(np.mean(lat_times)) if lat_times else 0.0
    return dict(train_time=train_time, decision_accuracy=acc, latency=latency)


def run_all():
    print('Preparing data...')
    n = int(os.environ.get('DATA_N', '1200'))
    X, Y = make_synthetic_sequences(n=n, seq_len=10)
    X_train, Y_train = X[:900], Y[:900]
    X_val, Y_val = X[900:1100], Y[900:1100]
    X_test, Y_test = X[1100:], Y[1100:]

    print('Training regressor (Transformer+LSTM)...')
    # default quick run uses 6 epochs; for paper-quality increase epochs
    epochs = int(os.environ.get('REGR_EPOCHS', '6'))
    R = train_regressor(X_train, Y_train, X_test, Y_test, epochs=epochs)
    print('Regressor metrics:', R['mse'], R['rmse'], R['mae'])
    RESULTS['lstm'] = R

    # supervised classifier experiment removed (per user request)

    print('Training PPO baseline...')
    ppo_episodes = int(os.environ.get('PPO_EPISODES', '300'))
    P = train_ppo(num_timesteps=6000, seed=0, seq_len=10, use_hybrid=False, episodes=ppo_episodes)
    print('PPO metrics:', P)
    RESULTS['ppo'] = P

    print('Training Hybrid PPO (with regressor)...')
    H = train_ppo(num_timesteps=6000, seed=1, seq_len=10, use_hybrid=True, regressor=R['model'], episodes=ppo_episodes)
    print('Hybrid metrics:', H)
    RESULTS['hybrid'] = H

    # format results into readable table-like dict
    out = {}
    # LSTM regressor
    out['Transformer-LSTM'] = {
        'MSE': R['mse'],
        'RMSE': R['rmse'],
        'MAE': R['mae'],
        'Training Time (s)': R['train_time'],
        'Prediction Latency (s)': R['latency'],
        'Decision Accuracy (%)': '-',
        'Avg Reward per Episode': '-'
    }
    out['PPO'] = {
        'MSE': '-',
        'RMSE': '-',
        'MAE': '-',
        'Training Time (s)': P['train_time'],
        'Prediction Latency (s)': P['latency'],
        'Decision Accuracy (%)': P['decision_accuracy'],
        'Avg Reward per Episode': '-'  # could compute if we collected rewards
    }
    out['Hybrid PPO+LSTM'] = {
        'MSE': '-',
        'RMSE': '-',
        'MAE': '-',
        'Training Time (s)': H['train_time'],
        'Prediction Latency (s)': H['latency'],
        'Decision Accuracy (%)': H['decision_accuracy'],
        'Avg Reward per Episode': '-'
    }
    # Supervised classifier entry removed

    print('\nFinal summary:')
    for k, v in out.items():
        print('\n==', k)
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f'  {kk:25}: {vv:.4f}')
            else:
                print(f'  {kk:25}: {vv}')

    return out


if __name__ == '__main__':
    res = run_all()
    # save to file
    import json
    (ROOT / 'experiments').mkdir(exist_ok=True)
    with open(ROOT / 'experiments' / 'quick_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('\nSaved results to experiments/quick_results.json')
