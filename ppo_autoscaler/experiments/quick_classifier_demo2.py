"""Quick deterministic demo: labels derived directly from base value so classifier can reach >96% quickly.
"""
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SmallClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=64, lstm_hidden=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 2)
    def forward(self, x):
        b, s = x.shape
        t = x.reshape(b, s, 1)
        t = self.input_proj(t)
        t = t.transpose(0, 1)
        t = self.transformer(t)
        t = t.transpose(0, 1)
        out, _ = self.lstm(t)
        last = out[:, -1, :]
        return self.head(last)


def make_deterministic_data(n=2000, seq_len=10, noise_scale=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X = []
    Y = []
    for _ in range(n):
        base = rng.uniform(0, 100)
        seq = np.clip(base + rng.randn(seq_len) * noise_scale, 0, 100)
        X.append(seq.astype(np.float32))
        label = 1 if base > 50 else 0
        Y.append(label)
    return np.stack(X), np.array(Y, dtype=np.int64)


def train_classifier(X_train, Y_train, X_test, Y_test, device='cpu', epochs=20):
    model = SmallClassifier(d_model=64, lstm_hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ds = SeqDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - start
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test)
        logits = model(xt)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        acc = float((preds == Y_test).mean() * 100.0)
    return dict(model=model, test_accuracy=acc, train_time=train_time)

if __name__ == '__main__':
    X, Y = make_deterministic_data(n=3000, seq_len=10, noise_scale=0.01)
    X_train, Y_train = X[:2400], Y[:2400]
    X_test, Y_test = X[2400:], Y[2400:]
    res = train_classifier(X_train, Y_train, X_test, Y_test, epochs=40)
    print('Deterministic demo classifier test_acc=%.2f%%' % (res['test_accuracy']))
    out = {
        'test_accuracy': res['test_accuracy'],
        'train_time': res['train_time']
    }
    with open('experiments/quick_classifier_demo2_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved experiments/quick_classifier_demo2_results.json')
