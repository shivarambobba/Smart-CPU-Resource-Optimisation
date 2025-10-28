"""Multi-step timeline prediction with uncertainty for the Transformer-LSTM regressor.

Approach:
- Load SmallRegressor and checkpoint if available.
- Generate synthetic dataset (same generator as experiments/run_evals.py) and use the validation set to estimate residual standard deviation.
- For selected test sequences, perform M Monte-Carlo rollouts of H-step iterative forecasting by feeding predicted values back into the sequence and adding Gaussian noise with sigma estimated from validation residuals.
- Compute mean, std, and 95% CI per horizon step and save results to JSON.

Usage:
  DATA_N=1200 DATA_NOISE=1.0 python3 experiments/timeline_predict.py

"""
import json
import os
from pathlib import Path
import numpy as np
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_evals import make_synthetic_sequences, SmallRegressor

ROOT = Path(__file__).resolve().parents[1]

def estimate_residual_std(model, X_val, Y_val, device='cpu'):
    model.to(device).eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
    residuals = preds - Y_val
    sigma = float(np.std(residuals))
    return sigma, residuals


def mc_multi_step(model, x0, H=10, M=200, sigma=1.0, device='cpu'):
    """Perform M rollouts of H steps starting from x0 (shape: seq_len, numpy).
    Returns array of shape (M, H) with predicted future values.
    """
    seq_len = x0.shape[0]
    results = np.zeros((M, H), dtype=np.float32)
    model.to(device).eval()
    for m in range(M):
        seq = x0.copy().astype(np.float32)
        preds = []
        for h in range(H):
            with torch.no_grad():
                p = model(torch.from_numpy(seq.reshape(1, -1)).float().to(device)).cpu().numpy()[0]
            # add Gaussian noise to simulate uncertainty
            noisy_p = p + np.random.randn() * sigma
            # clip to [0,100]
            noisy_p = float(np.clip(noisy_p, 0.0, 100.0))
            preds.append(noisy_p)
            # append to sequence (roll) for next-step prediction
            seq = np.roll(seq, -1)
            seq[-1] = noisy_p
        results[m] = np.array(preds, dtype=np.float32)
    return results


def summarize_rollouts(rollouts):
    # rollouts: (M, H)
    mean = float(np.mean(rollouts, axis=0).tolist())
    std = float(np.std(rollouts, axis=0).tolist())
    lower = np.percentile(rollouts, 2.5, axis=0).tolist()
    upper = np.percentile(rollouts, 97.5, axis=0).tolist()
    return dict(mean=np.mean(rollouts, axis=0).tolist(), std=np.std(rollouts, axis=0).tolist(), ci_lower=lower, ci_upper=upper)


def main():
    n = int(os.environ.get('DATA_N', '1200'))
    noise = float(os.environ.get('DATA_NOISE', '1.0'))
    H = int(os.environ.get('PRED_HORIZON', '10'))
    M = int(os.environ.get('MC_SAMPLES', '200'))

    print('Generating data (n=', n, ' noise=', noise, ')')
    X, Y = make_synthetic_sequences(n=n, seq_len=10, seed=0)
    X_train, Y_train = X[:900], Y[:900]
    X_val, Y_val = X[900:1100], Y[900:1100]
    X_test, Y_test = X[1100:], Y[1100:]

    model = SmallRegressor()
    ckpt = ROOT / 'models' / 'transformer_lstm.pth'
    if ckpt.exists():
        print('Loading checkpoint', ckpt)
        try:
            chk = torch.load(ckpt, map_location='cpu')
            if isinstance(chk, dict) and 'state_dict' in chk:
                state = chk['state_dict']
            else:
                state = chk
            model.load_state_dict(state)
            loaded_model = True
        except Exception as e:
            print('Warning: failed to load checkpoint into SmallRegressor:', e)
            print('Falling back to a simple statistical baseline for timeline predictions.')
            loaded_model = False
    else:
        print('No checkpoint found at', ckpt, '- using freshly initialized model')
        loaded_model = False

    sigma, residuals = estimate_residual_std(model, X_val, Y_val, device='cpu')
    print('Estimated residual std from validation set:', sigma)

    results = {
        'meta': {
            'DATA_N': n,
            'DATA_NOISE': noise,
            'PRED_HORIZON': H,
            'MC_SAMPLES': M,
            'residual_std': sigma,
        },
        'samples': []
    }

    # choose first 5 test samples to demo
    K = min(5, len(X_test))
    for i in range(K):
        x0 = X_test[i]
        y_true = []
        # create true future by rolling generator (we don't have oracle future beyond next-step, so we just use ground truth next for step 0)
        # For demonstration, we'll report the actual next-step (Y_test[i]) and otherwise None.
        y0 = float(Y_test[i])
        if loaded_model:
            rollouts = mc_multi_step(model, x0, H=H, M=M, sigma=sigma, device='cpu')
            summary = summarize_rollouts(rollouts)
        else:
            # statistical baseline: estimate mean increment and variance from training set deltas
            deltas = X_train[:, 1:] - X_train[:, :-1]
            mean_delta = float(np.mean(deltas))
            var_delta = float(np.var(deltas))
            # forecast mean and variance for each horizon step (random walk assumption)
            means = []
            lowers = []
            uppers = []
            stds = []
            last = float(x0[-1])
            for h in range(1, H+1):
                m = last + h * mean_delta
                var = h * var_delta + sigma**2  # include single-step residual variance
                sd = float(np.sqrt(var))
                means.append(float(m))
                stds.append(sd)
                lower = float(m - 1.96 * sd)
                upper = float(m + 1.96 * sd)
                lowers.append(lower)
                uppers.append(upper)
            summary = dict(mean=means, std=stds, ci_lower=lowers, ci_upper=uppers)
        sample_entry = {
            'index': int(i),
            'x0_last': float(x0[-1]),
            'true_next': y0,
            'horizon': H,
            'mc_samples': M,
            'summary': summary,
        }
        results['samples'].append(sample_entry)

        print(f'== Sample {i} (last observed={x0[-1]:.3f}, true next={y0:.3f})')
        for h in range(H):
            mval = summary['mean'][h]
            l = summary['ci_lower'][h]
            u = summary['ci_upper'][h]
            print(f'  t+{h+1}: mean={mval:.3f}  95% CI=({l:.3f},{u:.3f})')

    outp = ROOT / 'experiments' / 'timeline_predictions.json'
    (ROOT / 'experiments').mkdir(exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved timeline predictions to', outp)

if __name__ == '__main__':
    main()
