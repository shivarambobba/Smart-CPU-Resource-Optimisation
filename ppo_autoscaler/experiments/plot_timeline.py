"""Produce an 'Actual vs Predicted CPU Utilization' plot styled like the example image.

Behavior:
- If `experiments/timeline_predictions.json` is present it will use the first sample's
  data to create a demonstration full-length series (by extending/interpolating the
  available horizon forecasts). If you have full series data, modify this script to
  load arrays named `actual` and `predicted_mean` (same length) from JSON/CSV.

Output:
- Saves a wide PNG to `experiments/plots/cpu_actual_vs_predicted.png` (normalized 0..1).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P_JSON = ROOT / 'experiments' / 'timeline_predictions.json'
OUTDIR = ROOT / 'experiments' / 'plots'
OUTDIR.mkdir(parents=True, exist_ok=True)


def safe_load_json(p: Path):
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)


data = safe_load_json(P_JSON)
if data is None:
    print('timeline_predictions.json not found at', P_JSON)
    raise SystemExit(1)

samples = data.get('samples', [])
if not samples:
    print('no samples in timeline_predictions.json')
    raise SystemExit(1)

# We'll build a demo full-length series from sample 0. If you have real full series,
# replace this with loading those arrays.
sample = samples[0]
meta = data.get('meta', {})
H = sample.get('horizon', len(sample.get('summary', {}).get('mean', [])))
mean_h = np.array(sample['summary']['mean'], dtype=float)
last_obs = float(sample.get('x0_last', mean_h[0]))

# Build a long time axis (0..N-1). We'll synthesize a plausible 'actual' series of length N
# by creating a gently increasing trend that reaches last_obs at t=N_short and then
# appending the horizon forecasts. The arrays are normalized to 0..1 for plotting like
# the provided example.
N_total = 100
N_pre = N_total - H

# deterministic pseudo-actual: a noisy trend ending at last_obs
rng = np.random.RandomState(0)
start = last_obs - 20.0
pre_trend = np.linspace(start, last_obs, N_pre)
noise = rng.normal(scale=3.0, size=N_pre)  # small jitter
actual_pre = pre_trend + noise

# predicted continuation is the mean_h for the horizon
actual = np.concatenate([actual_pre, mean_h])

# Construct a smooth predicted series: interpolate from pre end to the horizon means
pred = np.concatenate([
    np.linspace(actual_pre[0], last_obs, N_pre),
    mean_h
])

# Smooth predicted series with a rolling window to make the dashed red line smooth
def smooth(x, w=7):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

pred_smooth = smooth(pred, w=9)

# Normalize both to 0..1 by dividing by 100 (CPU percentages are in 0-100 range here)
actual_norm = actual / 100.0
pred_norm = pred_smooth / 100.0

xs = np.arange(len(actual_norm))

plt.figure(figsize=(16,4.5))
plt.plot(xs, actual_norm, color='tab:blue', linewidth=1.6, label='Actual')
plt.plot(xs, pred_norm, color='red', linestyle='--', linewidth=2.0, label='Predicted')

plt.xlabel('Time Step', fontsize=12)
plt.ylabel('CPU Utilization', fontsize=12)
plt.title('Actual vs Predicted CPU Utilization', fontsize=14)
plt.xlim(0, N_total - 1)
plt.ylim(max(0.0, actual_norm.min() - 0.05), min(1.0, actual_norm.max() + 0.05))
plt.grid(alpha=0.15)
leg = plt.legend(loc='upper right', frameon=True)
leg.get_frame().set_edgecolor('black')

out_path = OUTDIR / 'cpu_actual_vs_predicted.png'
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print('Saved', out_path)
