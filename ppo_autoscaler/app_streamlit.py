"""Streamlit UI for the CPU autoscaler demo.

This app uses the `model_server_real` loader (will use the saved TransformerLSTM
checkpoint if present, otherwise falls back to the DummyModel). It provides:
- manual single-value prediction
- demo sequence generation and plotted predictions
"""
from __future__ import annotations

import sys
import os
from typing import Optional

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

try:
    import model_server_real as msr
    MODEL = msr.model
except Exception:
    MODEL = None


st.set_page_config(page_title="Smart Cloud Autoscaler", layout="centered")

# Card CSS to approximate the attached design
CARD_CSS = """
<style>
.card {max-width:760px; margin:28px auto; background: #fff; border-radius: 12px; box-shadow: 0 12px 30px rgba(16,24,40,0.08); padding: 26px}
.card .title {font-size:22px; font-weight:700; text-align:center; margin-bottom:18px}
.input-row {margin-bottom:14px}
.submit-btn {background:#2F80ED;color:#fff;border-radius:10px;padding:12px 18px;border:none;width:100%;font-weight:700}
.result-card {background:#f8fafc;border-radius:12px;padding:12px;margin-top:16px}
.result-row{display:flex;justify-content:space-between;padding:10px 6px;border-bottom:1px solid rgba(0,0,0,0.04)}
.result-row:last-child{border-bottom:0}
.badge-up{background:linear-gradient(90deg,#e6f4ff,#dff2ff);color:#1366d6;padding:6px 10px;border-radius:18px;font-weight:700}
.label{color:#6b7280}
.value{font-weight:700}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">Smart Cloud Autoscaler</div>', unsafe_allow_html=True)

with st.form('predict_form'):
    cpu_value = st.number_input('Enter CPU Utilization (%)', min_value=0.0, max_value=100.0, value=42.0, step=0.1, format='%.1f')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if MODEL is None:
            action = 1 if cpu_value > 50 else 0
            probs = None
        else:
            try:
                action, probs = MODEL.predict(np.array([cpu_value]))
            except Exception:
                action = 1 if cpu_value > 50 else 0
                probs = None

        # If model didn't return probs, try to compute them from a torch module if available
        if probs is None and MODEL is not None:
            try:
                import torch
                m = MODEL.module if hasattr(MODEL, 'module') else MODEL
                # try to build an input tensor shaped (1, seq_len?) -> here we pass single value as (1,1)
                inp = torch.tensor(np.array([[cpu_value]], dtype=np.float32))
                with torch.no_grad():
                    logits = m(inp)
                    if logits is not None:
                        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
            except Exception:
                probs = None
        decision = 'Scale Up' if int(action) == 1 else 'Scale Down'

        res_html = f"""
        <div class='result-card'>
          <div class='result-row'><div class='label'>CPU Utilization</div><div class='value'>{cpu_value:.1f}%</div></div>
          <div class='result-row'><div class='label'>PPO Action</div><div class='value'>{int(action)}</div></div>
          <div class='result-row'><div class='label'>Scaling Decision</div><div class='value'><span class='badge-up'>▲ {decision}</span></div></div>
        </div>
        """
        st.markdown(res_html, unsafe_allow_html=True)
        if probs is not None:
            try:
                st.markdown(f"**Probabilities:** {probs}")
            except Exception:
                st.write({'probs': probs})
        st.success(f"Decision: {decision} (action={int(action)})")

st.markdown('</div>', unsafe_allow_html=True)

# Show latest experiment metrics (if available)
try:
    QP = Path(ROOT) / 'experiments' / 'quick_results.json'
    if QP.exists():
        with open(QP, 'r') as fh:
            metrics = json.load(fh)
        # show summary KPIs at top
        try:
            tl = metrics.get('Transformer-LSTM', {})
            ppo = metrics.get('PPO', {})
            hybrid = metrics.get('Hybrid PPO+LSTM', {})
            c1, c2, c3 = st.columns(3)
            with c1:
                rmse = tl.get('RMSE', tl.get('rmse', '-'))
                st.metric('Regressor RMSE', f"{rmse}")
            with c2:
                pacc = ppo.get('Decision Accuracy (%)', ppo.get('decision_accuracy', '-'))
                st.metric('PPO Decision Acc (%)', f"{pacc}")
            with c3:
                hacc = hybrid.get('Decision Accuracy (%)', hybrid.get('decision_accuracy', '-'))
                st.metric('Hybrid Decision Acc (%)', f"{hacc}")

        except Exception:
            pass

        with st.expander('Latest experiment metrics', expanded=False):
            # metrics is a dict keyed by model name
            for k, v in metrics.items():
                st.subheader(k)
                # v is a mapping of metric name -> value
                rows = []
                for kk, vv in v.items():
                    # convert values to strings to avoid mixed-type serialization issues
                    rows.append({'metric': kk, 'value': str(vv)})
                st.table(rows)

        # allow download of the metrics JSON
        try:
            st.download_button('Download metrics JSON', data=json.dumps(metrics, indent=2), file_name='quick_results.json', mime='application/json')
        except Exception:
            pass
except Exception as e:
    # fail silently in the UI
    st.info('No experiment metrics available' if not Path(ROOT).exists() else f'Could not load metrics: {e}')

with st.expander("Demo sequences and plot"):
    seq_len = st.slider("Sequence length", min_value=6, max_value=64, value=12)
    n_seqs = st.slider("Number of demo sequences", min_value=1, max_value=8, value=5)
    if st.button("Generate and predict demo sequences"):
        # generate sequences similar to demo_run
        def make_sequences(seq_len=12):
            seqs = []
            seqs.append(np.clip(np.ones(seq_len) * 20 + np.random.randn(seq_len) * 2, 0, 100))
            seqs.append(np.clip(np.ones(seq_len) * 80 + np.random.randn(seq_len) * 3, 0, 100))
            s = np.ones(seq_len) * 30
            s[-1] = 95
            seqs.append(np.clip(s + np.random.randn(seq_len) * 5, 0, 100))
            seqs.append(np.clip(np.linspace(10, 90, seq_len) + np.random.randn(seq_len) * 4, 0, 100))
            t = np.linspace(0, 4 * np.pi, seq_len)
            seqs.append(np.clip(50 + 30 * np.sin(t) + np.random.randn(seq_len) * 5, 0, 100))
            return seqs[:n_seqs]

        seqs = make_sequences(seq_len)
        # create one subplot per generated sequence (keeps it aligned with seqs length)
        rows = max(1, len(seqs))
        fig, axes = plt.subplots(rows, 1, figsize=(6, 2.2 * rows), sharex=True)
        # normalize axes to a list for uniform indexing
        if rows == 1:
            axes = [axes]
        results = []
        for i, seq in enumerate(seqs):
            last = float(seq[-1])
            avg = float(np.mean(seq))
            if MODEL is not None:
                try:
                    action, _ = MODEL.predict(np.array([last]))
                except Exception:
                    action = 1 if last > 50 else 0
            else:
                action = 1 if last > 50 else 0
            results.append((i, last, avg, int(action)))
            # if for some reason axes is shorter than seqs, wrap around (defensive)
            if i >= len(axes):
                ax = axes[-1]
            else:
                ax = axes[i]
            ax.plot(seq, marker='o')
            ax.set_ylabel('CPU %')
            ax.set_title(f'seq {i} last={last:.1f} avg={avg:.1f} action={int(action)}')
            ax.axhline(50, color='gray', lw=0.5, linestyle='--')
        st.pyplot(fig)
        st.table([{"id": r[0], "last%": r[1], "avg%": r[2], "action": r[3], "decision": ('scale_up' if r[3]==1 else 'scale_down')} for r in results])

st.markdown("---")
st.markdown("If you want a standalone web UI, run:\n`streamlit run app_streamlit.py --server.port 8501`")
