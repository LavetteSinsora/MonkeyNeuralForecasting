"""
Results & Evaluation Page — Analyze model predictions in depth.
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard.dashboard_utils import (
    scan_checkpoints, load_checkpoint, collect_predictions,
    get_available_monkeys, detect_device,
)

st.title("Results & Evaluation")

# ---------- Checkpoint Selection ----------
checkpoints = scan_checkpoints()

if not checkpoints:
    st.warning(
        "No checkpoints found. Train a model first or place .pth files under "
        "replication/*/checkpoints/ or experiments/*/checkpoints/."
    )
    st.stop()

ckpt_labels = [c['label'] for c in checkpoints]
selected_idx = st.sidebar.selectbox(
    "Checkpoint",
    range(len(ckpt_labels)),
    format_func=lambda i: ckpt_labels[i],
)
selected_ckpt = checkpoints[selected_idx]

# Monkey selector (auto-detected, editable)
detected_monkey = selected_ckpt.get('monkey')
monkeys = get_available_monkeys()
default_monkey_idx = monkeys.index(detected_monkey) if detected_monkey in monkeys else 0
monkey = st.sidebar.selectbox("Monkey", monkeys, index=default_monkey_idx)

device = detect_device()


# ---------- Load & Evaluate ----------
@st.cache_resource(show_spinner="Loading model and computing predictions...")
def _load_and_predict(ckpt_path: str, monkey: str, device: str):
    """Load checkpoint, run predictions on validation set, cache results."""
    from utils.data import get_dataloaders
    from utils.metrics import compute_all_metrics

    model, ckpt_data = load_checkpoint(ckpt_path, monkey, device=device)
    _, val_loader, stats = get_dataloaders(monkey, batch_size=64, val_fraction=0.15, seed=42)
    preds, targets = collect_predictions(model, val_loader, device=device)
    metrics = compute_all_metrics(preds, targets)
    return preds, targets, metrics, stats


load_btn = st.sidebar.button("Load & Evaluate", type="primary")

# Use session state to persist results across interactions
if load_btn:
    st.session_state['results_data'] = _load_and_predict(
        selected_ckpt['path'], monkey, device
    )
    st.session_state['results_monkey'] = monkey

if 'results_data' not in st.session_state:
    st.info("Select a checkpoint and click **Load & Evaluate** to begin.")
    st.stop()

preds, targets, metrics, norm_stats = st.session_state['results_data']
N, T, C = preds.shape  # (N_samples, 10_pred_steps, C_channels)

# ---------- Summary Metrics Row ----------
st.subheader("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE (norm)", f"{metrics['mse']:.6f}")
col2.metric("RMSE (norm)", f"{metrics['rmse']:.6f}")
col3.metric("R\u00b2 (mean)", f"{metrics['r2']['mean']:.4f}")
col4.metric("Correlation (mean)", f"{metrics['correlation']['mean']:.4f}")

st.divider()

# ---------- Trial Selector ----------
trial_idx = st.number_input("Trial (sample) index", min_value=0, max_value=N - 1, value=0, step=1)

# ---------- Per-channel MSE for sorting ----------
ch_mse = metrics['per_channel_mse']  # (C,)
sorted_ch_indices = np.argsort(ch_mse)  # best (lowest MSE) first

# ---------- Channel Selector ----------
st.sidebar.subheader("Channel Selection")
n_show = st.sidebar.slider("Channels to display", 1, min(20, C), value=min(4, C))

channel_mode = st.sidebar.radio(
    "Selection mode",
    ["Manual", "Show Best N", "Show Worst N"],
    index=1,
)

if channel_mode == "Manual":
    selected_channels = st.sidebar.multiselect(
        "Channels",
        list(range(C)),
        default=list(range(min(n_show, C))),
    )
elif channel_mode == "Show Best N":
    selected_channels = sorted_ch_indices[:n_show].tolist()
    st.sidebar.caption(f"Best {n_show} channels by MSE: {selected_channels}")
else:
    selected_channels = sorted_ch_indices[-n_show:][::-1].tolist()
    st.sidebar.caption(f"Worst {n_show} channels by MSE: {selected_channels}")

if not selected_channels:
    st.warning("Please select at least one channel.")
    st.stop()

# ---------- Main Visualization: GT vs Predicted LMP ----------
st.subheader(f"Prediction vs Ground Truth  (Trial {trial_idx})")

time_ms = np.arange(T) * 30  # 0, 30, 60, ... 270 ms

n_cols = min(len(selected_channels), 3)
n_rows = (len(selected_channels) + n_cols - 1) // n_cols

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"Ch {ch} (MSE={ch_mse[ch]:.5f})" for ch in selected_channels],
    vertical_spacing=0.08,
    horizontal_spacing=0.06,
)

for i, ch in enumerate(selected_channels):
    row = i // n_cols + 1
    col = i % n_cols + 1
    gt = targets[trial_idx, :, ch]
    pr = preds[trial_idx, :, ch]

    fig.add_trace(
        go.Scatter(x=time_ms, y=gt, mode='lines+markers', name=f'GT ch{ch}',
                   line=dict(color='steelblue'), showlegend=(i == 0)),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=time_ms, y=pr, mode='lines+markers', name=f'Pred ch{ch}',
                   line=dict(color='tomato', dash='dash'), showlegend=(i == 0)),
        row=row, col=col,
    )
    fig.update_xaxes(title_text="Time (ms)", row=row, col=col)
    fig.update_yaxes(title_text="LMP (norm)", row=row, col=col)

fig.update_layout(
    height=300 * n_rows,
    title_text="Ground Truth (blue) vs Prediction (red dashed)",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Per-Channel MSE Bar Chart ----------
st.subheader("Per-Channel MSE")

ch_mse_sorted_idx = np.argsort(ch_mse)[::-1]  # worst first for bar chart
ch_colors = ['tomato' if i in selected_channels else 'steelblue' for i in ch_mse_sorted_idx]

fig_bar = go.Figure(
    go.Bar(
        x=[f"Ch {i}" for i in ch_mse_sorted_idx],
        y=ch_mse[ch_mse_sorted_idx],
        marker_color=ch_colors,
    )
)
fig_bar.update_layout(
    title="Per-Channel MSE (sorted, selected channels highlighted in red)",
    xaxis_title="Channel",
    yaxis_title="MSE",
    height=400,
    xaxis=dict(tickangle=45, dtick=max(1, C // 40)),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------- Per-Timestep MSE Curve ----------
st.subheader("Per-Timestep MSE (Error Growth Over Horizon)")

ts_mse = metrics['per_timestep_mse']  # (10,)
fig_ts = go.Figure(
    go.Scatter(
        x=time_ms, y=ts_mse,
        mode='lines+markers',
        marker=dict(size=8, color='steelblue'),
        line=dict(width=2),
    )
)
fig_ts.update_layout(
    title="MSE at Each Prediction Timestep",
    xaxis_title="Prediction Time (ms)",
    yaxis_title="MSE",
    height=350,
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------- Correlation Matrix Comparison ----------
st.subheader(f"Inter-Channel Correlation Structure (Trial {trial_idx})")

gt_trial = targets[trial_idx]   # (10, C)
pr_trial = preds[trial_idx]     # (10, C)

# Compute correlation matrices for this trial
def _corr_matrix(arr):
    """Compute (C, C) Pearson correlation from (T, C) array."""
    centered = arr - arr.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=0, keepdims=True) + 1e-12)
    normed = centered / norms
    return normed.T @ normed

corr_gt = _corr_matrix(gt_trial)
corr_pr = _corr_matrix(pr_trial)

col_left, col_right = st.columns(2)

with col_left:
    fig_cgt = px.imshow(
        corr_gt, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title="Ground Truth Channel Correlations",
        labels=dict(x="Channel", y="Channel"),
    )
    fig_cgt.update_layout(height=500)
    st.plotly_chart(fig_cgt, use_container_width=True)

with col_right:
    fig_cpr = px.imshow(
        corr_pr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title="Predicted Channel Correlations",
        labels=dict(x="Channel", y="Channel"),
    )
    fig_cpr.update_layout(height=500)
    st.plotly_chart(fig_cpr, use_container_width=True)

# Correlation difference
st.markdown("**Correlation difference (GT - Pred):**")
diff = corr_gt - corr_pr
fig_diff = px.imshow(
    diff, color_continuous_scale='RdBu_r',
    zmin=-np.abs(diff).max(), zmax=np.abs(diff).max(),
    title="Correlation Difference (GT - Pred)",
    labels=dict(x="Channel", y="Channel"),
)
fig_diff.update_layout(height=450)
st.plotly_chart(fig_diff, use_container_width=True)
