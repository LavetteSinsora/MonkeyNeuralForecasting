"""
Model Diagnosis Page — Inspect learned representations and model internals.
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.dashboard_utils import (
    scan_checkpoints, load_checkpoint, collect_predictions,
    collect_hidden_states, get_available_monkeys, detect_device,
)

st.title("Model Diagnosis")

# ---------- Checkpoint Selection ----------
checkpoints = scan_checkpoints()

if not checkpoints:
    st.warning("No checkpoints found. Train a model first.")
    st.stop()

ckpt_labels = [c['label'] for c in checkpoints]
selected_idx = st.sidebar.selectbox(
    "Checkpoint",
    range(len(ckpt_labels)),
    format_func=lambda i: ckpt_labels[i],
)
selected_ckpt = checkpoints[selected_idx]

detected_monkey = selected_ckpt.get('monkey')
monkeys = get_available_monkeys()
default_monkey_idx = monkeys.index(detected_monkey) if detected_monkey in monkeys else 0
monkey = st.sidebar.selectbox("Monkey", monkeys, index=default_monkey_idx)

device = detect_device()

FEATURE_NAMES = [
    "LMP", "PB 0.5-4Hz", "PB 4-8Hz", "PB 8-12Hz",
    "PB 12-25Hz", "PB 25-50Hz", "PB 50-100Hz", "PB 100-200Hz", "PB 200-400Hz",
]


# ---------- Load Model & Data ----------
@st.cache_resource(show_spinner="Loading model and data...")
def _load_model_and_data(ckpt_path: str, monkey: str, device: str):
    from utils.data import get_dataloaders

    model, ckpt_data = load_checkpoint(ckpt_path, monkey, device=device)
    _, val_loader, stats = get_dataloaders(monkey, batch_size=64, val_fraction=0.15, seed=42)

    # Collect all validation inputs and targets
    all_X, all_Y = [], []
    for X, Y in val_loader:
        all_X.append(X.numpy())
        all_Y.append(Y.numpy())
    X_all = np.concatenate(all_X, axis=0)  # (N, 10, C, 9)
    Y_all = np.concatenate(all_Y, axis=0)  # (N, 10, C)

    # Also collect predictions
    preds_list = []
    model.eval()
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            pred = model(X)
            preds_list.append(pred.cpu().numpy())
    P_all = np.concatenate(preds_list, axis=0)

    return model, X_all, Y_all, P_all, stats


load_btn = st.sidebar.button("Load Model", type="primary")

if load_btn:
    st.session_state['diag_data'] = _load_model_and_data(
        selected_ckpt['path'], monkey, device
    )
    st.session_state['diag_monkey'] = monkey

if 'diag_data' not in st.session_state:
    st.info("Select a checkpoint and click **Load Model** to begin diagnosis.")
    st.stop()

model, X_all, Y_all, P_all, norm_stats = st.session_state['diag_data']
N, T_ctx, C, F = X_all.shape  # (N, 10, C, 9)
T_pred = Y_all.shape[1]  # 10

# ---------- Trial & Channel Selectors ----------
trial_idx = st.sidebar.number_input("Trial index", min_value=0, max_value=N - 1, value=0, step=1)
channel_idx = st.sidebar.number_input("Channel index", min_value=0, max_value=C - 1, value=0, step=1)

# =====================================================================
# 1. Input Features Heatmap
# =====================================================================
st.subheader(f"Input Features Heatmap (Trial {trial_idx}, Channel {channel_idx})")
st.caption("Shows all 9 input features across the 10 context timesteps.")

input_data = X_all[trial_idx, :, channel_idx, :]  # (10, 9)
time_labels = [f"t={i}" for i in range(T_ctx)]

fig_input = px.imshow(
    input_data.T,
    x=time_labels,
    y=FEATURE_NAMES,
    color_continuous_scale='Viridis',
    aspect='auto',
    title=f"Input Features: Trial {trial_idx}, Channel {channel_idx}",
    labels=dict(x="Timestep", y="Feature", color="Value"),
)
fig_input.update_layout(height=350)
st.plotly_chart(fig_input, use_container_width=True)

# =====================================================================
# 2. Adjacency Matrix Visualization
# =====================================================================
st.subheader("Learned Adjacency Matrices")
st.caption("A_a (additive) and A_m (multiplicative) matrices, compared to initial correlation.")

try:
    A_a = model.si.A_a.detach().cpu().numpy()
    A_m = model.si.A_m.detach().cpu().numpy()

    col1, col2 = st.columns(2)

    with col1:
        fig_aa = px.imshow(
            np.tanh(A_a),
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="tanh(A_a) — Additive Adjacency",
            labels=dict(x="Channel", y="Channel"),
        )
        fig_aa.update_layout(height=480)
        st.plotly_chart(fig_aa, use_container_width=True)

    with col2:
        fig_am = px.imshow(
            np.tanh(A_m),
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="tanh(A_m) — Multiplicative Adjacency",
            labels=dict(x="Channel", y="Channel"),
        )
        fig_am.update_layout(height=480)
        st.plotly_chart(fig_am, use_container_width=True)

    # Show initial correlation matrix for comparison
    st.markdown("**Initial Correlation Matrix (from training data LMP):**")
    try:
        try:
            from replications.amag.model import compute_correlation_matrix
        except ImportError:
            from replication.amag.model import compute_correlation_matrix
        init_corr = compute_correlation_matrix(st.session_state.get('diag_monkey', 'affi'))
        fig_init = px.imshow(
            init_corr,
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="Initial LMP Correlation Matrix",
            labels=dict(x="Channel", y="Channel"),
        )
        fig_init.update_layout(height=480)
        st.plotly_chart(fig_init, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute initial correlation matrix: {e}")

except AttributeError:
    st.warning("Model does not have accessible A_a / A_m attributes (spatial interaction module).")

# =====================================================================
# 3. Beta Weights
# =====================================================================
st.subheader("Learned Beta Combination Weights")
st.caption("Controls the balance between self, additive, and multiplicative pathways.")

try:
    import torch.nn.functional as F_torch

    beta_raw = model.si.beta.detach().cpu()
    n_active = 1 + int(model.si.use_add) + int(model.si.use_mul)
    beta_norm = F_torch.softmax(beta_raw[:n_active], dim=0).numpy()

    labels = ["Self (identity)"]
    if model.si.use_add:
        labels.append("Additive")
    if model.si.use_mul:
        labels.append("Multiplicative")

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        for lbl, val in zip(labels, beta_norm):
            st.metric(lbl, f"{val:.4f}")
        st.caption(f"Raw beta: {beta_raw[:n_active].numpy()}")

    with col_b2:
        fig_beta = go.Figure(
            go.Bar(x=labels, y=beta_norm, marker_color=['steelblue', 'tomato', 'seagreen'][:n_active])
        )
        fig_beta.update_layout(
            title="Normalized Beta Weights (softmax)",
            yaxis_title="Weight", height=300,
        )
        st.plotly_chart(fig_beta, use_container_width=True)

except AttributeError:
    st.warning("Model does not have accessible beta weights.")

# =====================================================================
# 4. Hidden State Visualization
# =====================================================================
st.subheader(f"Hidden States (Trial {trial_idx})")
st.caption("Intermediate representations after Temporal Encoding (TE) and Spatial Interaction (SI).")

try:
    # Get a single trial as a batch of 1
    X_single = torch.tensor(X_all[trial_idx:trial_idx + 1], dtype=torch.float32)
    states = collect_hidden_states(model, X_single, device=device)

    # TE output: (1, T, C, d) -> (T, C) take mean over hidden dim
    te_out = states['te_output'][0]  # (T, C, d)
    si_out = states['si_output'][0]  # (T, C, d)

    # Show as heatmaps: mean activation magnitude per (timestep, channel)
    te_mag = np.sqrt((te_out ** 2).mean(axis=-1))  # (T, C)
    si_mag = np.sqrt((si_out ** 2).mean(axis=-1))  # (T, C)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig_te = px.imshow(
            te_mag,
            x=[f"Ch {i}" for i in range(C)],
            y=[f"t={i}" for i in range(te_mag.shape[0])],
            color_continuous_scale='Viridis',
            aspect='auto',
            title="TE Output: RMS Activation per (t, ch)",
            labels=dict(x="Channel", y="Timestep", color="RMS"),
        )
        fig_te.update_layout(height=400)
        st.plotly_chart(fig_te, use_container_width=True)

    with col_h2:
        fig_si = px.imshow(
            si_mag,
            x=[f"Ch {i}" for i in range(C)],
            y=[f"t={i}" for i in range(si_mag.shape[0])],
            color_continuous_scale='Viridis',
            aspect='auto',
            title="SI Output: RMS Activation per (t, ch)",
            labels=dict(x="Channel", y="Timestep", color="RMS"),
        )
        fig_si.update_layout(height=400)
        st.plotly_chart(fig_si, use_container_width=True)

    # Show full hidden state for selected channel
    st.markdown(f"**Full hidden state vector for Channel {channel_idx}:**")
    col_fh1, col_fh2 = st.columns(2)

    with col_fh1:
        te_ch = te_out[:, channel_idx, :]  # (T, d)
        fig_te_ch = px.imshow(
            te_ch,
            x=[f"d={i}" for i in range(te_ch.shape[1])],
            y=[f"t={i}" for i in range(te_ch.shape[0])],
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title=f"TE Hidden State (Ch {channel_idx})",
            labels=dict(x="Hidden dim", y="Timestep", color="Value"),
        )
        fig_te_ch.update_layout(height=350)
        st.plotly_chart(fig_te_ch, use_container_width=True)

    with col_fh2:
        si_ch = si_out[:, channel_idx, :]  # (T, d)
        fig_si_ch = px.imshow(
            si_ch,
            x=[f"d={i}" for i in range(si_ch.shape[1])],
            y=[f"t={i}" for i in range(si_ch.shape[0])],
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title=f"SI Hidden State (Ch {channel_idx})",
            labels=dict(x="Hidden dim", y="Timestep", color="Value"),
        )
        fig_si_ch.update_layout(height=350)
        st.plotly_chart(fig_si_ch, use_container_width=True)

except Exception as e:
    st.warning(f"Could not extract hidden states: {e}")
    import traceback
    st.code(traceback.format_exc())

# =====================================================================
# 5. Per-Channel Prediction Quality: R^2 vs MSE Scatter
# =====================================================================
st.subheader("Per-Channel Prediction Quality")
st.caption("Scatter plot of per-channel R-squared vs per-channel MSE across the full validation set.")

from utils.metrics import per_channel_mse, r2_score

ch_mse = per_channel_mse(P_all, Y_all)  # (C,)
r2_result = r2_score(P_all, Y_all)
ch_r2 = r2_result['per_channel']  # (C,)

fig_scatter = go.Figure(
    go.Scatter(
        x=ch_mse,
        y=ch_r2,
        mode='markers',
        text=[f"Ch {i}" for i in range(C)],
        marker=dict(
            size=7,
            color=ch_r2,
            colorscale='RdYlGn',
            colorbar=dict(title="R\u00b2"),
            showscale=True,
        ),
        hovertemplate="Ch %{text}<br>MSE: %{x:.5f}<br>R\u00b2: %{y:.4f}<extra></extra>",
    )
)
fig_scatter.update_layout(
    title="Per-Channel R\u00b2 vs MSE",
    xaxis_title="Channel MSE",
    yaxis_title="Channel R\u00b2",
    height=500,
)
# Add reference lines
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_scatter.add_hline(y=float(np.mean(ch_r2)), line_dash="dot", line_color="blue",
                       annotation_text=f"Mean R\u00b2={np.mean(ch_r2):.4f}")
st.plotly_chart(fig_scatter, use_container_width=True)

# Summary stats
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Mean Channel R\u00b2", f"{np.mean(ch_r2):.4f}")
col_s2.metric("Best Channel R\u00b2", f"{np.max(ch_r2):.4f} (Ch {np.argmax(ch_r2)})")
col_s3.metric("Worst Channel R\u00b2", f"{np.min(ch_r2):.4f} (Ch {np.argmin(ch_r2)})")
