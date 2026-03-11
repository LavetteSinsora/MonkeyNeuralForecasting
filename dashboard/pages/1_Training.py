"""
Training Page — Launch and monitor AMAG training runs.
"""

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import torch

from dashboard.dashboard_utils import get_available_monkeys, detect_device

st.title("Model Training")

# ---------- Sidebar Controls ----------
st.sidebar.header("Training Configuration")

arch = st.sidebar.selectbox("Architecture", ["AMAG"], index=0)
monkey_choice = st.sidebar.selectbox("Monkey", ["affi", "beignet", "both"], index=0)
lr = st.sidebar.number_input("Learning Rate", value=5e-4, format="%.1e", step=1e-4)
hidden_size = st.sidebar.number_input("Hidden Size", value=64, min_value=16, max_value=512, step=16)
n_epochs = st.sidebar.number_input("Epochs", value=200, min_value=1, max_value=2000, step=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
weight_decay = st.sidebar.number_input("Weight Decay", value=1e-5, format="%.1e", step=1e-5)
patience = st.sidebar.number_input("Early Stopping Patience", value=100, min_value=5, max_value=500, step=10)

device_name = detect_device()
st.sidebar.info(f"Device: **{device_name}**")

# ---------- Main Area ----------
st.markdown(
    f"""
**Selected configuration:**
- Architecture: {arch}
- Monkey: {monkey_choice}
- LR: {lr} | Hidden: {hidden_size} | Epochs: {n_epochs} | Batch: {batch_size}
- Weight decay: {weight_decay} | Patience: {patience}
"""
)

start_btn = st.button("Start Training", type="primary")

if start_btn:
    monkeys = ["affi", "beignet"] if monkey_choice == "both" else [monkey_choice]

    for monkey in monkeys:
        st.subheader(f"Training on {monkey.upper()}")
        status_text = st.empty()
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

        status_text.text(f"Loading data for {monkey}...")

        try:
            from utils.data import get_dataloaders

            train_loader, val_loader, stats = get_dataloaders(
                monkey, batch_size=batch_size, val_fraction=0.15, seed=42
            )
            status_text.text(
                f"Data loaded. Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}"
            )

            # Build model
            try:
                from replications.amag.model import build_model
            except ImportError:
                from replication.amag.model import build_model

            model_cfg = {
                'hidden_size': hidden_size,
                'use_adaptor': True,
                'compute_init_corr': True,
            }
            model = build_model(monkey, model_cfg)
            info = model.get_model_info()
            st.info(f"Model: {info['name']} | {info['n_params_M']}M parameters")

            # Setup training components
            device = torch.device(device_name)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
            criterion = torch.nn.MSELoss()

            # Checkpoint directory
            ckpt_dir = os.path.join(_root, 'dashboard', 'training_checkpoints', monkey)
            os.makedirs(ckpt_dir, exist_ok=True)

            best_val_mse = float('inf')
            patience_counter = 0
            history = []

            # Training loop with live UI updates
            for epoch in range(1, n_epochs + 1):
                # Train
                model.train()
                train_loss = 0.0
                n_batches = 0
                for X, Y in train_loader:
                    X, Y = X.to(device), Y.to(device)
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = criterion(pred, Y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches += 1
                train_mse = train_loss / max(n_batches, 1)

                # Validate
                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for X, Y in val_loader:
                        X, Y = X.to(device), Y.to(device)
                        pred = model(X)
                        val_loss += criterion(pred, Y).item()
                        n_val += 1
                val_mse = val_loss / max(n_val, 1)

                scheduler.step()

                # Track best
                improved = val_mse < best_val_mse
                if improved:
                    best_val_mse = val_mse
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'best_val_mse': best_val_mse,
                        'norm_stats': stats,
                        'config': {
                            'hidden_size': hidden_size,
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'batch_size': batch_size,
                        },
                    }, os.path.join(ckpt_dir, 'best.pth'))
                else:
                    patience_counter += 1

                history.append({
                    'epoch': epoch,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'best_val_mse': best_val_mse,
                })

                # Update UI every 1 epoch (or less frequently for long runs)
                update_interval = max(1, n_epochs // 200)
                if epoch % update_interval == 0 or epoch == 1 or epoch == n_epochs:
                    progress_bar.progress(epoch / n_epochs)
                    mark = " *" if improved else ""
                    status_text.text(
                        f"Epoch {epoch}/{n_epochs}{mark} | "
                        f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | "
                        f"Best: {best_val_mse:.6f}"
                    )
                    df = pd.DataFrame(history)
                    chart_placeholder.line_chart(
                        df.set_index('epoch')[['train_mse', 'val_mse']],
                        use_container_width=True,
                    )

                # Early stopping
                if patience_counter >= patience:
                    status_text.text(
                        f"Early stopping at epoch {epoch}. Best val MSE: {best_val_mse:.6f}"
                    )
                    break

            progress_bar.progress(1.0)

            # Final results
            st.success(
                f"Training complete for {monkey.upper()}! "
                f"Best Val MSE: {best_val_mse:.6f} | "
                f"Epochs trained: {len(history)}"
            )
            metrics_placeholder.metric("Best Validation MSE", f"{best_val_mse:.6f}")
            st.info(f"Checkpoint saved to: {os.path.join(ckpt_dir, 'best.pth')}")

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
