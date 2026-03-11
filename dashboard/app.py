"""
Neural Forecasting Dashboard — Main Entry Point
================================================
Run with: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Neural Forecasting Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------
st.sidebar.title("Neural Forecasting")
st.sidebar.markdown(
    """
**Project**: Monkey Neural Activity Forecasting
**Competition**: CodaBench #9806
**Metric**: MSE on LMP predictions (lower is better)
**Monkeys**: Affi (239 ch) / Beignet (89 ch)
**Model**: AMAG (GRU + Graph NN)
"""
)
st.sidebar.divider()
st.sidebar.markdown("Navigate using the pages in the sidebar above.")

# ---------- Main Page ----------
st.title("Monkey Neural Forecasting Dashboard")

st.markdown(
    """
Welcome to the interactive dashboard for the **Monkey Neural Activity Forecasting** project.

### Task
Predict future Local Motor Potential (LMP) signals from rhesus macaque micro-ECoG arrays
during reaching tasks.

| Property | Value |
|----------|-------|
| Input | 10 timesteps x C channels x 9 features |
| Output | 10 timesteps x C channels (LMP only) |
| Resolution | 30 ms per timestep (600 ms total window) |
| Affi channels | 239 |
| Beignet channels | 89 |

### Pages

- **Training** -- Launch and monitor model training runs
- **Results** -- Evaluate checkpoints, visualize predictions vs ground truth
- **Diagnosis** -- Inspect learned adjacency matrices, hidden states, feature heatmaps

Use the sidebar to navigate between pages.
"""
)
