"""
Page 1 · Data Exploration
Preprocessing, statistical profiling, correlation analysis, outlier detection.
"""

import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_utils import generate_dataset, make_dataframe, AVAILABLE_DATASETS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Exploration", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#050d1a,#0a1628,#06111f); }
    .section { color:#00d4ff; font-size:1.2rem; font-weight:600;
               border-left:3px solid #7b2fff; padding-left:.7rem; margin:1.2rem 0 .6rem; }
    .stat-card { background:#0d1f3c; border:1px solid #00d4ff22; border-radius:10px;
                 padding:1rem; text-align:center; }
    .stat-val  { font-size:1.5rem; color:#00d4ff; font-weight:700; font-family:monospace; }
    .stat-lbl  { color:#64748b; font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
</style>""", unsafe_allow_html=True)

st.title("🔬 Data Exploration")
st.caption("Preprocess and profile datasets before feeding them into classical or quantum models.")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Dataset Settings")
    dataset  = st.selectbox("Dataset",   AVAILABLE_DATASETS)
    n_samples = st.slider("Samples",     100, 1000, 300, step=50)
    noise     = st.slider("Noise level", 0.0, 0.5,  0.2, step=0.05)
    test_size = st.slider("Test split %", 10, 40, 20) / 100
    seed      = st.number_input("Random seed", 0, 999, 42)

# ── Generate data ─────────────────────────────────────────────────────────────
X, y, feat_names = generate_dataset(dataset, n_samples=n_samples, noise=noise, random_state=seed)
df = make_dataframe(X, y, feat_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section">📊 Summary Statistics</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
for col, lbl, val in [
    (c1, "Total Samples",   n_samples),
    (c2, "Features",        2),
    (c3, "Classes",         2),
    (c4, "Train Samples",   len(X_tr)),
    (c5, "Test Samples",    len(X_te)),
]:
    col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div>'
                 f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Descriptive stats table ───────────────────────────────────────────────────
st.markdown('<div class="section">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
st.dataframe(df.describe().round(4), use_container_width=True)

# ── Plots ─────────────────────────────────────────────────────────────────────
dark = "#050d1a"
plt.rcParams.update({
    "figure.facecolor": dark, "axes.facecolor": "#0a1628",
    "text.color": "#c0d0e0", "axes.labelcolor": "#8899aa",
    "xtick.color": "#8899aa", "ytick.color": "#8899aa",
    "grid.color": "#1a2a3a", "axes.edgecolor": "#00d4ff33",
})

col_a, col_b = st.columns(2)

# Scatter plot
with col_a:
    st.markdown('<div class="section">🎯 Class Distribution (Scatter)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    palette = {0: "#00d4ff", 1: "#7b2fff"}
    for cls, color in palette.items():
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=18, label=f"Class {cls}")
    ax.set_xlabel(feat_names[0]); ax.set_ylabel(feat_names[1])
    ax.legend(framealpha=0.2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# Histogram of features
with col_b:
    st.markdown('<div class="section">📈 Feature Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(5, 4))
    for i, (fname, ax) in enumerate(zip(feat_names, axes)):
        ax.hist(X[:, i][y==0], bins=30, color="#00d4ff", alpha=0.6, label="Class 0")
        ax.hist(X[:, i][y==1], bins=30, color="#7b2fff", alpha=0.6, label="Class 1")
        ax.set_title(fname, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

col_c, col_d = st.columns(2)

# Correlation heatmap
with col_c:
    st.markdown('<div class="section">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, linecolor="#0d1f3c",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlations", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# Box plot for outlier detection
with col_d:
    st.markdown('<div class="section">📦 Outlier Detection (Box Plots)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    data_to_plot = [df[fn].values for fn in feat_names]
    bp = ax.boxplot(data_to_plot, labels=feat_names, patch_artist=True,
                    medianprops=dict(color="#00d4ff", linewidth=2))
    colors = ["#00d4ff44", "#7b2fff44"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title("Feature Spread & Outliers", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Class balance ──────────────────────────────────────────────────────────────
st.markdown('<div class="section">⚖️ Class Balance</div>', unsafe_allow_html=True)
balance = pd.Series(y).value_counts().rename_axis("Class").reset_index(name="Count")
balance["Percentage"] = (balance["Count"] / len(y) * 100).round(1)
st.dataframe(balance, use_container_width=True, hide_index=True)

# ── Raw data preview ──────────────────────────────────────────────────────────
with st.expander("🗂️ Raw Data Preview (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

st.success(f"✅ Dataset '{dataset}' ready · {len(X_tr)} train / {len(X_te)} test samples")
