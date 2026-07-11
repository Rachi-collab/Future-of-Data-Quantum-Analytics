"""
Page 4 · Quantum Machine Learning
Variational Quantum Classifier (VQC) with Qiskit statevector simulation.
"""

import sys
sys.path.append(".")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

from src.data_utils import generate_dataset, AVAILABLE_DATASETS
from src.quantum_ml import VQClassifier, zz_feature_map
from qiskit.quantum_info import Statevector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quantum ML", page_icon="🌀", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#050d1a,#0a1628,#06111f); }
    .section { color:#a78bfa; font-size:1.2rem; font-weight:600;
               border-left:3px solid #00d4ff; padding-left:.7rem; margin:1.2rem 0 .6rem; }
    .metric-box { background:#0d1f3c; border:1px solid #7b2fff33; border-radius:10px;
                  padding:.9rem; text-align:center; }
    .mv  { font-size:1.6rem; color:#a78bfa; font-weight:700; font-family:monospace; }
    .ml  { color:#64748b; font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
    .qnote { background:#0a0f1e; border:1px solid #7b2fff44; border-radius:8px;
             padding:.8rem; color:#94a3b8; font-size:.85rem; }
</style>""", unsafe_allow_html=True)

st.title("🌀 Quantum Machine Learning")
st.caption("Variational Quantum Classifier using Qiskit's ZZ Feature Map + RY ansatz, optimised with COBYLA.")

st.markdown("""
<div class="qnote">
<b>ℹ️ How it works</b><br>
1. Data is encoded into a quantum state via the <b>ZZ Feature Map</b> (Pauli rotations + ZZ interactions).<br>
2. A <b>variational ansatz</b> of RY gates + CNOT entanglement is applied.<br>
3. The expectation value of <b>Z⊗I</b> is measured — positive → Class 1, negative → Class 0.<br>
4. Parameters are optimised with <b>COBYLA</b> to minimise MSE loss.<br>
Note: Statevector simulation is exact but slow — training uses a 40-sample sub-batch.
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ VQC Settings")
    dataset   = st.selectbox("Dataset",       AVAILABLE_DATASETS[:3])  # limit to 2-class easy sets
    n_samples = st.slider("Samples",          60, 200, 100, 20)
    noise     = st.slider("Noise",            0.0, 0.4, 0.15, 0.05)
    max_iter  = st.slider("COBYLA max iter",  20,  120, 60,   10)
    test_pct  = st.slider("Test split %",     20,  40,  25)
    run_btn   = st.button("▶ Train VQC", type="primary", use_container_width=True)

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, feat_names = generate_dataset(dataset, n_samples=n_samples, noise=noise)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42)

dark = "#050d1a"
plt.rcParams.update({
    "figure.facecolor": dark, "axes.facecolor": "#0a1628",
    "text.color": "#c0d0e0", "axes.labelcolor": "#8899aa",
    "xtick.color": "#8899aa", "ytick.color": "#8899aa",
    "grid.color": "#1a2a3a", "axes.edgecolor": "#7b2fff33",
})

if run_btn:
    with st.spinner("Training VQC via statevector simulation (this may take ~30s)…"):
        vqc = VQClassifier(n_params=2, max_iter=max_iter, random_state=42)
        vqc.fit(X_tr, y_tr)
        y_pred = vqc.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        cm     = confusion_matrix(y_te, y_pred)

    st.markdown('<div class="section">📊 VQC Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Test Accuracy",   f"{acc:.3f}"),
        (c2, "Train Size",      len(X_tr)),
        (c3, "Test Size",       len(X_te)),
        (c4, "Train (ms)",      f"{vqc.train_time_ms_:.0f}"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="mv">{val}</div>'
                     f'<div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Decision boundary
    with col_a:
        st.markdown('<div class="section">🗺️ VQC Decision Boundary</div>', unsafe_allow_html=True)
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 60),
                             np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 60))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = vqc.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.RdBu_r)
        ax.contour(xx, yy, Z, colors="#7b2fff", linewidths=1.2, alpha=0.5)
        for cls, color in enumerate(["#00d4ff","#7b2fff"]):
            mask = y == cls
            ax.scatter(X[mask,0], X[mask,1], c=color, s=14, alpha=0.7, label=f"Class {cls}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title("VQC — Decision Boundary", fontsize=10, color="#a78bfa")
        ax.set_xlabel(feat_names[0]); ax.set_ylabel(feat_names[1])
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Confusion matrix
    with col_b:
        st.markdown('<div class="section">🔍 Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                    xticklabels=["Pred 0","Pred 1"],
                    yticklabels=["True 0","True 1"],
                    linewidths=1, linecolor="#0d1f3c",
                    cbar_kws={"shrink":.8})
        ax.set_title("Confusion Matrix", fontsize=10, color="#a78bfa")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Optimal parameters
    st.markdown('<div class="section">🎛️ Optimised Parameters</div>', unsafe_allow_html=True)
    param_df = pd.DataFrame({
        "Parameter": [f"θ_{i}" for i in range(len(vqc.params_))],
        "Optimised Value (rad)": vqc.params_.round(5),
        "Cos(θ)": np.cos(vqc.params_).round(5),
        "Sin(θ)": np.sin(vqc.params_).round(5),
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)

else:
    # Show feature map visualisation when not yet trained
    st.markdown('<div class="section">🗺️ ZZ Feature Map — Amplitude Landscape</div>', unsafe_allow_html=True)
    st.info("Configure settings in the sidebar and click **▶ Train VQC** to run the classifier.")

    st.markdown("**Preview: |amplitude|² across x₀ for a fixed x₁ = π/4**")
    x1_fixed = np.pi / 4
    x0_vals  = np.linspace(0, np.pi, 40)
    probs_0  = []  # P(|00⟩)
    for x0 in x0_vals:
        from sklearn.preprocessing import MinMaxScaler
        qc = zz_feature_map(np.array([x0, x1_fixed]), reps=2)
        sv = Statevector(qc)
        probs_0.append(np.abs(sv.data[0])**2)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x0_vals, probs_0, color="#7b2fff", linewidth=2)
    ax.fill_between(x0_vals, probs_0, alpha=0.2, color="#7b2fff")
    ax.set_xlabel("x₀ (rad)"); ax.set_ylabel("|⟨00|φ(x)⟩|²")
    ax.set_title("ZZ Feature Map — P(|00⟩) vs x₀  (x₁ = π/4 fixed)", color="#a78bfa", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
