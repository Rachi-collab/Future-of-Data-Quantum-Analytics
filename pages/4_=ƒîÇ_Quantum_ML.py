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
from src.quantum_ml import VQClassifier, QSVClassifier, zz_feature_map
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Quantum ML Settings")
    model_type = st.radio("Quantum Model", ["VQC (Variational)", "QSVC (Kernel-based)"])
    dataset   = st.selectbox("Dataset",       AVAILABLE_DATASETS[:3])  # limit to 2-class easy sets
    n_samples = st.slider("Samples",          60, 200, 100, 20)
    noise     = st.slider("Noise",            0.0, 0.4, 0.15, 0.05)
    
    if model_type == "VQC (Variational)":
        max_iter  = st.slider("COBYLA max iter",  20,  120, 60,   10)
        run_btn   = st.button("▶ Train VQC", type="primary", use_container_width=True)
    else:
        run_btn   = st.button("▶ Train QSVC", type="primary", use_container_width=True)
        
    test_pct  = st.slider("Test split %",     20,  40,  25)

# Dynamic info note
if model_type == "VQC (Variational)":
    st.caption("Variational Quantum Classifier using Qiskit's ZZ Feature Map + RY ansatz, optimised with COBYLA.")
    st.markdown("""
    <div class="qnote">
    <b>ℹ️ How it works: Variational Quantum Classifier (VQC)</b><br>
    1. Data is encoded into a quantum state via the <b>ZZ Feature Map</b> (Pauli rotations + ZZ interactions).<br>
    2. A <b>variational ansatz</b> of RY gates + CNOT entanglement is applied.<br>
    3. The expectation value of <b>Z⊗I</b> is measured — positive → Class 1, negative → Class 0.<br>
    4. Parameters are optimised with <b>COBYLA</b> to minimise MSE loss.<br>
    Note: Statevector simulation is exact but slow — training uses a 40-sample sub-batch.
    </div>""", unsafe_allow_html=True)
else:
    st.caption("Quantum Support Vector Classifier using Qiskit's ZZ Feature Map for Quantum Kernel Estimation.")
    st.markdown("""
    <div class="qnote">
    <b>ℹ️ How it works: Quantum Support Vector Classifier (QSVC)</b><br>
    1. Data is encoded into a quantum state via the <b>ZZ Feature Map</b> (Pauli rotations + ZZ interactions).<br>
    2. A pairwise <b>Quantum Kernel Matrix</b> $K_{ij} = |\\langle\\phi(x_i)|\\phi(x_j)\\rangle|^2$ is computed via statevector inner products.<br>
    3. A classical <b>Support Vector Classifier (SVC)</b> with a precomputed kernel finds the optimal separating hyperplane in the Hilbert space.<br>
    Note: QSVC does not need variational parameter optimization, but calculating the $O(N^2)$ state overlaps is CPU-bound on classical simulators.
    </div>""", unsafe_allow_html=True)

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
    if model_type == "VQC (Variational)":
        with st.spinner("Training VQC via statevector simulation (this may take ~30s)…"):
            model = VQClassifier(n_params=2, max_iter=max_iter, random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)
            cm     = confusion_matrix(y_te, y_pred)
            train_time_ms = model.train_time_ms_
        title_prefix = "VQC"
    else:
        with st.spinner("Training QSVC via quantum kernel estimation (this may take ~5s)…"):
            model = QSVClassifier(random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)
            cm     = confusion_matrix(y_te, y_pred)
            train_time_ms = model.train_time_ms_
        title_prefix = "QSVC"

    st.markdown(f'<div class="section">📊 {title_prefix} Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Test Accuracy",   f"{acc:.3f}"),
        (c2, "Train Size",      len(X_tr)),
        (c3, "Test Size",       len(X_te)),
        (c4, "Train (ms)",      f"{train_time_ms:.0f}"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="mv">{val}</div>'
                     f'<div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Decision boundary
    with col_a:
        st.markdown(f'<div class="section">🗺️ {title_prefix} Decision Boundary</div>', unsafe_allow_html=True)
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 60),
                             np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 60))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.RdBu_r)
        ax.contour(xx, yy, Z, colors="#7b2fff", linewidths=1.2, alpha=0.5)
        for cls, color in enumerate(["#00d4ff","#7b2fff"]):
            mask = y == cls
            ax.scatter(X[mask,0], X[mask,1], c=color, s=14, alpha=0.7, label=f"Class {cls}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title(f"{title_prefix} — Decision Boundary", fontsize=10, color="#a78bfa")
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

    # Model specific details
    if title_prefix == "VQC":
        st.markdown('<div class="section">🎛️ Optimised Parameters</div>', unsafe_allow_html=True)
        param_df = pd.DataFrame({
            "Parameter": [f"θ_{i}" for i in range(len(model.params_))],
            "Optimised Value (rad)": model.params_.round(5),
            "Cos(θ)": np.cos(model.params_).round(5),
            "Sin(θ)": np.sin(model.params_).round(5),
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="section">📐 Support Vectors Information</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="qnote" style="color: #c0d0e0">
        • <b>Number of Support Vectors:</b> {len(model.svc_.support_)} (out of {len(X_tr)} training samples)<br>
        • <b>Indices of Support Vectors:</b> {model.svc_.support_.tolist()}<br>
        • <b>Dual Coefficients (Alpha):</b> {model.svc_.dual_coef_[0].round(4).tolist()}<br>
        • <b>Intercept (Bias):</b> {model.svc_.intercept_[0]:.4f}
        </div>
        """, unsafe_allow_html=True)

else:
    # Show feature map visualisation when not yet trained
    st.markdown('<div class="section">🗺️ ZZ Feature Map — Amplitude Landscape</div>', unsafe_allow_html=True)
    st.info(f"Configure settings in the sidebar and click **▶ Train {model_type.split(' ')[0]}** to run the classifier.")

    st.markdown("**Preview: |amplitude|² across x₀ for a fixed x₁ = π/4**")
    x1_fixed = np.pi / 4
    x0_vals  = np.linspace(0, np.pi, 40)
    probs_0  = []  # P(|00⟩)
    for x0 in x0_vals:
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
