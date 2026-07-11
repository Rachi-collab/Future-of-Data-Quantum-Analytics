"""
Page 5 · Comparative Analysis
Classical vs Quantum — accuracy, speed, and scalability benchmarks.
"""

import sys
sys.path.append(".")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy, time

from src.data_utils import generate_dataset, AVAILABLE_DATASETS
from src.classical_ml import train_and_evaluate, CLASSIFIERS
from src.quantum_ml import VQClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Comparative Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#050d1a,#0a1628,#06111f); }
    .section { color:#67e8f9; font-size:1.2rem; font-weight:600;
               border-left:3px solid #7b2fff; padding-left:.7rem; margin:1.2rem 0 .6rem; }
    .vs-badge { display:inline-block; background:linear-gradient(90deg,#00d4ff33,#7b2fff33);
                border:1px solid #7b2fff55; border-radius:6px; padding:3px 12px;
                font-size:.85rem; color:#c4b5fd; font-family:monospace; }
    .win { color:#4ade80; font-weight:700; }
    .lose { color:#f87171; }
</style>""", unsafe_allow_html=True)

st.title("📊 Comparative Analysis")
st.caption("Side-by-side benchmarking: Classical ML vs Quantum VQC.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Benchmark Settings")
    dataset   = st.selectbox("Dataset",       AVAILABLE_DATASETS[:3])
    n_samples = st.slider("Samples",          80, 200, 120, 20)
    noise     = st.slider("Noise",            0.0, 0.4, 0.15, 0.05)
    test_pct  = st.slider("Test split %",     20, 40, 25)
    vqc_iter  = st.slider("VQC max iter",     20, 80, 40, 10)
    run_btn   = st.button("▶ Run Benchmark", type="primary", use_container_width=True)

dark = "#050d1a"
plt.rcParams.update({
    "figure.facecolor": dark, "axes.facecolor": "#0a1628",
    "text.color": "#c0d0e0", "axes.labelcolor": "#8899aa",
    "xtick.color": "#8899aa", "ytick.color": "#8899aa",
    "grid.color": "#1a2a3a", "axes.edgecolor": "#00d4ff33",
})

if run_btn:
    X, y, feat_names = generate_dataset(dataset, n_samples=n_samples, noise=noise)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42)

    results = {}

    # Classical models
    with st.spinner("Training classical models…"):
        for name, base_clf in CLASSIFIERS.items():
            clf = copy.deepcopy(base_clf)
            m = train_and_evaluate(clf, X_tr, y_tr, X_te, y_te)
            results[name] = {
                "accuracy":       m["accuracy"],
                "train_time_ms":  m["train_time_ms"],
                "infer_time_ms":  m["infer_time_ms"],
                "type":           "Classical",
            }

    # Quantum VQC
    with st.spinner("Training Quantum VQC (may take ~30s)…"):
        vqc = VQClassifier(n_params=2, max_iter=vqc_iter, random_state=42)
        vqc.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        y_pred_vqc = vqc.predict(X_te)
        infer_ms = (time.perf_counter() - t0) * 1000
        results["VQC (Quantum)"] = {
            "accuracy":       accuracy_score(y_te, y_pred_vqc),
            "train_time_ms":  vqc.train_time_ms_,
            "infer_time_ms":  infer_ms,
            "type":           "Quantum",
        }

    df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    df_res["accuracy"]      = df_res["accuracy"].astype(float)
    df_res["train_time_ms"] = df_res["train_time_ms"].astype(float)
    df_res["infer_time_ms"] = df_res["infer_time_ms"].astype(float)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section">📋 Benchmark Summary</div>', unsafe_allow_html=True)
    best_acc = df_res["accuracy"].max()
    display = df_res[["Model","type","accuracy","train_time_ms","infer_time_ms"]].copy()
    display.columns = ["Model","Type","Accuracy","Train (ms)","Infer (ms)"]
    display["Accuracy"] = display["Accuracy"].round(4)
    display["Train (ms)"] = display["Train (ms)"].round(1)
    display["Infer (ms)"] = display["Infer (ms)"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Plots ──────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    models = df_res["Model"].tolist()
    colors = ["#00d4ff" if t == "Classical" else "#7b2fff" for t in df_res["type"].tolist()]

    with col_a:
        st.markdown('<div class="section">🎯 Accuracy Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.barh(models, df_res["accuracy"].values, color=colors, edgecolor="#0d1f3c", height=0.5)
        for bar, v in zip(bars, df_res["accuracy"].values):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9, color="#c0d0e0")
        ax.axvline(0.5, color="#f87171", linestyle="--", linewidth=1, alpha=0.5, label="Chance")
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Test Accuracy")
        ax.set_title("Accuracy by Model", color="#67e8f9", fontsize=10)
        # Legend
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor="#00d4ff", label="Classical"),
                        Patch(facecolor="#7b2fff", label="Quantum")]
        ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown('<div class="section">⏱️ Training Time Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.barh(models, df_res["train_time_ms"].values, color=colors, edgecolor="#0d1f3c", height=0.5)
        for bar, v in zip(bars, df_res["train_time_ms"].values):
            ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                    f"{v:.0f} ms", va="center", fontsize=9, color="#c0d0e0")
        ax.set_xlabel("Training Time (ms)")
        ax.set_title("Training Time by Model", color="#67e8f9", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        legend_elems = [Patch(facecolor="#00d4ff", label="Classical"),
                        Patch(facecolor="#7b2fff", label="Quantum")]
        ax.legend(handles=legend_elems, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section">🕸️ Multi-Metric Radar</div>', unsafe_allow_html=True)

    # Normalise for radar: accuracy as-is, speed = 1/log(time+1) normalised
    max_t = df_res["train_time_ms"].max()
    df_res["speed_score"] = 1 - (np.log1p(df_res["train_time_ms"]) / np.log1p(max_t))
    df_res["infer_score"] = 1 - (np.log1p(df_res["infer_time_ms"]) / np.log1p(df_res["infer_time_ms"].max()))

    radar_metrics = ["accuracy", "speed_score", "infer_score"]
    radar_labels  = ["Accuracy", "Train Speed", "Infer Speed"]
    N = len(radar_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0a1628")
    fig.patch.set_facecolor(dark)
    ax.tick_params(colors="#8899aa")
    ax.spines["polar"].set_color("#00d4ff33")

    for _, row in df_res.iterrows():
        vals = [float(row[m]) for m in radar_metrics]
        vals += vals[:1]
        color = "#00d4ff" if row["type"] == "Classical" else "#7b2fff"
        ax.plot(angles, vals, color=color, linewidth=1.5, linestyle="solid")
        ax.fill(angles, vals, color=color, alpha=0.1)
        # Label at first point
        ax.annotate(row["Model"].split(" ")[0],
                    xy=(angles[0], vals[0]),
                    color=color, fontsize=7, ha="center")

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=7)
    ax.set_title("Model Comparison Radar", color="#67e8f9", fontsize=10, pad=15)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section">💡 Key Insights</div>', unsafe_allow_html=True)
    best_model = df_res.loc[df_res["accuracy"].idxmax(), "Model"]
    fastest    = df_res.loc[df_res["train_time_ms"].idxmin(), "Model"]
    vqc_acc    = results["VQC (Quantum)"]["accuracy"]
    best_cl_acc = max(v["accuracy"] for k,v in results.items() if v["type"]=="Classical")
    gap        = vqc_acc - best_cl_acc

    insights = [
        f"🏆 **Best accuracy overall:** {best_model} ({df_res.loc[df_res['accuracy'].idxmax(), 'accuracy']:.4f})",
        f"⚡ **Fastest to train:** {fastest} ({df_res.loc[df_res['train_time_ms'].idxmin(), 'train_time_ms']:.1f} ms)",
        f"⚛️ **VQC accuracy:** {vqc_acc:.4f}  |  Best classical: {best_cl_acc:.4f}  |  Gap: {gap:+.4f}",
        "📈 Classical models win on **speed** — VQC sim is CPU-bound; real quantum hardware changes this.",
        "🔬 On **small, noisy datasets** (~100 samples), quantum models can match classical performance.",
        "🚀 Quantum advantage expected at scale with fault-tolerant hardware and larger feature spaces.",
    ]
    for ins in insights:
        st.markdown(ins)

else:
    st.info("⬅️ Configure settings in the sidebar and click **▶ Run Benchmark** to compare models.")

    # Static explainer
    st.markdown('<div class="section">📚 What This Page Shows</div>', unsafe_allow_html=True)
    st.markdown("""
    This module runs a head-to-head benchmark between:

    - **Random Forest** — ensemble of decision trees
    - **SVM (RBF kernel)** — maximal-margin hyperplane
    - **Logistic Regression** — linear probabilistic classifier
    - **VQC (Quantum)** — Variational Quantum Classifier via Qiskit statevector simulation

    Metrics compared: **test accuracy**, **training time**, and **inference time**.

    The radar chart normalises all metrics to [0,1] for an at-a-glance multi-dimensional comparison.
    """)
