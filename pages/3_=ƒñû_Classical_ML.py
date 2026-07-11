"""
Page 3 · Classical Machine Learning
Train RF, SVM, and Logistic Regression with detailed evaluation.
"""

import sys
sys.path.append(".")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.data_utils import generate_dataset, AVAILABLE_DATASETS
from src.classical_ml import train_and_evaluate, cross_validate, CLASSIFIERS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Classical ML", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#050d1a,#0a1628,#06111f); }
    .section { color:#00d4ff; font-size:1.2rem; font-weight:600;
               border-left:3px solid #7b2fff; padding-left:.7rem; margin:1.2rem 0 .6rem; }
    .metric-box { background:#0d1f3c; border:1px solid #00d4ff22; border-radius:10px;
                  padding:.9rem; text-align:center; }
    .mv  { font-size:1.6rem; color:#00d4ff; font-weight:700; font-family:monospace; }
    .ml  { color:#64748b; font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
</style>""", unsafe_allow_html=True)

st.title("🤖 Classical Machine Learning")
st.caption("Train, evaluate, and compare classical classifiers on the selected dataset.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    dataset   = st.selectbox("Dataset",     AVAILABLE_DATASETS)
    n_samples = st.slider("Samples",        100, 1000, 300, 50)
    noise     = st.slider("Noise",          0.0, 0.5,  0.2, 0.05)
    test_pct  = st.slider("Test split %",   10,  40,   20)
    clf_name  = st.selectbox("Classifier",  list(CLASSIFIERS.keys()))
    cv_folds  = st.slider("CV folds",       3,   10,   5)
    run_btn   = st.button("▶ Train Model", type="primary", use_container_width=True)

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, feat_names = generate_dataset(dataset, n_samples=n_samples, noise=noise)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42)

# Clone classifier to avoid state leakage
import copy
clf = copy.deepcopy(CLASSIFIERS[clf_name])

dark = "#050d1a"
plt.rcParams.update({
    "figure.facecolor": dark, "axes.facecolor": "#0a1628",
    "text.color": "#c0d0e0", "axes.labelcolor": "#8899aa",
    "xtick.color": "#8899aa", "ytick.color": "#8899aa",
    "grid.color": "#1a2a3a", "axes.edgecolor": "#00d4ff33",
})

# ── Training ──────────────────────────────────────────────────────────────────
if run_btn or True:
    with st.spinner(f"Training {clf_name}…"):
        metrics = train_and_evaluate(clf, X_tr, y_tr, X_te, y_te)
        cv_mean, cv_std = cross_validate(copy.deepcopy(CLASSIFIERS[clf_name]), X, y, cv=cv_folds)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section">📊 Performance Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, lbl, val in [
        (c1, "Accuracy",  f"{metrics['accuracy']:.3f}"),
        (c2, "Precision", f"{metrics['precision']:.3f}"),
        (c3, "Recall",    f"{metrics['recall']:.3f}"),
        (c4, "F1 Score",  f"{metrics['f1']:.3f}"),
        (c5, f"CV ({cv_folds}-fold)", f"{cv_mean:.3f}±{cv_std:.3f}"),
        (c6, "Train (ms)", f"{metrics['train_time_ms']:.1f}"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="mv">{val}</div>'
                     f'<div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Decision boundary
    with col_a:
        st.markdown('<div class="section">🗺️ Decision Boundary</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        disp = DecisionBoundaryDisplay.from_estimator(
            clf, X, response_method="predict",
            cmap=plt.cm.RdBu_r, alpha=0.35, ax=ax,
            xlabel=feat_names[0], ylabel=feat_names[1],
        )
        scatter_colors = ["#00d4ff", "#7b2fff"]
        for cls, color in enumerate(scatter_colors):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=14, alpha=0.7, label=f"Class {cls}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title(f"{clf_name} — Decision Boundary", fontsize=10, color="#00d4ff")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Confusion matrix
    with col_b:
        st.markdown('<div class="section">🔍 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0","Pred 1"],
                    yticklabels=["True 0","True 1"],
                    linewidths=1, linecolor="#0d1f3c",
                    cbar_kws={"shrink":0.8})
        ax.set_title("Confusion Matrix", fontsize=10, color="#00d4ff")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Feature importance (RF only) ──────────────────────────────────────────
    if clf_name == "Random Forest":
        st.markdown('<div class="section">🌳 Feature Importances</div>', unsafe_allow_html=True)
        importances = clf.feature_importances_
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(feat_names, importances, color=["#00d4ff","#7b2fff"], edgecolor="#0d1f3c")
        ax.set_xlabel("Importance"); ax.set_title("Feature Importances", color="#00d4ff", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── All classifiers comparison ─────────────────────────────────────────────
    st.markdown('<div class="section">📋 All Classifiers Comparison</div>', unsafe_allow_html=True)
    rows = []
    import copy as _copy
    for name, base_clf in CLASSIFIERS.items():
        m = train_and_evaluate(_copy.deepcopy(base_clf), X_tr, y_tr, X_te, y_te)
        cv_m, cv_s = cross_validate(_copy.deepcopy(base_clf), X, y, cv=5)
        rows.append({
            "Classifier":    name,
            "Accuracy":      round(m["accuracy"], 4),
            "Precision":     round(m["precision"], 4),
            "Recall":        round(m["recall"], 4),
            "F1":            round(m["f1"], 4),
            "CV Accuracy":   f"{cv_m:.3f}±{cv_s:.3f}",
            "Train (ms)":    round(m["train_time_ms"], 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
