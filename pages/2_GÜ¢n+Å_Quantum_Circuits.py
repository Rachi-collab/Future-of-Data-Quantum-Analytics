"""
Page 2 · Quantum Circuits
Interactive Qiskit circuit builder, simulation, and visualisation.
"""

import sys
sys.path.append(".")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from src.quantum_ml import (
    build_bell_circuit, build_ghz_circuit,
    build_grover_circuit, build_qft_circuit,
    simulate_counts, zz_feature_map,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quantum Circuits", page_icon="⚛️", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#050d1a,#0a1628,#06111f); }
    .section { color:#00d4ff; font-size:1.2rem; font-weight:600;
               border-left:3px solid #7b2fff; padding-left:.7rem; margin:1.2rem 0 .6rem; }
    code { background:#0d1f3c !important; color:#a78bfa !important; }
    .info-card { background:#0d1f3c; border:1px solid #7b2fff33; border-radius:10px; padding:1rem; }
</style>""", unsafe_allow_html=True)

st.title("⚛️ Quantum Circuits")
st.caption("Explore fundamental quantum circuits — simulate them on a statevector backend and inspect the measurement outcomes.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Circuit Controls")
    circuit_type = st.selectbox("Circuit", [
        "Bell State (|Φ+⟩)",
        "GHZ State",
        "Grover's Search (2-qubit)",
        "Quantum Fourier Transform",
        "ZZ Feature Map (ML)",
    ])
    shots = st.slider("Shots (samples)", 256, 4096, 1024, step=256)
    if circuit_type == "GHZ State":
        n_qubits = st.slider("Qubits", 2, 5, 3)
    if circuit_type == "Quantum Fourier Transform":
        n_qubits_qft = st.slider("Qubits", 2, 4, 3)
    if circuit_type == "ZZ Feature Map (ML)":
        x0 = st.slider("x₀", 0.0, np.pi, np.pi / 3, step=0.05)
        x1 = st.slider("x₁", 0.0, np.pi, np.pi / 4, step=0.05)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Build circuit ─────────────────────────────────────────────────────────────
if circuit_type == "Bell State (|Φ+⟩)":
    qc = build_bell_circuit()
    description = ("Creates the maximally entangled Bell state **|Φ+⟩ = (|00⟩ + |11⟩)/√2**. "
                   "A Hadamard gate puts qubit-0 in superposition; a CNOT entangles qubit-1. "
                   "Measuring always gives correlated bits: either **00** or **11**.")
elif circuit_type == "GHZ State":
    qc = build_ghz_circuit(n_qubits)
    description = (f"Generalised Bell state across **{n_qubits} qubits**: "
                   f"**(|0…0⟩ + |1…1⟩)/√2**. "
                   "Used in quantum error correction and multi-party entanglement protocols.")
elif circuit_type == "Grover's Search (2-qubit)":
    qc = build_grover_circuit()
    description = ("Grover's algorithm amplifies the amplitude of the **marked state |11⟩**. "
                   "After one iteration, measuring gives **|11⟩** with high probability — "
                   "a quadratic speedup over classical unstructured search.")
elif circuit_type == "Quantum Fourier Transform":
    qc = build_qft_circuit(n_qubits_qft)
    qc.measure_all()
    description = (f"**{n_qubits_qft}-qubit QFT** — the quantum analogue of the DFT. "
                   "Core subroutine in Shor's algorithm and quantum phase estimation. "
                   "Measurement outcomes show a near-uniform distribution over all basis states.")
else:  # ZZ Feature Map
    qc_no_meas = zz_feature_map(np.array([x0, x1]), reps=2)
    qc = qc_no_meas.copy()
    qc.measure_all()
    description = (f"**ZZ Feature Map** encodes data point **(x₀={x0:.2f}, x₁={x1:.2f})** "
                   "into a 2-qubit quantum state via Pauli rotations and entangling ZZ interactions. "
                   "Used as the first layer of a Quantum Kernel or VQC.")

# ── Draw circuit ──────────────────────────────────────────────────────────────
st.markdown('<div class="section">🔌 Circuit Diagram</div>', unsafe_allow_html=True)

dark = "#050d1a"
plt.rcParams.update({
    "figure.facecolor": dark, "axes.facecolor": "#0a1628",
    "text.color": "#c0d0e0", "axes.labelcolor": "#8899aa",
    "xtick.color": "#8899aa", "ytick.color": "#8899aa",
    "axes.edgecolor": "#00d4ff33",
})

fig = qc.draw("mpl", style={
    "backgroundcolor": "#0a1628",
    "textcolor": "#c0d0e0",
    "gatefacecolor": "#1a2a4a",
    "gatetextcolor": "#00d4ff",
    "subtextcolor": "#7b2fff",
    "linecolor": "#00d4ff88",
    "creglinecolor": "#7b2fff88",
    "measurecolor": "#7b2fff",
    "latexdrawerstyle": False,
})
st.pyplot(fig); plt.close(fig)

st.markdown(f'<div class="info-card"><p style="color:#c0d0e0">{description}</p></div>',
            unsafe_allow_html=True)

# ── Simulate ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section">📡 Simulation Results</div>', unsafe_allow_html=True)

if run_btn or True:  # always show initial results
    with st.spinner("Running statevector simulation…"):
        try:
            counts = simulate_counts(qc, shots=shots)
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.stop()

    # Sort by bitstring
    sorted_counts = dict(sorted(counts.items()))
    labels = list(sorted_counts.keys())
    values = list(sorted_counts.values())
    probabilities = [v / shots for v in values]

    # Bar chart
    fig2, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), 4))
    bars = ax.bar(labels, probabilities, color="#00d4ff", alpha=0.8, edgecolor="#7b2fff", linewidth=1.2)
    for bar, p in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.3f}", ha="center", va="bottom", fontsize=9, color="#c0d0e0")
    ax.set_xlabel("Basis State", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(f"Measurement Probabilities  (shots = {shots})", fontsize=11, color="#00d4ff")
    ax.set_ylim(0, max(probabilities) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    # Counts table
    col1, col2 = st.columns(2)
    with col1:
        import pandas as pd
        df_counts = pd.DataFrame({
            "State": labels,
            "Counts": values,
            "Probability": [f"{p:.4f}" for p in probabilities],
        })
        st.dataframe(df_counts, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section">📐 Circuit Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <table style="width:100%; color:#c0d0e0; font-size:.9rem">
                <tr><td style="color:#8899aa">Qubits</td><td><b>{qc.num_qubits}</b></td></tr>
                <tr><td style="color:#8899aa">Classical bits</td><td><b>{qc.num_clbits}</b></td></tr>
                <tr><td style="color:#8899aa">Circuit depth</td><td><b>{qc.depth()}</b></td></tr>
                <tr><td style="color:#8899aa">Gate count</td><td><b>{sum(qc.count_ops().values())}</b></td></tr>
                <tr><td style="color:#8899aa">Shots</td><td><b>{shots}</b></td></tr>
                <tr><td style="color:#8899aa">Distinct states observed</td><td><b>{len(counts)}</b></td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # Statevector
    with st.expander("🔭 View Statevector (exact amplitudes)"):
        from qiskit.quantum_info import Statevector
        qc_no_m = qc.remove_final_measurements(inplace=False)
        sv = Statevector(qc_no_m)
        amps = sv.data
        n_states = len(amps)
        basis = [format(i, f"0{qc.num_qubits}b") for i in range(n_states)]
        sv_df = pd.DataFrame({
            "Basis |ψ⟩": basis,
            "Amplitude (Re)": amps.real.round(6),
            "Amplitude (Im)": amps.imag.round(6),
            "|Amplitude|²": (np.abs(amps)**2).round(6),
        })
        st.dataframe(sv_df, use_container_width=True, hide_index=True)
