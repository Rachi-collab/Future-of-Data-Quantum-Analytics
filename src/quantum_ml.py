"""
Quantum machine-learning helpers using Qiskit's statevector simulator.

Implements a lightweight Variational Quantum Classifier (VQC) via a
parameterised ansatz evaluated with a simple kernel or expectation value.
"""

import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Quantum feature map helpers
# ---------------------------------------------------------------------------

def zz_feature_map(x: np.ndarray, reps: int = 1) -> QuantumCircuit:
    """
    Build a 2-qubit ZZFeatureMap circuit for a 2-D input x.
    """
    assert len(x) == 2, "Only 2-feature inputs supported."
    n = 2
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(2.0 * x[i], i)
        qc.cx(0, 1)
        qc.p(2.0 * (np.pi - x[0]) * (np.pi - x[1]), 1)
        qc.cx(0, 1)
    return qc


def rx_ansatz(params: np.ndarray, n_qubits: int = 2, reps: int = 1) -> QuantumCircuit:
    """
    Simple RY-CNOT variational ansatz.
    """
    n_params = n_qubits * reps
    assert len(params) == n_params
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc


# ---------------------------------------------------------------------------
# Quantum Kernel (for QKE)
# ---------------------------------------------------------------------------

def quantum_kernel_matrix(X_train: np.ndarray, X_test: np.ndarray = None) -> np.ndarray:
    """
    Compute the quantum kernel matrix K[i,j] = |<φ(x_i)|φ(x_j)>|²
    using statevector inner products.  O(n²) — fine for small datasets.
    """
    from qiskit.quantum_info import Statevector

    if X_test is None:
        X_test = X_train

    def statevec(x):
        qc = zz_feature_map(x, reps=2)
        return Statevector(qc)

    sv_train = [statevec(x) for x in X_train]
    sv_test = [statevec(x) for x in X_test]

    K = np.zeros((len(sv_test), len(sv_train)))
    for i, st in enumerate(sv_test):
        for j, ss in enumerate(sv_train):
            inner = np.abs(st.data.conj() @ ss.data) ** 2
            K[i, j] = inner
    return K


# ---------------------------------------------------------------------------
# Simple VQC using expectation value of Z⊗I
# ---------------------------------------------------------------------------

def _expectation_z0(params: np.ndarray, x: np.ndarray) -> float:
    """
    Compute <Z⊗I> for feature-map + ansatz circuit.
    Uses Qiskit's Statevector for exact simulation.
    """
    from qiskit.quantum_info import Statevector, SparsePauliOp

    fm = zz_feature_map(x, reps=1)
    ans = rx_ansatz(params, n_qubits=2, reps=1)
    qc = fm.compose(ans)

    sv = Statevector(qc)
    op = SparsePauliOp.from_list([("IZ", 1.0)])
    exp_val = sv.expectation_value(op).real
    return float(exp_val)


class VQClassifier(BaseEstimator, ClassifierMixin):
    """
    Minimal Variational Quantum Classifier.

    Uses gradient-free COBYLA optimisation of a 2-qubit RY ansatz
    evaluated via statevector simulation. Only suitable for 2-feature,
    binary classification problems.
    """

    def __init__(self, n_params: int = 2, max_iter: int = 60, random_state: int = 42):
        self.n_params = n_params
        self.max_iter = max_iter
        self.random_state = random_state
        self.params_ = None
        self.scaler_ = MinMaxScaler(feature_range=(0, np.pi))
        self.train_time_ms_ = 0.0

    def _cost(self, params, X_sub, y_sub):
        preds = np.array([_expectation_z0(params, x) for x in X_sub])
        # Map [-1, 1] → {0, 1} logistic loss
        loss = np.mean((preds - (2 * y_sub - 1)) ** 2)
        return loss

    def fit(self, X, y):
        from scipy.optimize import minimize

        rng = np.random.RandomState(self.random_state)
        X_sc = self.scaler_.fit_transform(X)

        # Sub-sample for speed (statevector sim is O(2^n) per shot)
        n = min(len(X_sc), 40)
        idx = rng.choice(len(X_sc), n, replace=False)
        X_sub, y_sub = X_sc[idx], y[idx]

        init_params = rng.uniform(0, 2 * np.pi, self.n_params)

        t0 = time.perf_counter()
        result = minimize(
            self._cost, init_params, args=(X_sub, y_sub),
            method="COBYLA",
            options={"maxiter": self.max_iter, "rhobeg": 0.5},
        )
        self.train_time_ms_ = (time.perf_counter() - t0) * 1000
        self.params_ = result.x
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_sc = self.scaler_.transform(X)
        exp_vals = np.array([_expectation_z0(self.params_, x) for x in X_sc])
        return (exp_vals >= 0.0).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# Quantum circuit demos (used in the Circuits page)
# ---------------------------------------------------------------------------

def build_bell_circuit() -> QuantumCircuit:
    """Returns the canonical Bell state circuit |Φ+⟩."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_ghz_circuit(n: int = 3) -> QuantumCircuit:
    """Returns an n-qubit GHZ state circuit."""
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def build_grover_circuit() -> QuantumCircuit:
    """2-qubit Grover's algorithm targeting |11⟩."""
    qc = QuantumCircuit(2, 2)
    # Initialise uniform superposition
    qc.h([0, 1])
    # Oracle: mark |11⟩ with phase flip
    qc.cz(0, 1)
    # Diffusion operator
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def build_qft_circuit(n: int = 3) -> QuantumCircuit:
    """Quantum Fourier Transform on n qubits."""
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2 ** (k - j), j, k)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def simulate_counts(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Simulate a circuit using Qiskit's StatevectorSampler and return counts.
    """
    from qiskit.primitives import StatevectorSampler

    # StatevectorSampler needs a circuit with measurements
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()
    # Extract bit-string counts from the first PUB result
    pub_result = result[0]
    data = pub_result.data
    # Get the first ClassicalRegister's BitArray
    creg_name = list(data.__dict__.keys())[0]
    bit_array = getattr(data, creg_name)
    counts_raw = bit_array.get_counts()
    # Normalise keys to zero-padded strings
    n_bits = qc.num_clbits
    counts = {k.replace(" ", "").zfill(n_bits): v for k, v in counts_raw.items()}
    return counts
