from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectModule:
    icon: str
    name: str
    description: str
    category: str


PROJECT_MODULES = [
    ProjectModule(
        icon="🔬",
        name="Data Exploration",
        description="Preprocess and explore datasets with statistical profiling, correlation analysis, and outlier detection.",
        category="classical",
    ),
    ProjectModule(
        icon="⚛️",
        name="Quantum Circuits",
        description="Build and simulate quantum gates, circuits, and entanglement using Qiskit's statevector simulator.",
        category="quantum",
    ),
    ProjectModule(
        icon="🤖",
        name="Classical ML",
        description="Train Random Forest, SVM, and Logistic Regression classifiers with performance benchmarks.",
        category="classical",
    ),
    ProjectModule(
        icon="🌀",
        name="Quantum ML",
        description="Implement a Variational Quantum Classifier (VQC) and a Quantum Support Vector Classifier (QSVC) using Quantum Kernels.",
        category="quantum",
    ),
    ProjectModule(
        icon="📊",
        name="Comparative Analysis",
        description="Side-by-side benchmarking of classical vs quantum approaches across accuracy, speed, and scalability.",
        category="both",
    ),
]


def get_module_tag_html(category: str) -> str:
    if category == "quantum":
        return '<span class="quantum-tag">QUANTUM</span>'
    if category == "classical":
        return '<span class="classical-tag">CLASSICAL</span>'
    return '<span class="classical-tag">CLASSICAL</span> <span class="quantum-tag">QUANTUM</span>'
