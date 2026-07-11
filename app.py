"""
Future of Data: Quantum Analytics
Main Streamlit application entry point
"""

import streamlit as st

from src.portfolio_utils import PROJECT_MODULES, get_module_tag_html

st.set_page_config(
    page_title="Future of Data: Quantum Analytics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for quantum aesthetic
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #050d1a 0%, #0a1628 50%, #06111f 100%);
    }

    .main-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2fff, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        text-align: center;
    }

    @keyframes shimmer {
        to { background-position: 200% center; }
    }

    .subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #00d4ff88;
        text-align: center;
        font-size: 0.95rem;
        letter-spacing: 0.15em;
        margin-top: -0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #0d1f3c, #111f3a);
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 0 20px #00d4ff15;
    }

    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        color: #00d4ff;
        font-weight: bold;
    }

    .metric-label {
        color: #8899aa;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .section-header {
        font-family: 'Rajdhani', sans-serif;
        color: #00d4ff;
        font-size: 1.4rem;
        font-weight: 600;
        border-left: 3px solid #7b2fff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 0.8rem 0;
    }

    .quantum-tag {
        display: inline-block;
        background: #7b2fff22;
        border: 1px solid #7b2fff55;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #a78bfa;
        font-family: 'Share Tech Mono', monospace;
    }

    .classical-tag {
        display: inline-block;
        background: #00d4ff22;
        border: 1px solid #00d4ff55;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #67e8f9;
        font-family: 'Share Tech Mono', monospace;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050d1a, #0a1628);
        border-right: 1px solid #00d4ff22;
    }

    .stSelectbox label, .stSlider label, .stRadio label {
        color: #8899aa !important;
        font-size: 0.85rem !important;
    }

    hr {
        border-color: #00d4ff22;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">⚛ FUTURE OF DATA: QUANTUM ANALYTICS</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">[ EXPLORING CLASSICAL vs QUANTUM COMPUTING FOR DATA SCIENCE ]</div>',
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown(
    """
This project presents an interactive comparison between classical machine learning and emerging quantum computing approaches for analytics workflows.
It is built as a portfolio-ready Streamlit app that highlights both technical depth and clear storytelling for recruiters, collaborators, and curious visitors.
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# Overview cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-value">5</div>
        <div class="metric-label">Analysis Modules</div>
    </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-value">Qiskit</div>
        <div class="metric-label">Quantum Engine</div>
    </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-value">ML + QML</div>
        <div class="metric-label">Model Types</div>
    </div>""",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-value">Real-time</div>
        <div class="metric-label">Simulation</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Module overview
st.markdown(
    '<div class="section-header">📡 Project Modules</div>', unsafe_allow_html=True
)

for module in PROJECT_MODULES:
    st.markdown(
        f"""
    <div style="background:#0d1f3c; border:1px solid #00d4ff22; border-radius:10px; padding:1rem; margin:0.5rem 0; display:flex; align-items:flex-start; gap:1rem;">
        <span style="font-size:1.8rem">{module.icon}</span>
        <div>
            <div style="font-family:'Rajdhani',sans-serif; color:#e2e8f0; font-size:1.1rem; font-weight:600;">{module.name} &nbsp; {get_module_tag_html(module.category)}</div>
            <div style="color:#64748b; font-size:0.88rem; margin-top:0.3rem;">{module.description}</div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">🚀 Navigate the App</div>', unsafe_allow_html=True
)
st.info(
    "Use the **sidebar** (←) to navigate between modules. Each page is self-contained and interactive."
)

st.markdown("---")
st.markdown(
    "<p style=\"text-align:center; color:#334155; font-size:0.8rem; font-family:'Share Tech Mono',monospace;\">Future of Data: Quantum Analytics · Built with Qiskit, scikit-learn & Streamlit</p>",
    unsafe_allow_html=True,
)
