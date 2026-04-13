"""
app.py — SupplAI: AI Supply Chain Disruption Monitor
=====================================================
Streamlit dashboard with 4 tabs:
  🌐 Network Graph  |  🔥 Risk Analysis  |  🔁 Rerouting  |  🤖 AI Brief

Run with:
    conda activate condaVE
    streamlit run app.py
"""

import sys
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Add src/ to Python path so we can import our modules
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disruption_input import parse_disruption
from news_fetcher     import get_live_disruptions
from weather_monitor  import get_weather_disruptions
from graph_builder    import build_graph, load_supply_metadata, get_graph_summary
from cascade_model    import run_cascade, get_cascade_stats
from risk_scoring     import score_nodes, compute_centrality
from reroute          import find_alternates, format_path
from llm_brief        import generate_brief
from shap_explain     import compute_shap, shap_bar_figure, shap_waterfall_figure, shap_to_text, FEATURE_DESCRIPTIONS, FEATURE_LABELS
from anomaly_detector import load_anomaly_model, score_anomalies, anomaly_bar_figure
from material_flow    import (
    get_edge_material, get_node_material_label,
    get_disrupted_materials, summarise_materials_at_risk,
)
from supply_chain_agent import run_agent

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "SupplAI — AI Supply Chain Intelligence",
    page_icon  = "🔗",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ===== Google Fonts — Inter + Google Sans feel ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ===== Keyframe Animations ===== */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(66, 133, 244, 0.3); }
    50%       { box-shadow: 0 0 22px rgba(66, 133, 244, 0.7), 0 0 40px rgba(234, 67, 53, 0.2); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-5px); }
}
@keyframes ping {
    75%, 100% { transform: scale(2); opacity: 0; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(66,133,244,0.4); }
    33%       { border-color: rgba(234,67,53,0.4); }
    66%       { border-color: rgba(251,188,4,0.4); }
}

/* ===== Global Dark Background ===== */
html, body {
    background-color: #060b18 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Force dark bg on all Streamlit wrappers */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.block-container,
.main,
.main > div,
section[data-testid="stMain"],
div[data-testid="stMainBlockContainer"] {
    background-color: #060b18 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* All generic divs and paragraphs in main */
.stMarkdown, .stMarkdown p, .stMarkdown div,
.element-container p, .element-container div {
    color: #e2e8f0;
}

/* ===== Sidebar — deep navy glassmorphism ===== */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #07101f 0%, #0d1829 50%, #111f38 100%) !important;
    border-right: 1px solid rgba(66,133,244,0.18) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ===== Main content area ===== */
.main .block-container {
    padding: 1.2rem 2rem;
    max-width: 100%;
}

/* ===== Metric cards — Google-style glassmorphism ===== */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9)) !important;
    border: 1px solid rgba(66,133,244,0.2) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease !important;
    animation: fadeInUp 0.4s ease both;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(66,133,244,0.55) !important;
    box-shadow: 0 6px 32px rgba(66,133,244,0.18), 0 0 0 1px rgba(66,133,244,0.1) !important;
    transform: translateY(-2px) !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] div {
    color: #94a3b8 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== Tab styling — Google Material-inspired ===== */
[data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.8) !important;
    border-radius: 10px;
    padding: 5px;
    gap: 3px;
    border: 1px solid rgba(66,133,244,0.12);
    backdrop-filter: blur(8px);
}
[data-baseweb="tab"] {
    border-radius: 7px;
    font-weight: 500;
    color: #64748b !important;
    padding: 9px 18px;
    font-size: 0.88rem;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
}
[data-baseweb="tab"]:hover {
    color: #cbd5e1 !important;
    background: rgba(66,133,244,0.08) !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #1a73e8, #4285f4) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(26,115,232,0.45) !important;
    font-weight: 600 !important;
}
[data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.2rem;
}

/* ===== Horizontal rule ===== */
hr {
    border-color: rgba(51,65,85,0.6) !important;
    margin: 1rem 0;
}

/* ===== Dataframe / tables ===== */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(51,65,85,0.5);
}

/* ===== Buttons — Google Blue primary ===== */
.stButton > button {
    background: linear-gradient(135deg, #1a73e8, #4285f4) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    width: 100%;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(26,115,232,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1557b0, #1a73e8) !important;
    box-shadow: 0 4px 20px rgba(26,115,232,0.5);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(26,115,232,0.3);
}

/* ===== Text inputs ===== */
.stTextArea textarea {
    background: rgba(30,41,59,0.7) !important;
    border: 1px solid rgba(51,65,85,0.8) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextArea textarea::placeholder {
    color: #475569 !important;
}
.stTextArea textarea:focus {
    border-color: #4285f4 !important;
    box-shadow: 0 0 0 3px rgba(66,133,244,0.18) !important;
}

/* ===== Selectbox ===== */
[data-baseweb="select"] > div {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(51,65,85,0.8) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s ease;
}
[data-baseweb="select"] span {
    color: #e2e8f0 !important;
}
[data-baseweb="popover"] { background: #1e293b !important; }
[data-baseweb="menu"]    { background: #1e293b !important; }
[data-baseweb="list-item"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
[data-baseweb="list-item"]:hover {
    background: rgba(66,133,244,0.15) !important;
}

/* ===== Slider ===== */
[data-testid="stSlider"] .stSlider > div { color: #e2e8f0 !important; }
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #475569 !important; }

/* ===== Alert / info boxes ===== */
[data-testid="stAlert"] {
    background: rgba(30,41,59,0.8) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    backdrop-filter: blur(6px);
}

/* ===== Custom cards — glassmorphism ===== */
.risk-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9));
    border: 1px solid rgba(51,65,85,0.7);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    color: #e2e8f0;
    backdrop-filter: blur(8px);
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    animation: fadeInUp 0.35s ease both;
}
.risk-card:hover {
    border-color: rgba(66,133,244,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(66,133,244,0.15);
}
.risk-card-critical { border-left: 4px solid #ea4335; }
.risk-card-high     { border-left: 4px solid #fa7b17; }
.risk-card-medium   { border-left: 4px solid #fbbc04; }
.risk-card-low      { border-left: 4px solid #34a853; }

.route-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9));
    border: 1px solid rgba(51,65,85,0.7);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    backdrop-filter: blur(8px);
    transition: all 0.22s ease;
    animation: fadeInUp 0.35s ease both;
}
.route-found  { border-left: 4px solid #34a853; }
.route-none   { border-left: 4px solid #ea4335; }

/* ===== Section headers — Google gradient ===== */
.section-header {
    background: linear-gradient(90deg, #4285f4, #34a853, #fbbc04, #ea4335);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    font-family: 'Space Grotesk', sans-serif;
    animation: shimmer 4s linear infinite;
}

/* ===== Brief sections ===== */
.brief-section {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9));
    border: 1px solid rgba(51,65,85,0.7);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(8px);
    animation: fadeInUp 0.4s ease both;
}
.brief-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #4285f4;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
}
.brief-content {
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 0.93rem;
}

/* ===== Google-colored severity badges ===== */
.badge-high   { background: rgba(234,67,53,0.18); color: #ff6b6b; border: 1px solid rgba(234,67,53,0.5); border-radius: 6px; padding: 3px 12px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; }
.badge-medium { background: rgba(251,188,4,0.15);  color: #fdd663; border: 1px solid rgba(251,188,4,0.4);  border-radius: 6px; padding: 3px 12px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; }
.badge-low    { background: rgba(52,168,83,0.15);  color: #5cbf7e; border: 1px solid rgba(52,168,83,0.4);  border-radius: 6px; padding: 3px 12px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; }

/* ===== Hero banner ===== */
.hero-banner {
    background: linear-gradient(135deg, #07101f 0%, #0d1829 40%, #091424 70%, #060b18 100%);
    border: 1px solid rgba(66,133,244,0.25);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    animation: pulse-glow 4s ease-in-out infinite;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(66,133,244,0.07) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(234,67,53,0.05) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

/* ===== Badge ===== */
.hackathon-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(66,133,244,0.15), rgba(234,67,53,0.1));
    border: 1px solid rgba(66,133,244,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    animation: borderGlow 3s ease-in-out infinite;
}

/* ===== Live indicator dot ===== */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34a853;
    box-shadow: 0 0 6px rgba(52,168,83,0.8);
    animation: ping 1.5s cubic-bezier(0,0,0.2,1) infinite;
    vertical-align: middle;
    margin-right: 4px;
}

/* ===== Stat pill chips ===== */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(66,133,244,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 500;
}
.stat-chip b { color: #e2e8f0; }

/* ===== Google-brand feature cards ===== */
.feature-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(10,14,26,0.95));
    border: 1px solid rgba(51,65,85,0.6);
    border-radius: 18px;
    padding: 1.8rem 1.5rem;
    width: 180px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    animation: fadeInUp 0.5s ease both;
    cursor: default;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(66,133,244,0.2);
    border-color: rgba(66,133,244,0.35);
}
.feature-card-icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
    animation: float 3s ease-in-out infinite;
}

/* ===== Spinner ===== */
.stSpinner > div { border-top-color: #4285f4 !important; }

/* ===== Scrollbar — Google subtle ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060b18; }
::-webkit-scrollbar-thumb { background: rgba(66,133,244,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(66,133,244,0.6); }

/* ===== Headings ===== */
h1, h2, h3, h4, h5, h6 {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== stInfo / stWarning ===== */
[data-testid="stNotification"] {
    background: rgba(30,41,59,0.85) !important;
    color: #e2e8f0 !important;
    backdrop-filter: blur(8px);
    border-radius: 10px !important;
}

/* ===== Google colors utility classes ===== */
.g-blue   { color: #4285f4; }
.g-red    { color: #ea4335; }
.g-yellow { color: #fbbc04; }
.g-green  { color: #34a853; }

/* ===== Hide / dark-theme the Streamlit top toolbar ===== */
[data-testid="stHeader"],
[data-testid="stToolbar"],
header[data-testid="stHeader"] {
    background: #060b18 !important;
    border-bottom: 1px solid rgba(66,133,244,0.12) !important;
}
/* Status widget + hamburger menu */
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] { display: none !important; }
#stDecoration { display: none !important; }

/* Deploy button area */
[data-testid="stHeader"] button {
    color: #64748b !important;
    background: transparent !important;
}

/* ===== Hide default Streamlit footer ===== */
footer { visibility: hidden !important; }
footer::after { display: none !important; }

/* ===== Download button — Google green style ===== */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1e7e34, #34a853) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    box-shadow: 0 2px 8px rgba(52,168,83,0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #166428, #2d9247) !important;
    box-shadow: 0 4px 16px rgba(52,168,83,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ===== Radio buttons ===== */
[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
}
.stRadio > div {
    gap: 0.5rem;
    flex-wrap: wrap;
}

/* ===== Slider track ===== */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1a73e8, #ea4335) !important;
}

/* ===== Expander — glassmorphism ===== */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(51,65,85,0.6) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 0.75rem 1rem;
}
[data-testid="stExpander"] summary:hover {
    color: #e2e8f0 !important;
}
[data-testid="stExpander"] > div > div {
    background: transparent !important;
}

/* ===== Toggle / checkbox ===== */
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}

/* ===== Progress bar ===== */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1a73e8, #34a853) !important;
}

/* ===== Tight spacing between br tags ===== */
br { line-height: 0.5; }

/* ===== Column gap tightening ===== */
[data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
}

/* ===== Dataframe header ===== */
[data-testid="stDataFrame"] thead th {
    background: rgba(26,115,232,0.12) !important;
    color: #4285f4 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(66,133,244,0.07) !important;
}

/* ===== Spinner text ===== */
[data-testid="stSpinner"] p {
    color: #4285f4 !important;
    font-weight: 500 !important;
}

/* ===== st.info overrides for Google blue ===== */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 4px solid #4285f4 !important;
}

/* ===== Success toasts ===== */
[data-testid="stToast"] {
    background: rgba(15,23,42,0.95) !important;
    border: 1px solid rgba(52,168,83,0.4) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
    color: #e2e8f0 !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}

/* ===== Number input ===== */
[data-testid="stNumberInput"] input {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(51,65,85,0.8) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ===== Column wrapper spacing ===== */
div[data-testid="column"] {
    padding: 0 0.3rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders — run once per session, not on every re-render
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔗 Building supply chain graph …")
def _load_graph():
    G = build_graph()
    return G

@st.cache_resource(show_spinner="📊 Loading city metadata …")
def _load_supply():
    return load_supply_metadata()

@st.cache_resource(show_spinner="🧠 Loading pretrained delay model …")
def _load_delay_model():
    from delay_model import load_model
    return load_model()

@st.cache_data(show_spinner=False)
def _compute_centrality(_G):
    return compute_centrality(_G)

@st.cache_resource(show_spinner="🔍 Loading anomaly detection model …")
def _load_anomaly_model():
    return load_anomaly_model()


# ---------------------------------------------------------------------------
# Graph visualisation with Plotly
# ---------------------------------------------------------------------------
TIER_COLORS = {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#22c55e", 5: "#3b82f6"}
TIER_LABELS = {1: "Tier-1 Direct Supplier", 2: "Tier-2 Supplier", 3: "Tier-3 Upstream",
               4: "Tier-4 Deep Upstream", 5: "Tier-5 Raw Source"}


def build_plotly_graph(
    G:              nx.DiGraph,
    cascade_result: dict,
    risk_df:        pd.DataFrame,
    supply_df:      pd.DataFrame,
    seed_nodes:     list,
    color_by:       str = "risk",   # "risk" | "tier"
) -> go.Figure:
    """Build an interactive Plotly geo-scatter map coloured by risk level or supply-chain tier."""

    # Build node risk lookup
    risk_lookup = {}
    if not risk_df.empty:
        for _, row in risk_df.iterrows():
            risk_lookup[row["node"]] = row["risk_score"]

    cascade_nodes = set(cascade_result.keys())
    seed_set      = set(seed_nodes)

    # Colour + size helpers
    def node_colour(node):
        if color_by == "tier":
            if node not in cascade_nodes:
                return "#1e293b"
            tier = int(G.nodes[node].get("tier", 3))
            return TIER_COLORS.get(tier, "#334155")
        # default: risk score
        if node in seed_set:   return "#ef4444"
        rs = risk_lookup.get(node, 0)
        if rs >= 0.65:         return "#f97316"
        if rs >= 0.40:         return "#eab308"
        if rs >= 0.10:         return "#3b82f6"
        return "#334155"

    def node_size(node):
        if node in seed_set:   return 16
        rs = risk_lookup.get(node, 0)
        if rs >= 0.65:         return 13
        if rs >= 0.40:         return 11
        if rs >= 0.10:         return 9
        return 7

    # ------------------------------------------------------------------ #
    # Arc lines — only high-priority cascade edges to keep JSON small    #
    # Priority: seed→depth1, depth1→depth2, then sample remaining.      #
    # Hard cap at 400 arcs to prevent browser JSON.parse failure.       #
    # ------------------------------------------------------------------ #
    MAX_ARCS = 400

    # Depth lookup for quick access
    depth_map = cascade_result  # node -> depth int

    # Sort edges by priority: lower max-depth of endpoints = higher priority
    priority_edges = []
    for src, dst in G.edges():
        if src in cascade_nodes and dst in cascade_nodes:
            d = depth_map.get(src, 99) + depth_map.get(dst, 99)
            priority_edges.append((d, src, dst))
    priority_edges.sort()   # lowest combined depth first
    priority_edges = priority_edges[:MAX_ARCS]

    arc_lats, arc_lons = [], []
    mid_lats, mid_lons, mid_texts = [], [], []

    for _, src, dst in priority_edges:
        src_lat = G.nodes[src].get("lat", 0)
        src_lon = G.nodes[src].get("lon", 0)
        dst_lat = G.nodes[dst].get("lat", 0)
        dst_lon = G.nodes[dst].get("lon", 0)
        arc_lats += [src_lat, dst_lat, None]
        arc_lons += [src_lon, dst_lon, None]

        mid_lat  = (src_lat + dst_lat) / 2
        mid_lon  = (src_lon + dst_lon) / 2
        material = get_edge_material(G, src, dst)
        src_name = G.nodes[src].get("city_name", src)
        dst_name = G.nodes[dst].get("city_name", dst)
        hover_txt = (
            f"<b>{material}</b><br>"
            f"{src_name} to {dst_name}<br>"
            f"{G.nodes[src].get('country','?')} to {G.nodes[dst].get('country','?')}"
        )
        mid_lats.append(mid_lat)
        mid_lons.append(mid_lon)
        mid_texts.append(hover_txt)

    arc_trace = go.Scattergeo(
        lat=arc_lats, lon=arc_lons,
        mode="lines",
        line=dict(width=0.8, color="rgba(239,68,68,0.35)"),
        hoverinfo="none",
        name="Disrupted Routes",
        showlegend=False,
    )

    # Midpoint hover markers (capped — same count as arcs)
    material_trace = go.Scattergeo(
        lat=mid_lats, lon=mid_lons,
        mode="markers",
        hoverinfo="text",
        text=mid_texts,
        marker=dict(
            size=6,
            color="rgba(251,191,36,0.7)",
            symbol="diamond",
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        ),
        name="Material Flows",
        showlegend=False,
    )

    # ------------------------------------------------------------------ #
    # Node scatter                                                        #
    # ------------------------------------------------------------------ #
    lats, lons, texts, colours, sizes = [], [], [], [], []

    for node in G.nodes():
        lat = G.nodes[node].get("lat", 0)
        lon = G.nodes[node].get("lon", 0)
        if lat == 0 and lon == 0:
            continue

        nd    = G.nodes[node]
        rs    = risk_lookup.get(node, 0)
        depth = cascade_result.get(node, -1)
        tier  = int(nd.get("tier", 3))
        status_str   = f"Cascade depth: {depth}" if depth >= 0 else "Unaffected"
        material_lbl = get_node_material_label(G, node)
        tier_lbl     = TIER_LABELS.get(tier, f"Tier-{tier}")
        hover = (
            f"<b>{nd.get('city_name', node)}</b><br>"
            f"{nd.get('country', '?')} | {nd.get('region', '?')}<br>"
            f"Produces: <b>{material_lbl}</b><br>"
            f"Category: {nd.get('product_category', '?')}<br>"
            f"<b>{tier_lbl}</b><br>"
            f"{status_str}<br>"
            f"Risk Score: <b>{rs:.3f}</b>"
        )
        lats.append(lat)
        lons.append(lon)
        texts.append(hover)
        colours.append(node_colour(node))
        sizes.append(node_size(node))

    node_trace = go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers",
        hoverinfo="text",
        text=texts,
        marker=dict(
            color=colours,
            size=sizes,
            line=dict(width=0.8, color="rgba(255,255,255,0.15)"),
            symbol="circle",
        ),
        name="Supply Chain Nodes",
        showlegend=False,
    )

    fig = go.Figure(data=[arc_trace, material_trace, node_trace])
    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#1a2235",
        showocean=True,      oceancolor="#0a0e1a",
        showcoastlines=True, coastlinecolor="#2d3f5f",
        showframe=False,
        showcountries=True,  countrycolor="#1e3050",
        bgcolor="#0a0e1a",
    )
    fig.update_layout(
        title=dict(
            text="Supply Chain Network — Global Disruption Impact Map",
            font=dict(color="#e2e8f0", size=16, family="Space Grotesk, Inter, sans-serif"),
        ),
        paper_bgcolor="#060b18",
        margin=dict(l=0, r=0, t=45, b=0),
        height=580,
        annotations=[
            dict(x=0.01, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text=(
                     "&#9632; Tier-1 Direct &nbsp; &#9632; Tier-2 &nbsp; &#9632; Tier-3 &nbsp; &#9632; Tier-4 &nbsp; &#9632; Tier-5 &nbsp; &#9632; Unaffected"
                     if color_by == "tier" else
                     "&#9632; Disruption Source &nbsp; &#9632; Critical Cascade &nbsp; &#9632; High Risk &nbsp; &#9632; Monitoring &nbsp; &#9632; Unaffected"
                 ),
                 font=dict(color="#94a3b8", size=11, family="Space Grotesk, Inter, sans-serif"),
                 bgcolor="rgba(10,14,26,0.7)", borderpad=6, align="left"),
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Cascade Animation — Plotly animated figure, one frame per depth level
# ---------------------------------------------------------------------------
def build_cascade_animation(
    G:              nx.DiGraph,
    cascade_result: dict,
    risk_df:        pd.DataFrame,
    seed_nodes:     list,
) -> go.Figure | None:
    """
    Animated Plotly geo-scatter that shows the disruption spreading
    depth by depth across the supply chain network.

    Frame N shows all nodes at cascade depth ≤ N lit up in their
    risk colour; nodes not yet reached remain as dim dots.
    """
    if not cascade_result:
        return None

    max_depth = max(cascade_result.values())
    seed_set  = set(seed_nodes)

    # Risk score lookup for colour grading
    risk_lookup: dict[str, float] = {}
    if not risk_df.empty:
        for _, row in risk_df.iterrows():
            risk_lookup[row["node"]] = float(row["risk_score"])

    # Depth → hex colour (seed=red → orange → yellow → blue)
    DEPTH_COLS = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#22c55e", "#8b5cf6"]

    def _node_col(node: str, shown_depth: int) -> str:
        if node in seed_set:
            return "#ef4444"
        d = cascade_result.get(node, 9999)
        if d > shown_depth:
            return "#1e2535"       # not yet reached — near-invisible
        return DEPTH_COLS[min(d, len(DEPTH_COLS) - 1)]

    def _node_size(node: str, shown_depth: int) -> int:
        d = cascade_result.get(node, 9999)
        if d > shown_depth:
            return 4
        if node in seed_set:
            return 17
        return max(7, 14 - d * 2)

    def _node_opacity(node: str, shown_depth: int) -> float:
        d = cascade_result.get(node, 9999)
        return 1.0 if d <= shown_depth else 0.18

    # Pre-collect all node coordinates once
    all_nodes = [
        (n, G.nodes[n].get("lat", 0), G.nodes[n].get("lon", 0))
        for n in G.nodes()
        if not (G.nodes[n].get("lat", 0) == 0 and G.nodes[n].get("lon", 0) == 0)
    ]

    def _build_traces(shown_depth: int):
        # Arc lines for edges where BOTH endpoints are in the visible cascade
        visible = {n for n, d in cascade_result.items() if d <= shown_depth}
        arc_lats, arc_lons = [], []
        for src, dst in G.edges():
            if src in visible and dst in visible:
                arc_lats += [G.nodes[src].get("lat", 0), G.nodes[dst].get("lat", 0), None]
                arc_lons += [G.nodes[src].get("lon", 0), G.nodes[dst].get("lon", 0), None]

        arc_trace = go.Scattergeo(
            lat=arc_lats, lon=arc_lons,
            mode="lines",
            line=dict(width=0.9, color="rgba(239,68,68,0.30)"),
            hoverinfo="none",
            showlegend=False,
        )

        # Nodes
        lats, lons, texts, cols, sizes, opacities = [], [], [], [], [], []
        for node, lat, lon in all_nodes:
            nd  = G.nodes[node]
            d   = cascade_result.get(node, 9999)
            col = _node_col(node, shown_depth)
            sz  = _node_size(node, shown_depth)
            op  = _node_opacity(node, shown_depth)

            if d <= shown_depth:
                status = "🔴 Disruption Origin" if d == 0 else f"⚡ Cascade — depth {d}"
            else:
                status = "✅ Unaffected"

            hover = (
                f"<b>{nd.get('city_name', node)}</b><br>"
                f"🌍 {nd.get('country', '?')} · {nd.get('region', '?')}<br>"
                f"📦 {nd.get('product_category', '?')} · Tier {nd.get('tier', '?')}<br>"
                f"{status}"
            )
            lats.append(lat); lons.append(lon)
            texts.append(hover); cols.append(col)
            sizes.append(sz);   opacities.append(op)

        node_trace = go.Scattergeo(
            lat=lats, lon=lons,
            mode="markers",
            hoverinfo="text",
            text=texts,
            marker=dict(
                color=cols, size=sizes, opacity=opacities,
                line=dict(width=0.6, color="rgba(255,255,255,0.12)"),
            ),
            showlegend=False,
        )
        return arc_trace, node_trace

    # Build one Plotly frame per depth level
    frames = []
    for depth in range(max_depth + 1):
        arc_t, node_t = _build_traces(depth)
        n_affected = sum(1 for d in cascade_result.values() if d <= depth)
        frames.append(go.Frame(
            data=[arc_t, node_t],
            name=str(depth),
            layout=go.Layout(
                annotations=[dict(
                    x=0.5, y=1.04, xref="paper", yref="paper", showarrow=False,
                    text=(
                        f"<b style='color:#ef4444;'>Cascade Depth {depth}</b>"
                        f" — <b style='color:#a5b4fc;'>{n_affected}</b> nodes affected"
                    ),
                    font=dict(color="#e2e8f0", size=15, family="Space Grotesk, Inter, sans-serif"),
                )]
            ),
        ))

    # Initial data = first frame (just seeds)
    init_arc, init_node = _build_traces(0)

    fig = go.Figure(data=[init_arc, init_node], frames=frames)

    # Geo layout
    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#1a2235",
        showocean=True,      oceancolor="#0a0e1a",
        showcoastlines=True, coastlinecolor="#2d3f5f",
        showcountries=True,  countrycolor="#1e3050",
        showframe=False,
        bgcolor="#060b18",
    )

    # Animation controls
    play_btn = dict(
        label="▶  Play Cascade",
        method="animate",
        args=[None, {
            "frame":      {"duration": 950, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 350, "easing": "cubic-in-out"},
        }],
    )
    pause_btn = dict(
        label="⏸  Pause",
        method="animate",
        args=[[None], {
            "frame":  {"duration": 0, "redraw": False},
            "mode":   "immediate",
        }],
    )

    slider_steps = [
        dict(
            args=[[str(d)], {"frame": {"duration": 400, "redraw": True}, "mode": "immediate",
                             "transition": {"duration": 200}}],
            label=f"D{d}",
            method="animate",
        )
        for d in range(max_depth + 1)
    ]

    fig.update_layout(
        paper_bgcolor="#060b18",
        margin=dict(l=0, r=0, t=60, b=80),
        height=560,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.5, y=-0.06,
            xanchor="center", yanchor="top",
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(color="#e2e8f0", size=13, family="Space Grotesk, Inter, sans-serif"),
            buttons=[play_btn, pause_btn],
            pad={"r": 10, "t": 4},
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.05, len=0.9,
            y=0.0,
            xanchor="left", yanchor="top",
            currentvalue=dict(
                prefix="Cascade depth: ",
                visible=True,
                xanchor="center",
                font=dict(color="#a5b4fc", size=13, family="Space Grotesk, Inter, sans-serif"),
            ),
            font=dict(color="#94a3b8", size=11),
            bgcolor="#1e293b",
            bordercolor="#334155",
            tickcolor="#334155",
            transition=dict(duration=300),
            pad=dict(b=10, t=30),
        )],
        annotations=[dict(
            x=0.5, y=1.04, xref="paper", yref="paper", showarrow=False,
            text=f"<b style='color:#ef4444;'>Cascade Depth 0</b> — "
                 f"<b style='color:#a5b4fc;'>{sum(1 for d in cascade_result.values() if d==0)}</b> origin nodes",
            font=dict(color="#e2e8f0", size=15, family="Space Grotesk, Inter, sans-serif"),
        )],
    )

    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Logo/title
        st.markdown("""
        <div style="text-align:center; padding: 0.8rem 0 0.5rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.4rem; animation: float 3s ease-in-out infinite;">🔗</div>
            <div style="font-size: 1.5rem; font-weight: 800;
                        background: linear-gradient(90deg, #4285f4, #34a853, #fbbc04, #ea4335);
                        background-size: 200% auto;
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        animation: shimmer 3s linear infinite;
                        font-family: 'Space Grotesk', sans-serif;">
                SupplAI
            </div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.15rem; letter-spacing: 0.06em; text-transform: uppercase;"> 
                AI Supply Chain Intelligence
            </div>
            <div style="margin-top: 0.6rem;">
            </div>
        </div>
        <hr style="border-color: rgba(66,133,244,0.15); margin: 0.8rem 0;">
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Disruption Input")

        # Text area for disruption description
        event_text = st.text_area(
            label       = "Describe the disruption event:",
            value       = st.session_state.get("event_text", ""),
            height      = 120,
            placeholder = "e.g. Factory shutdown in China affecting electronics supply chain...",
            help        = "Describe the supply chain disruption in plain English.",
        )

        # Severity override
        severity_override = st.selectbox(
            "Severity Override",
            options=["Auto-detect", "High", "Medium", "Low"],
            index=0,
            help="Override the auto-detected severity level.",
        )

        # Cascade depth
        max_depth = st.slider(
            "Cascade Depth",
            min_value=1, max_value=6,
            value=4,
            help="How many hops downstream to propagate the disruption.",
        )

        st.markdown("---")

        # ---- Live Intelligence Section ----
        st.markdown("### 📡 Live Intelligence")
        auto_refresh = st.toggle("🔄 Auto-refresh (30 min)", value=False,
                                 help="Automatically re-fetch news & weather every 30 minutes")
        col_n, col_w = st.columns(2)
        with col_n:
            if st.button("🌍 News", width="stretch", help="Fetch latest geopolitical news"):
                with st.spinner("Fetching news …"):
                    news_events = get_live_disruptions()
                weather_events = st.session_state.get("weather_events", [])
                st.session_state["live_events"] = news_events + weather_events
                st.rerun()
        with col_w:
            if st.button("🌩️ Weather", width="stretch", help="Check live weather & earthquakes"):
                with st.spinner("Checking weather & quakes …"):
                    weather_events = get_weather_disruptions()
                news_events = st.session_state.get("live_events", [])
                st.session_state["weather_events"] = weather_events
                st.session_state["live_events"] = news_events + weather_events
                st.rerun()

        live_events = st.session_state.get("live_events")
        if live_events:
            st.markdown(f"**{len(live_events)} disruption(s) detected:**")
            for i, evt in enumerate(live_events):
                sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(evt["severity"], "⚪")
                src_icon = {"USGS": "🌋", "OpenWeatherMap": "🌩️", "Open-Meteo": "🌩️"}.get(evt.get("source",""), "📰")
                label = f"{sev_icon}{src_icon} {evt['title'][:40]}{'…' if len(evt['title']) > 40 else ''}"
                if st.button(label, key=f"live_evt_{i}", width="stretch"):
                    st.session_state["event_text"] = evt["event_text"]
                    st.rerun()
        elif live_events is not None:
            st.info("No disruptions detected right now.")

        st.markdown("---")

        # Demo scenario button
        st.markdown("**🧪 Quick Demo:**")
        if st.button("🏭 China Electronics Shutdown"):
            st.session_state["event_text"] = (
                "Factory shutdown in China affecting electronics supply chain"
            )
            st.rerun()

        if st.button("🌊 Southeast Asia Flood"):
            st.session_state["event_text"] = (
                "Severe flooding in Vietnam affecting textile and electronics manufacturing"
            )
            st.rerun()

        if st.button("⚡ Korea Semiconductor Strike"):
            st.session_state["event_text"] = (
                "Labor strike in South Korea disrupting semiconductor production"
            )
            st.rerun()

        st.markdown("---")

        # Analyse button
        run_analysis = st.button("🚀 Analyse Disruption", type="primary", width="stretch")

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.68rem; color: #334155; text-align: center; padding: 0.5rem 0; line-height: 1.8;">
            <span style="color:#4285f4;">■</span>
            <span style="color:#ea4335;">■</span>
            <span style="color:#fbbc04;">■</span>
            <span style="color:#34a853;">■</span>
            <br>
            <span style="color:#475569;">Powered by</span>
            <span style="color:#4285f4; font-weight:600;"> Google Gemini</span> ·
            <span style="color:#64748b;">NetworkX · XGBoost · SHAP</span>
            <br>
            <span style="color:#334155;">© 2025 SupplAI</span>
        </div>
        """, unsafe_allow_html=True)

    # Sync text area value to session state
    st.session_state["event_text"] = event_text

    st.session_state["_auto_refresh_on"] = auto_refresh
    return event_text, severity_override, max_depth, run_analysis


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Init session state ----
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "event_text" not in st.session_state:
        st.session_state["event_text"] = ""
    if "live_events" not in st.session_state:
        st.session_state["live_events"] = None
    if "auto_load_attempted" not in st.session_state:
        st.session_state["auto_load_attempted"] = False
    if "weather_events" not in st.session_state:
        st.session_state["weather_events"] = []
    if "_auto_refresh_on" not in st.session_state:
        st.session_state["_auto_refresh_on"] = False
    if "_last_refresh_count" not in st.session_state:
        st.session_state["_last_refresh_count"] = 0

    # ---- Load resources ----
    G         = _load_graph()
    supply_df = _load_supply()
    centrality = _compute_centrality(G)

    # ---- Sidebar ----
    event_text, severity_override, max_depth, run_analysis = render_sidebar()

    # ---- Auto-refresh trigger ----
    if st.session_state.get("_auto_refresh_on"):
        refresh_count = st_autorefresh(interval=30 * 60 * 1000, key="autorefresh")
        if refresh_count > st.session_state.get("_last_refresh_count", 0):
            st.session_state["_last_refresh_count"] = refresh_count
            st.session_state["auto_load_attempted"]  = False   # force re-fetch on next cycle

    # ---- Hero Header ----
    summary = get_graph_summary(G)
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.3rem;">
                    <span style="font-size:2rem;">🔗</span>
                    <h1 style="margin:0; font-size:1.75rem; font-weight:800;
                               background:linear-gradient(90deg,#4285f4,#34a853,#fbbc04,#ea4335);
                               background-size:200% auto;
                               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                               animation:shimmer 4s linear infinite;
                               font-family:'Space Grotesk',sans-serif; line-height:1.2;">SupplAI</h1>
                </div>
                <p style="margin:0; color:#64748b; font-size:0.88rem; letter-spacing:0.02em;">
                    <span class="live-dot"></span> Live AI Supply Chain Disruption Intelligence Platform
                </p>
            </div>
            <div style="display:flex; gap:0.6rem; flex-wrap:wrap; align-items:center;">
                <span class="stat-chip">🌐 <b>{summary['nodes']:,}</b> Nodes</span>
                <span class="stat-chip">🔗 <b>{summary['edges']:,}</b> Routes</span>
                <span class="stat-chip">🌍 <b>{summary['countries']}</b> Countries</span>
                <span class="stat-chip">📊 <b>{summary['avg_degree']:,.1f}</b> Avg Degree</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Secondary metric row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌐 Network Nodes",   f"{summary['nodes']:,}")
    col2.metric("🔗 Supply Routes",   f"{summary['edges']:,}")
    col3.metric("🌍 Countries",       f"{summary['countries']}")
    col4.metric("📊 Avg Connections", f"{summary['avg_degree']:,.1f}")
    col5.metric("⚡ Graph Connected", "Yes" if summary["is_connected"] else "Partial")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Helper: Disruption Focus Map — show ONLY affected nodes + reroute paths
    # ---------------------------------------------------------------------------
    def build_focus_map(G, cascade_result, risk_df, reroute_suggestions, seed_nodes):
        """
        Build a Plotly Scattergeo showing only:
          • Seed nodes (red, pulsing border)
          • Cascade nodes coloured by depth
          • Alternate route paths (green lines + intermediate nodes)
        Everything else is hidden.
        """
        import plotly.graph_objects as go

        depth_colors = {0: "#ef4444", 1: "#f97316", 2: "#eab308",
                        3: "#3b82f6", 4: "#22c55e", 5: "#8b5cf6"}

        # Collect all nodes to show
        focus_nodes = set(cascade_result.keys())

        # Collect reroute path nodes & edges
        reroute_edges = []   # list of (lat_seq, lon_seq, label)
        for r in reroute_suggestions:
            if r["status"] == "✅ Alternate Found" and r.get("alternate_path"):
                path = r["alternate_path"]
                focus_nodes.update(path)
                lats = [G.nodes[n].get("lat", 0) for n in path if n in G.nodes]
                lons = [G.nodes[n].get("lon", 0) for n in path if n in G.nodes]
                src_name  = r.get("source_name",      path[0])
                dst_name  = r.get("destination_name", path[-1])
                vstat     = r.get("route_validation", "")
                reroute_edges.append((lats, lons, f"{src_name} → {dst_name}<br>{vstat}"))

        traces = []

        # ── Reroute path lines (green) ────────────────────────────────────────
        for lats, lons, label in reroute_edges:
            # Add None separators to break lines between segments
            seg_lats, seg_lons = [], []
            for i in range(len(lats)):
                seg_lats.append(lats[i])
                seg_lons.append(lons[i])
                if i < len(lats) - 1:
                    seg_lats.append(None)
                    seg_lons.append(None)
            traces.append(go.Scattergeo(
                lat=seg_lats, lon=seg_lons,
                mode="lines",
                line=dict(color="#22c55e", width=2.5),
                hoverinfo="text", text=label,
                name="Alternate Route",
                showlegend=False,
            ))

        # ── Disrupted path lines (red dashed) ────────────────────────────────
        for r in reroute_suggestions:
            if r.get("disrupted_path") and len(r["disrupted_path"]) >= 2:
                path = r["disrupted_path"]
                lats = [G.nodes[n].get("lat", 0) for n in path if n in G.nodes]
                lons = [G.nodes[n].get("lon", 0) for n in path if n in G.nodes]
                traces.append(go.Scattergeo(
                    lat=lats, lon=lons,
                    mode="lines",
                    line=dict(color="#ef4444", width=1.5, dash="dot"),
                    hoverinfo="skip",
                    name="Disrupted Route",
                    showlegend=False,
                ))

        # ── Nodes ─────────────────────────────────────────────────────────────
        node_lats, node_lons, node_text, node_colors, node_sizes, node_symbols = \
            [], [], [], [], [], []

        risk_lookup = {}
        if not risk_df.empty and "node" in risk_df.columns:
            risk_lookup = dict(zip(risk_df["node"], risk_df["risk_score"]))

        for node in sorted(focus_nodes):
            if node not in G.nodes:
                continue
            nd   = G.nodes[node]
            depth = cascade_result.get(node, -1)
            lat, lon = nd.get("lat", 0), nd.get("lon", 0)
            city = nd.get("city_name", node)
            country = nd.get("country", "")
            industry = nd.get("product_category", "")
            risk  = risk_lookup.get(node, 0)

            if node in seed_nodes:
                color  = "#ef4444"
                size   = 16
                symbol = "star"
                label  = f"<b>DISRUPTED: {city}</b><br>{country} | {industry}<br>Depth 0 (Origin)"
            elif depth >= 0:
                color  = depth_colors.get(depth, "#64748b")
                size   = max(8, 14 - depth * 2)
                symbol = "circle"
                label  = f"<b>{city}</b><br>{country} | {industry}<br>Cascade Depth {depth}<br>Risk: {risk:.0%}"
            else:
                # reroute intermediate node not in cascade
                color  = "#22c55e"
                size   = 10
                symbol = "diamond"
                label  = f"<b>{city}</b> (Alternate Hop)<br>{country} | {industry}"

            node_lats.append(lat)
            node_lons.append(lon)
            node_text.append(label)
            node_colors.append(color)
            node_sizes.append(size)
            node_symbols.append(symbol)

        traces.append(go.Scattergeo(
            lat=node_lats, lon=node_lons,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                line=dict(color="#ffffff", width=1),
                opacity=0.95,
            ),
            text=node_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        # ── Legend annotations ─────────────────────────────────────────────────
        n_alt = len([r for r in reroute_suggestions if r["status"] == "✅ Alternate Found"])
        n_dis = len(seed_nodes)

        fig = go.Figure(traces)
        fig.update_layout(
            geo=dict(
                showland=True, landcolor="#1e293b",
                showocean=True, oceancolor="#0f172a",
                showcountries=True, countrycolor="#334155",
                showcoastlines=True, coastlinecolor="#475569",
                bgcolor="#0f172a", projection_type="natural earth",
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            margin=dict(l=0, r=0, t=40, b=0),
            height=520,
            title=dict(
                text=(
                    f"<b style='color:#f8fafc'>Disruption Focus</b>"
                    f"  <span style='color:#ef4444;font-size:13px'>★ {n_dis} disrupted</span>"
                    f"  <span style='color:#22c55e;font-size:13px'>⟶ {n_alt} alternate route(s)</span>"
                    f"  <span style='color:#64748b;font-size:13px'>● cascade nodes</span>"
                ),
                font=dict(size=14, color="#f8fafc"),
                x=0.02,
            ),
        )
        return fig

    # ---------------------------------------------------------------------------
    # Helper: run full analysis pipeline for a given disruption_info dict
    # ---------------------------------------------------------------------------
    def _run_pipeline(disruption_info, max_depth):
        with st.spinner("⚡ Simulating cascade propagation …"):
            cascade_result = run_cascade(G, disruption_info["affected_nodes"], max_depth)
            cascade_stats  = get_cascade_stats(cascade_result)

        with st.spinner("📊 Scoring risk nodes …"):
            try:
                delay_artifact = _load_delay_model()
            except FileNotFoundError as model_err:
                st.warning(f"{model_err}")
                delay_artifact = None
            except Exception:
                delay_artifact = None
            risk_df = score_nodes(G, cascade_result, centrality, delay_artifact)

        with st.spinner("🔁 Finding alternate routes …"):
            reroute_suggestions = find_alternates(
                G, disruption_info["affected_nodes"], cascade_result, supply_df=supply_df
            )

        with st.spinner("🔍 Computing SHAP explainability …"):
            shap_results = {}
            shap_context = None
            if delay_artifact is not None:
                try:
                    shap_results = compute_shap(delay_artifact, risk_df, G, top_n=20)
                    if shap_results and not risk_df.empty:
                        top_node = risk_df.iloc[0]["node"]
                        top_name = risk_df.iloc[0]["city_name"]
                        if top_node in shap_results:
                            shap_context = shap_to_text(shap_results[top_node], top_name)
                except Exception as _shap_err:
                    st.warning(f"SHAP computation skipped: {_shap_err}")

        with st.spinner("🤖 Generating AI operations brief …"):
            brief = generate_brief(
                disruption_info, risk_df, reroute_suggestions,
                shap_context=shap_context,
            )

        with st.spinner("🔎 Running anomaly detection …"):
            try:
                anomaly_artifact = _load_anomaly_model()
                anomaly_df = score_anomalies(anomaly_artifact, G, supply_df)
            except FileNotFoundError as model_err:
                st.warning(f"{model_err}")
                anomaly_df = pd.DataFrame()
            except Exception:
                anomaly_df = pd.DataFrame()

        with st.spinner("🤖 Running AI Agent loop …"):
            try:
                mat_summary_for_agent = summarise_materials_at_risk(G, cascade_result)
                agent_result = run_agent(
                    G                   = G,
                    cascade_result      = cascade_result,
                    risk_df             = risk_df,
                    reroute_suggestions = reroute_suggestions,
                    material_summary    = mat_summary_for_agent,
                    anomaly_df          = anomaly_df,
                    disruption_info     = disruption_info,
                )
            except Exception as _agent_err:
                print(f"  [agent] Error: {_agent_err}")
                agent_result = None

        return {
            "disruption_info":     disruption_info,
            "cascade_result":      cascade_result,
            "cascade_stats":       cascade_stats,
            "risk_df":             risk_df,
            "reroute_suggestions": reroute_suggestions,
            "brief":               brief,
            "shap_results":        shap_results,
            "anomaly_df":          anomaly_df,
            "agent_result":        agent_result,
        }

    # ---------------------------------------------------------------------------
    # Auto-load on first visit: fetch live news and analyse top event
    # ---------------------------------------------------------------------------
    if st.session_state["results"] is None and not st.session_state.get("auto_load_attempted"):
        st.session_state["auto_load_attempted"] = True
        with st.spinner("📡 Fetching live intelligence — news + weather + earthquakes …"):
            try:
                news_events    = get_live_disruptions()
            except Exception:
                news_events    = []
            try:
                weather_events = get_weather_disruptions()
            except Exception:
                weather_events = []
            live_events = news_events + weather_events

        if live_events:
            st.session_state["live_events"]    = live_events
            st.session_state["weather_events"] = weather_events
            priority  = {"high": 0, "medium": 1, "low": 2}
            top_event = min(live_events, key=lambda e: priority.get(e["severity"], 1))
            st.session_state["event_text"] = top_event["event_text"]
            st.session_state["results"]    = _run_pipeline(top_event, max_depth)
            st.rerun()

    # ---- Run analysis on button click ----
    if run_analysis and event_text.strip():
        with st.spinner("🔍 Parsing disruption event …"):
            disruption_info = parse_disruption(event_text.strip())
            if severity_override != "Auto-detect":
                disruption_info["severity"] = severity_override.lower()

        src_icon  = "🤖" if disruption_info.get("llm_source") == "gemini" else "🔑"
        src_label = "AI-classified" if disruption_info.get("llm_source") == "gemini" else "Keyword-matched"
        st.info(
            f"**Event parsed** {src_icon} {src_label} | "
            f"🏙️ {len(disruption_info['affected_nodes'])} seed nodes identified "
            f"| ⚠️ Severity: **{disruption_info['severity'].upper()}** "
            f"| 📂 Category: **{disruption_info['category'].replace('_', ' ').title()}**"
        )
        if disruption_info.get("reasoning"):
            st.markdown(
                f"<div style='background:#0f172a; border-left:3px solid #6366f1; "
                f"padding:0.55rem 1rem; border-radius:0 8px 8px 0; margin-top:-0.6rem; "
                f"color:#94a3b8; font-size:0.86rem;'>"
                f"🧠 <b style='color:#a5b4fc;'>AI Reasoning:</b> {disruption_info['reasoning']}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.session_state["results"] = _run_pipeline(disruption_info, max_depth)

    elif run_analysis and not event_text.strip():
        st.warning("Please enter a disruption description in the sidebar.")

    # ---- Display results ----
    results = st.session_state.get("results")

    if results is None:
        st.markdown("""
        <div style="text-align:center; padding: 3.5rem 2rem 1rem; background: transparent;">
            <div style="margin-bottom:1.2rem;">
            </div>
            <div style="font-size: 3.5rem; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;">🔗</div>
            <h1 style="font-size:2.5rem; font-weight:800; margin-bottom:0.6rem; line-height:1.2;
                       background:linear-gradient(90deg,#4285f4 0%,#34a853 33%,#fbbc04 66%,#ea4335 100%);
                       background-size:200% auto;
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       animation:shimmer 4s linear infinite;
                       font-family:'Space Grotesk',sans-serif;">
                SupplAI
            </h1>
            <p style="color:#64748b; font-size:1.05rem; max-width:580px; margin:0 auto 0.8rem; line-height:1.6;">
                Real-time AI supply chain disruption intelligence powered by
                <span style="color:#4285f4;font-weight:600;">Google Gemini</span>.
                Simulate cascades · Score risk · Plan reroutes · Brief your CSCO.
            </p>
            <div style="display:flex; justify-content:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:2.5rem;">
                <span class="stat-chip">⚡ Cascade Simulation</span>
                <span class="stat-chip">🧠 SHAP Explainability</span>
                <span class="stat-chip">🔍 Anomaly Detection</span>
                <span class="stat-chip">🤖 AI Agent Loop</span>
            </div>
        </div>

        <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap; margin:0 auto; max-width:1000px; padding-bottom:1rem;">
            <div class="feature-card" style="border-top:3px solid #4285f4;">
                <div class="feature-card-icon" style="animation-delay:0s;">🌐</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">Network Graph</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">Interactive geo-scatter supply chain map with cascade animation</div>
            </div>
            <div class="feature-card" style="border-top:3px solid #ea4335;">
                <div class="feature-card-icon" style="animation-delay:0.3s;">🔥</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">Risk Analysis</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">XGBoost ML scoring with tier breakdown and material flows</div>
            </div>
            <div class="feature-card" style="border-top:3px solid #34a853;">
                <div class="feature-card-icon" style="animation-delay:0.6s;">🔁</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">Rerouting</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">Dijkstra alternate paths with upstream validation checks</div>
            </div>
            <div class="feature-card" style="border-top:3px solid #fbbc04;">
                <div class="feature-card-icon" style="animation-delay:0.9s;">🤖</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">Gemini AI Brief</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">Structured CSCO operations brief generated by Gemini 1.5</div>
            </div>
            <div class="feature-card" style="border-top:3px solid #8b5cf6;">
                <div class="feature-card-icon" style="animation-delay:1.2s;">🔍</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">ML Explainability</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">SHAP waterfall & bar charts for full model transparency</div>
            </div>
            <div class="feature-card" style="border-top:3px solid #f97316;">
                <div class="feature-card-icon" style="animation-delay:1.5s;">🚨</div>
                <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.35rem;font-family:'Space Grotesk',sans-serif;">Anomaly Detection</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">Isolation Forest for structural network outlier detection</div>
            </div>
        </div>

        <div style="text-align:center;margin-top:1.5rem;">
            <p style="color:#334155;font-size:0.8rem;">
                <span style="color:#4285f4;">■</span>
                <span style="color:#ea4335;">■</span>
                <span style="color:#fbbc04;">■</span>
                <span style="color:#34a853;">■</span>
                &nbsp; Powered by Google Gemini · NetworkX · XGBoost · SHAP · Plotly
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ---- Unpack results ----
    disruption_info     = results["disruption_info"]
    cascade_result      = results["cascade_result"]
    cascade_stats       = results["cascade_stats"]
    risk_df             = results["risk_df"]
    reroute_suggestions = results["reroute_suggestions"]
    brief               = results["brief"]
    shap_results        = results.get("shap_results", {})

    # ---- Cascade stats banner — premium redesign ----
    sev = disruption_info["severity"].upper()
    sev_color  = {"HIGH": "#ea4335", "MEDIUM": "#fbbc04", "LOW": "#34a853"}.get(sev, "#fbbc04")
    sev_bg     = {"HIGH": "rgba(234,67,53,0.12)", "MEDIUM": "rgba(251,188,4,0.10)", "LOW": "rgba(52,168,83,0.12)"}.get(sev, "rgba(251,188,4,0.10)")
    sev_border = {"HIGH": "rgba(234,67,53,0.35)", "MEDIUM": "rgba(251,188,4,0.30)", "LOW": "rgba(52,168,83,0.30)"}.get(sev, "rgba(251,188,4,0.30)")
    cat_label  = disruption_info['category'].replace('_',' ').title()
    n_countries = risk_df['country'].nunique() if not risk_df.empty else 0

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(15,23,42,0.92),rgba(6,11,24,0.96));
                border:1px solid rgba(66,133,244,0.18); border-radius:16px;
                padding:1rem 1.4rem; margin:0.4rem 0;
                backdrop-filter:blur(10px);
                box-shadow:0 2px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
                display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
        <span style="background:{sev_bg}; color:{sev_color};
                     border:1px solid {sev_border}; border-radius:8px;
                     padding:5px 14px; font-size:0.75rem; font-weight:800;
                     letter-spacing:0.08em; text-transform:uppercase;
                     font-family:'Space Grotesk',sans-serif;">{sev} SEVERITY</span>
        <span class="stat-chip">📂 {cat_label}</span>
        <span class="stat-chip">🏙️ <b>{cascade_stats['seed_count']}</b> origins</span>
        <span class="stat-chip">⚡ <b>{cascade_stats['total_affected']}</b> nodes</span>
        <span class="stat-chip">📏 depth <b>{cascade_stats['max_depth']}</b></span>
        <span class="stat-chip">🌍 <b>{n_countries}</b> countries</span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Risk threshold alert banner ----
    if not risk_df.empty:
        critical_nodes = risk_df[risk_df["risk_score"] >= 0.8]
        if not critical_nodes.empty:
            top_critical = critical_nodes.head(3)
            node_labels  = " &nbsp;·&nbsp; ".join(
                f"<b style='color:#fca5a5;'>{r['city_name']}</b> <span style='color:#7f1d1d;'>({r['country']})</span> <span style='color:#ef4444;font-weight:700;'>{r['risk_score']:.2f}</span>"
                for _, r in top_critical.iterrows()
            )
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(127,29,29,0.35),rgba(69,10,10,0.5));
                        border:1px solid rgba(234,67,53,0.4); border-radius:12px;
                        padding:0.85rem 1.2rem; margin:0.5rem 0;
                        backdrop-filter:blur(8px);
                        display:flex; align-items:center; gap:1rem;">
                <span style="font-size:1.4rem;">🚨</span>
                <div style="flex:1;">
                    <div style="color:#fca5a5; font-weight:700; font-size:0.9rem; margin-bottom:0.25rem;
                                font-family:'Space Grotesk',sans-serif;">
                        CRITICAL RISK ALERT &nbsp;&mdash;&nbsp; {len(critical_nodes)} node(s) above 0.80 threshold
                    </div>
                    <div style="color:#fecaca; font-size:0.82rem; line-height:1.6;">{node_labels}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Deep-tier hidden risk banner ----
    if not risk_df.empty:
        hidden_risk = risk_df[(risk_df["tier"].astype(int) >= 3) & (risk_df["risk_score"] >= 0.40)]
        if not hidden_risk.empty:
            tier3_ct = len(hidden_risk[hidden_risk["tier"].astype(int) == 3])
            tier4_ct = len(hidden_risk[hidden_risk["tier"].astype(int) >= 4])
            parts    = []
            if tier3_ct: parts.append(f"<b style='color:#c4b5fd;'>{tier3_ct}</b> Tier-3")
            if tier4_ct: parts.append(f"<b style='color:#a78bfa;'>{tier4_ct}</b> Tier-4")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(26,10,46,0.7),rgba(45,27,78,0.75));
                        border:1px solid rgba(124,58,237,0.4); border-radius:12px;
                        padding:0.75rem 1.2rem; margin:0.4rem 0;
                        backdrop-filter:blur(8px);
                        display:flex; align-items:center; gap:1rem;">
                <span style="font-size:1.3rem;">🔍</span>
                <div>
                    <div style="color:#c4b5fd; font-weight:700; font-size:0.88rem;
                                font-family:'Space Grotesk',sans-serif; margin-bottom:0.15rem;">
                        DEEP-TIER HIDDEN RISK &nbsp;&mdash;&nbsp; {" + ".join(parts)} upstream nodes at elevated risk
                    </div>
                    <div style="color:#a78bfa; font-size:0.8rem;">
                        These Tier-3/4 suppliers feed into your Tier-1 chain.
                        See <b>Risk Analysis → Tier Breakdown</b> for details.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- 7 tabs ----
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌐 Network Graph",
        "🔥 Risk Analysis",
        "🔁 Rerouting",
        "🤖 AI Brief",
        "🔍 ML Explainability",
        "🚨 Anomaly Detection",
        "🧠 AI Agent",
    ])

    # ==================================================================
    # TAB 1 — Network Graph
    # ==================================================================
    with tab1:
        st.markdown('<div class="section-header">Supply Chain Network</div>', unsafe_allow_html=True)
        st.markdown(
            f"Showing **{G.number_of_nodes()}** supply chain nodes. "
            f"**{cascade_stats['total_affected']}** nodes are impacted in this disruption cascade.",
            unsafe_allow_html=True,
        )

        # View toggle
        view_mode = st.radio(
            "View mode",
            ["🗺️ Risk Map", "🏭 Tier Structure Map", "🎬 Cascade Animation", "🎯 Disruption Focus"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if view_mode == "🎬 Cascade Animation":
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Press <b style='color:#a5b4fc;'>▶ Play Cascade</b> to watch the disruption spread "
                "across the supply chain network in real time. Use the depth slider to step manually."
                "</span>",
                unsafe_allow_html=True,
            )
            anim_fig = build_cascade_animation(
                G, cascade_result, risk_df,
                seed_nodes=disruption_info["affected_nodes"],
            )
            if anim_fig:
                st.plotly_chart(anim_fig, width="stretch")
            else:
                st.info("Not enough cascade data to animate.")

        elif view_mode == "🏭 Tier Structure Map":
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Nodes coloured by their supply-chain tier relative to your company — "
                "<b style='color:#ef4444;'>Tier-1</b> are your direct suppliers, "
                "<b style='color:#eab308;'>Tier-3/4</b> are deep upstream. "
                "Over one-third of real disruptions start at Tier-3/4."
                "</span>",
                unsafe_allow_html=True,
            )
            fig = build_plotly_graph(
                G, cascade_result, risk_df, supply_df,
                seed_nodes=disruption_info["affected_nodes"],
                color_by="tier",
            )
            st.plotly_chart(fig, width="stretch")

        elif view_mode == "🎯 Disruption Focus":
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Shows <b style='color:#ef4444;'>only the affected nodes</b> and "
                "<b style='color:#22c55e;'>alternate route paths</b> — everything else hidden. "
                "Red stars = origin. Dashed red = blocked route. Solid green = safe alternate."
                "</span>",
                unsafe_allow_html=True,
            )
            focus_fig = build_focus_map(
                G, cascade_result, risk_df, reroute_suggestions,
                seed_nodes=set(disruption_info["affected_nodes"]),
            )
            st.plotly_chart(focus_fig, width="stretch")

        else:
            fig = build_plotly_graph(
                G, cascade_result, risk_df, supply_df,
                seed_nodes=disruption_info["affected_nodes"],
                color_by="risk",
            )
            st.plotly_chart(fig, width="stretch")

        # Depth breakdown
        st.markdown("#### Cascade Depth Breakdown")
        depth_data = cascade_stats.get("by_depth", {})
        if depth_data:
            depth_df = pd.DataFrame(
                [{"Cascade Depth": f"Depth {d}", "Nodes Affected": n}
                 for d, n in sorted(depth_data.items())]
            )
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(depth_df, hide_index=True, width="stretch")
            with col_b:
                # Simple bar chart using plotly
                bar_fig = go.Figure(go.Bar(
                    x=[str(d) for d in sorted(depth_data.keys())],
                    y=[depth_data[d] for d in sorted(depth_data.keys())],
                    marker=dict(
                        color=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#22c55e", "#6366f1"][:len(depth_data)],
                    ),
                    text=[depth_data[d] for d in sorted(depth_data.keys())],
                    textposition="outside",
                ))
                bar_fig.update_layout(
                    paper_bgcolor="#060b18", plot_bgcolor="#060b18",
                    xaxis=dict(title="Cascade Depth", color="#94a3b8", gridcolor="#1e293b"),
                    yaxis=dict(title="Nodes Affected", color="#94a3b8", gridcolor="#1e293b"),
                    font=dict(color="#e2e8f0", family="Space Grotesk, Inter, sans-serif"),
                    height=280, margin=dict(t=20, b=30, l=20, r=20),
                )
                st.plotly_chart(bar_fig, width="stretch")

    # ==================================================================
    # TAB 2 — Risk Analysis
    # ==================================================================
    with tab2:
        st.markdown('<div class="section-header">Risk Scoring Analysis</div>', unsafe_allow_html=True)

        if risk_df.empty:
            st.warning("No risk data — no disruption nodes matched the graph.")
        else:
            # Top 3 critical nodes as metric cards
            top3 = risk_df.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="risk-card {'risk-card-critical' if row['risk_score']>=0.65 else 'risk-card-high' if row['risk_score']>=0.40 else 'risk-card-medium'}">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">#{i+1} Highest Risk</div>
                        <div style="font-size:1.15rem;font-weight:700;margin:0.3rem 0">{row['city_name']}</div>
                        <div style="color:#94a3b8;font-size:0.85rem;">{row['country']} · {row['product_category']}</div>
                        <div style="margin-top:0.6rem;display:flex;align-items:center;gap:0.5rem;">
                            <span style="font-size:1.4rem;font-weight:700;color:{'#ef4444' if row['risk_score']>=0.65 else '#f97316' if row['risk_score']>=0.40 else '#eab308'}">{row['risk_score']:.3f}</span>
                            <span style="font-size:0.8rem;">{row['risk_level']}</span>
                        </div>
                        <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">{row['status']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------------------------
            # Tier Structure Breakdown
            # ----------------------------------------------------------
            st.markdown("#### 🏭 Supply Chain Tier Breakdown")
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "How disruption spreads <i>relative to your company</i> — "
                "Tier-1 = direct suppliers you buy from, Tier-3/4 = deep upstream you may not even monitor."
                "</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            tier_summary = (
                risk_df.groupby("tier", as_index=False)
                .agg(
                    disrupted   =("node", "count"),
                    avg_risk    =("risk_score", "mean"),
                    max_risk    =("risk_score", "max"),
                    critical_ct =("risk_score", lambda x: int((x >= 0.65).sum())),
                )
                .sort_values("tier")
            )

            tier_cols = st.columns(min(len(tier_summary), 4))
            for i, (_, tr) in enumerate(tier_summary.iterrows()):
                t      = int(tr["tier"])
                col_h  = TIER_COLORS.get(t, "#94a3b8")
                lbl    = TIER_LABELS.get(t, f"Tier-{t}")
                badge  = "🔴 Critical" if tr["max_risk"] >= 0.65 else "🟠 High" if tr["max_risk"] >= 0.40 else "🟡 Medium"
                with tier_cols[i % 4]:
                    st.markdown(f"""
                    <div class="risk-card" style="border-top:3px solid {col_h}; text-align:center; padding:1rem;">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:0.3rem;">{lbl}</div>
                        <div style="font-size:2rem;font-weight:800;color:{col_h};">{int(tr['disrupted'])}</div>
                        <div style="color:#94a3b8;font-size:0.78rem;margin-bottom:0.4rem;">nodes disrupted</div>
                        <div style="font-size:0.82rem;color:#e2e8f0;">
                            Avg risk <b>{tr['avg_risk']:.3f}</b>
                        </div>
                        <div style="margin-top:0.4rem;font-size:0.78rem;">{badge}</div>
                        {f'<div style="margin-top:0.4rem;color:#ef4444;font-size:0.75rem;font-weight:600;">{int(tr["critical_ct"])} critical node(s)</div>' if tr["critical_ct"] > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)

            # Deep-tier alert
            deep_tier_df = risk_df[risk_df["tier"].astype(int) >= 3]
            deep_critical = deep_tier_df[deep_tier_df["risk_score"] >= 0.40]
            if not deep_critical.empty:
                def _tier_lbl(t):
                    return TIER_LABELS.get(int(t), f"Tier-{int(t)}")
                sample_nodes = ", ".join(
                    f"<b>{r['city_name']}</b> ({r['country']}, {_tier_lbl(r['tier'])}, score {r['risk_score']:.2f})"
                    for _, r in deep_critical.head(3).iterrows()
                )
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a0a2e,#2d1b4e);
                            border:1px solid #7c3aed; border-radius:10px;
                            padding:0.9rem 1.2rem; margin:0.8rem 0;">
                    <div style="color:#c4b5fd; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">
                        🔍 Deep-Tier Risk Detected — {len(deep_critical)} Tier-3/4 nodes at elevated risk
                    </div>
                    <div style="color:#ddd6fe; font-size:0.85rem; line-height:1.6;">
                        These upstream suppliers are often invisible to monitoring systems yet feed directly
                        into your Tier-1 chain. Over one-third of real disruptions originate here.<br>
                        <span style="color:#a78bfa;">{sample_nodes}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Tier risk bar chart
            tier_fig = go.Figure()
            for _, tr in tier_summary.iterrows():
                t = int(tr["tier"])
                tier_fig.add_trace(go.Bar(
                    x=[f"Tier-{t}"],
                    y=[round(tr["avg_risk"], 4)],
                    name=TIER_LABELS.get(t, f"Tier-{t}"),
                    marker_color=TIER_COLORS.get(t, "#94a3b8"),
                    text=[f"{tr['avg_risk']:.3f}"],
                    textposition="outside",
                    hovertemplate=f"<b>{TIER_LABELS.get(t,'Tier-'+str(t))}</b><br>"
                                  f"Avg risk: {tr['avg_risk']:.3f}<br>"
                                  f"Disrupted nodes: {int(tr['disrupted'])}<br>"
                                  f"<extra></extra>",
                ))
            tier_fig.update_layout(
                paper_bgcolor="#060b18", plot_bgcolor="#060b18",
                xaxis=dict(title="Supply Chain Tier", color="#94a3b8", gridcolor="#1e293b"),
                yaxis=dict(title="Average Risk Score", color="#94a3b8", gridcolor="#1e293b",
                           range=[0, 1]),
                font=dict(color="#e2e8f0", family="Space Grotesk, Inter, sans-serif"),
                height=250, margin=dict(t=15, b=30, l=20, r=20),
                showlegend=False,
                bargap=0.3,
            )
            st.plotly_chart(tier_fig, width="stretch")

            st.markdown("---")

            # Full risk table
            tbl_col, export_col = st.columns([5, 1])
            with tbl_col:
                st.markdown("#### All Affected Nodes")
            with export_col:
                st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
                display_cols = [
                    "node", "city_name", "country", "product_category",
                    "tier", "cascade_depth", "delay_prob",
                    "centrality_score", "risk_score", "risk_level", "status"
                ]
                available_cols = [c for c in display_cols if c in risk_df.columns]
                export_df = risk_df[available_cols].rename(columns={
                    "node": "Node ID", "city_name": "City", "country": "Country",
                    "product_category": "Product", "tier": "Tier",
                    "cascade_depth": "Depth", "delay_prob": "Delay Prob",
                    "centrality_score": "Centrality", "risk_score": "Risk Score",
                    "risk_level": "Risk Level", "status": "Status",
                })
                st.download_button(
                    label="⬇️ Export CSV",
                    data=export_df.to_csv(index=False),
                    file_name="supplai_risk_report.csv",
                    mime="text/csv",
                    width="stretch",
                )

            st.dataframe(
                export_df,
                hide_index=True,
                width="stretch",
                height=400,
            )

            # ----------------------------------------------------------
            # Materials at Risk
            # ----------------------------------------------------------
            st.markdown("#### 📦 Materials at Risk")
            mat_summary = summarise_materials_at_risk(G, cascade_result)
            if mat_summary.empty:
                st.info("No material flow data available for this disruption.")
            else:
                # KPI row
                total_routes   = int(mat_summary["Disrupted Routes"].sum())
                unique_mats    = len(mat_summary)
                total_countries= int(mat_summary["Countries Affected"].max())
                mk1, mk2, mk3 = st.columns(3)
                mk1.metric("📦 Material Types Disrupted", unique_mats)
                mk2.metric("🔗 Disrupted Supply Routes",  total_routes)
                mk3.metric("🌍 Max Countries Affected",   total_countries)

                st.markdown("<br>", unsafe_allow_html=True)

                # Bar chart — routes disrupted per material
                mat_fig = go.Figure(go.Bar(
                    x=mat_summary["Material Flow"],
                    y=mat_summary["Disrupted Routes"],
                    marker=dict(
                        color=mat_summary["Disrupted Routes"],
                        colorscale=[[0,"#3b82f6"],[0.5,"#f97316"],[1,"#ef4444"]],
                        showscale=False,
                    ),
                    text=mat_summary["Disrupted Routes"],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Disrupted routes: %{y}<br>"
                        "<extra></extra>"
                    ),
                ))
                mat_fig.update_layout(
                    paper_bgcolor="#060b18", plot_bgcolor="#060b18",
                    xaxis=dict(
                        title="Material Flow Type", color="#94a3b8",
                        gridcolor="#1e293b", tickangle=-30,
                    ),
                    yaxis=dict(title="Disrupted Routes", color="#94a3b8", gridcolor="#1e293b"),
                    font=dict(color="#e2e8f0", family="Space Grotesk, Inter, sans-serif"),
                    height=300, margin=dict(t=15, b=100, l=20, r=20),
                )
                st.plotly_chart(mat_fig, width="stretch")

                # Detail table — expand to see specific items and example routes
                with st.expander("🔍 View full material flow breakdown"):
                    st.dataframe(
                        mat_summary,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Material Flow":      st.column_config.TextColumn("Material Type", width="medium"),
                            "Specific Items":     st.column_config.TextColumn("Specific Items at Risk", width="large"),
                            "Disrupted Routes":   st.column_config.NumberColumn("Routes Disrupted", width="small"),
                            "Countries Affected": st.column_config.NumberColumn("Countries", width="small"),
                            "Example Route":      st.column_config.TextColumn("Example Route", width="large"),
                        },
                    )

                    # Per-edge detail table
                    st.markdown("##### All disrupted material flows (edge level)")
                    edge_df = get_disrupted_materials(G, cascade_result)
                    if not edge_df.empty:
                        display_edge = edge_df[[
                            "from_city","from_country","material",
                            "to_city","to_country","from_tier","to_tier"
                        ]].rename(columns={
                            "from_city":    "From City",
                            "from_country": "From Country",
                            "material":     "Material",
                            "to_city":      "To City",
                            "to_country":   "To Country",
                            "from_tier":    "From Tier",
                            "to_tier":      "To Tier",
                        })
                        st.dataframe(display_edge, hide_index=True, width="stretch", height=300)

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk score distribution
            st.markdown("#### Risk Score Distribution")
            # Build per-score color list for the histogram bars
            _score_colours = risk_df["risk_score"].apply(
                lambda s: "#ef4444" if s >= 0.65 else "#f97316" if s >= 0.40 else "#eab308" if s >= 0.20 else "#22c55e"
            ).tolist()
            risk_fig = go.Figure()
            risk_fig.add_trace(go.Histogram(
                x=risk_df["risk_score"],
                nbinsx=20,
                marker=dict(
                    color=risk_df["risk_score"].tolist(),
                    colorscale=[[0, "#22c55e"], [0.33, "#eab308"], [0.66, "#f97316"], [1, "#ef4444"]],
                    cmin=0, cmax=1,
                    showscale=False,
                ),
                name="Risk Scores",
            ))
            risk_fig.update_layout(
                paper_bgcolor="#060b18", plot_bgcolor="#060b18",
                xaxis=dict(title="Risk Score", color="#94a3b8", gridcolor="#1e293b"),
                yaxis=dict(title="Node Count", color="#94a3b8", gridcolor="#1e293b"),
                font=dict(color="#e2e8f0", family="Space Grotesk, Inter, sans-serif"),
                height=260, margin=dict(t=10, b=30, l=20, r=20),
                bargap=0.1,
            )
            st.plotly_chart(risk_fig, width="stretch")

    # ==================================================================
    # TAB 3 — Rerouting
    # ==================================================================
    with tab3:
        st.markdown('<div class="section-header">Alternate Route Recommendations</div>', unsafe_allow_html=True)

        if not reroute_suggestions:
            st.info("No rerouting suggestions — the disruption may be too isolated or no alternate paths exist.")
        else:
            n_found   = len([r for r in reroute_suggestions if r["status"] == "✅ Alternate Found"])
            n_blocked = len(reroute_suggestions) - n_found
            n_clean   = len([r for r in reroute_suggestions if r.get("route_validation") == "✅ Clean"])
            n_hidden  = len([r for r in reroute_suggestions if r.get("route_validation") == "⛔ Hidden Dependency"])
            n_partial = len([r for r in reroute_suggestions if r.get("route_validation") == "⚠️ Partial Exposure"])

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("🔄 Total Suggestions", len(reroute_suggestions))
            col_r2.metric("✅ Alternates Found",  n_found)
            col_r3.metric("⚠️ Blocked Routes",    n_blocked)

            st.markdown("<br>", unsafe_allow_html=True)

            # Upstream validation summary banner
            if n_found > 0:
                if n_hidden > 0:
                    st.markdown(
                        f'<div style="border-left:4px solid #ef4444;background:#1e0a0a;padding:0.8rem 1.2rem;border-radius:8px;margin-bottom:1rem;">'
                        f'<b style="color:#ef4444">⛔ Hidden Dependency Alert</b> &nbsp;·&nbsp; '
                        f'<span style="color:#fca5a5">{n_hidden} alternate route(s) route through suppliers whose upstream chains '
                        f'are also exposed to the disrupted region. These routes may fail under the same event.</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif n_partial > 0:
                    st.markdown(
                        f'<div style="border-left:4px solid #eab308;background:#1a1500;padding:0.8rem 1.2rem;border-radius:8px;margin-bottom:1rem;">'
                        f'<b style="color:#eab308">⚠️ Partial Exposure Detected</b> &nbsp;·&nbsp; '
                        f'<span style="color:#fde68a">{n_partial} route(s) have intermediate nodes with partial upstream '
                        f'exposure. Monitor closely.</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="border-left:4px solid #22c55e;background:#0a1a0a;padding:0.8rem 1.2rem;border-radius:8px;margin-bottom:1rem;">'
                        f'<b style="color:#22c55e">✅ All {n_clean} alternate route(s) validated clean</b> &nbsp;·&nbsp; '
                        f'<span style="color:#86efac">No hidden upstream dependencies on the disrupted region detected.</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            for i, route in enumerate(reroute_suggestions):
                found   = route["status"] == "✅ Alternate Found"
                card_cls = "route-found" if found else "route-none"

                alt_path_str  = format_path(route.get("alternate_path", []), supply_df)
                orig_path_str = format_path(route.get("disrupted_path", []), supply_df)

                delta_str = ""
                if found and route.get("distance_delta_km") is not None:
                    delta = route["distance_delta_km"]
                    detour = route["detour_pct"]
                    sign  = "+" if delta >= 0 else ""
                    delta_str = f"<span style='color:#eab308'>{sign}{delta:,.0f} km ({sign}{detour:.1f}%)</span>"

                # Validation badge styling
                vstat = route.get("route_validation", "N/A")
                if vstat == "✅ Clean":
                    vbadge_color = "#22c55e"
                    vbadge_bg    = "rgba(34,197,94,0.12)"
                elif vstat == "⚠️ Partial Exposure":
                    vbadge_color = "#eab308"
                    vbadge_bg    = "rgba(234,179,8,0.12)"
                elif vstat == "⛔ Hidden Dependency":
                    vbadge_color = "#ef4444"
                    vbadge_bg    = "rgba(239,68,68,0.12)"
                else:
                    vbadge_color = "#64748b"
                    vbadge_bg    = "rgba(100,116,139,0.12)"

                exposure_pct  = int(route.get("worst_exposure_ratio", 0) * 100)
                vnote         = route.get("validation_note", "")
                exposed_ints  = route.get("exposed_intermediates", [])

                # Build exposed-node detail HTML
                exposed_detail_html = ""
                if exposed_ints:
                    items_html = "".join(
                        f'<div style="padding:0.4rem 0.6rem;background:#0f172a;border-radius:6px;margin-top:0.3rem;font-size:0.78rem;color:#cbd5e1;">'
                        f'<b style="color:{vbadge_color}">{e["city_name"]}</b> — '
                        f'{int(e["exposure_ratio"]*100)}% upstream exposed'
                        f'{"  <span style=color:#94a3b8>(" + ", ".join(e["exposed_upstream"][:3]) + ")</span>" if e["exposed_upstream"] else ""}'
                        f'</div>'
                        for e in exposed_ints
                    )
                    exposed_detail_html = (
                        f'<div style="margin-top:0.5rem;">'
                        f'<span style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">Exposed Intermediates</span>'
                        f'{items_html}'
                        f'</div>'
                    )

                validation_block = (
                    f'<div style="margin-top:0.8rem;padding:0.7rem 0.9rem;border-radius:8px;'
                    f'background:{vbadge_bg};border:1px solid {vbadge_color}33;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.4rem;">'
                    f'<span style="font-weight:700;color:{vbadge_color};font-size:0.88rem;">🔍 Upstream Validation: {vstat}</span>'
                    f'{"<span style=font-size:0.78rem;color:" + vbadge_color + ";>" + str(exposure_pct) + "% upstream exposed</span>" if found else ""}'
                    f'</div>'
                    f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{vnote}</div>'
                    f'{exposed_detail_html}'
                    f'</div>'
                ) if found else ""

                financial_block = ""
                if found and route.get("orig_cost_usd") is not None and route.get("alt_cost_usd") is not None:
                    orig_cost = route["orig_cost_usd"]
                    alt_cost = route["alt_cost_usd"]
                    delta_cost = route["cost_delta_usd"]
                    sign_cost = "+" if delta_cost > 0 else ""
                    color_cost = "#22c55e" if delta_cost <= 0 else "#ef4444"
                    financial_block = (
                        f'<div style="margin-top:0.8rem;padding:0.7rem 0.9rem;border-radius:8px;'
                        f'background:rgba(15,23,42,0.6);border:1px solid rgba(66,133,244,0.3);">'
                        f'<div style="font-size:0.75rem;color:#93c5fd;text-transform:uppercase;font-weight:700;margin-bottom:0.3rem;letter-spacing:0.05em;">💸 SeaRates Freight Cost Estimate (Inc. Tariffs)</div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">'
                        f'<span style="color:#94a3b8;font-size:0.85rem;">Original: <b style="color:#e2e8f0">${orig_cost:,.0f}</b></span>'
                        f'<span style="color:#94a3b8;font-size:0.85rem;">Alternate: <b style="color:#e2e8f0">${alt_cost:,.0f}</b></span>'
                        f'<span style="color:{color_cost};font-weight:700;font-size:0.88rem;padding:0.15rem 0.4rem;background:{color_cost}22;border-radius:4px;">{sign_cost}${delta_cost:,.0f} Cost</span>'
                        f'</div></div>'
                    )

                # Remove blank lines by dynamically building strings
                f_block_str = financial_block if financial_block else ""
                v_block_str = validation_block if validation_block else ""
                
                dist_str = f"<span>📏 Extra distance: <b>{delta_str}</b></span>" if delta_str else ""
                tariff_str = f"<span>💰 Max Border Tariff: <b>{str(route.get('max_tariff_pct', 0))}%</b></span>" if route.get('max_tariff_pct') else ""

                card_html = f'''<div class="route-card {card_cls}">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
<div><span style="font-size:0.75rem;color:#64748b;text-transform:uppercase;">Route {i+1}</span>
<div style="font-weight:700;font-size:1rem;margin:0.2rem 0">📍 {route['source_name']} → 📍 {route['destination_name']}</div></div>
<span style="font-size:0.9rem;">{route['status']}</span></div>
'''
                if orig_path_str and orig_path_str != 'No path':
                    card_html += f'<div style="margin-top:0.8rem;padding:0.6rem;background:#0f172a;border-radius:8px;font-size:0.82rem;color:#94a3b8;"><b style=color:#ef4444>⚠️ Original:</b> {orig_path_str}</div>\n'
                if found and alt_path_str and alt_path_str != 'No path':
                    card_html += f'<div style="margin-top:0.4rem;padding:0.6rem;background:#0f172a;border-radius:8px;font-size:0.82rem;color:#94a3b8;"><b style=color:#22c55e>✅ Alternate:</b> {alt_path_str}</div>\n'
                if f_block_str:
                    card_html += f_block_str + '\n'
                if v_block_str:
                    card_html += v_block_str + '\n'
                card_html += f'<div style="margin-top:0.6rem;display:flex;gap:1.5rem;font-size:0.82rem;color:#64748b;flex-wrap:wrap;">\n<span>🛤️ Orig hops: <b>{route["hops_original"]}</b></span>\n<span>🛤️ Alt hops: <b>{route["hops_alternate"]}</b></span>\n'
                if dist_str: card_html += dist_str + '\n'
                if tariff_str: card_html += tariff_str + '\n'
                card_html += '</div></div>'
                
                st.markdown(card_html, unsafe_allow_html=True)

    # ==================================================================
    # TAB 4 — AI Brief
    # ==================================================================
    with tab4:
        st.markdown('<div class="section-header">AI Operations Brief</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------
        # CSCO Supplier Decision Table
        # ----------------------------------------------------------
        st.markdown("#### 📊 Supplier Decision Table")
        st.markdown(
            "<span style='color:#64748b;font-size:0.85rem;'>"
            "Structured action decision for every at-risk supplier — scan in 10 seconds, know exactly what to do."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if not risk_df.empty:
            def _csco_row(r):
                score = r["risk_score"]
                tier  = int(r.get("tier", 3))
                depth = int(r.get("cascade_depth", 0))
                prod  = r.get("product_category", "General")
                cent  = float(r.get("centrality_score", 0))
                out_d = cascade_result.get(r["node"], 0)

                if score >= 0.65:
                    decision = "REPLACE"
                    action   = "Find 2+ alternatives within 72 hrs. Initiate emergency procurement."
                    dec_col  = "#ef4444"
                elif score >= 0.40:
                    decision = "MONITOR"
                    action   = "Check weekly. Pre-qualify backup supplier. Prepare contingency."
                    dec_col  = "#f97316"
                else:
                    decision = "STANDARD"
                    action   = "Standard monitoring protocol. No immediate action required."
                    dec_col  = "#22c55e"

                # Build reason sentence
                reason_parts = []
                if depth == 0:
                    reason_parts.append("Direct disruption origin")
                else:
                    reason_parts.append(f"Cascade depth {depth} from origin")
                if tier <= 2:
                    reason_parts.append(f"Tier-{tier} critical hub")
                if cent >= 0.5:
                    reason_parts.append("high network centrality (bottleneck)")
                reason_parts.append(f"{prod} sector")

                return {
                    "node":     r["node"],
                    "Supplier": r["city_name"],
                    "Country":  r["country"],
                    "Sector":   prod,
                    "Tier":     tier,
                    "Score":    round(score, 3),
                    "Decision": decision,
                    "Action":   action,
                    "Reason":   ". ".join(reason_parts) + ".",
                    "_dec_col": dec_col,
                }

            csco_rows = [_csco_row(r) for _, r in risk_df.head(15).iterrows()]

            # KPI strip
            n_replace  = sum(1 for x in csco_rows if x["Decision"] == "REPLACE")
            n_monitor  = sum(1 for x in csco_rows if x["Decision"] == "MONITOR")
            n_standard = sum(1 for x in csco_rows if x["Decision"] == "STANDARD")
            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("🔴 REPLACE",  n_replace,  help="Immediate alternative sourcing required")
            kc2.metric("🟠 MONITOR",  n_monitor,  help="Elevated watch, contingency prep")
            kc3.metric("🟢 STANDARD", n_standard, help="Standard monitoring")

            st.markdown("<br>", unsafe_allow_html=True)

            # Render decision table rows
            for row in csco_rows:
                dec_col = row["_dec_col"]
                st.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px;
                            padding:0.8rem 1.1rem; margin-bottom:0.5rem;
                            border-left:4px solid {dec_col};">
                    <div style="display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
                        <div style="min-width:160px;">
                            <div style="font-weight:700; color:#e2e8f0;">{row['Supplier']}</div>
                            <div style="color:#64748b; font-size:0.78rem;">
                                {row['Country']} · {row['Sector']} · Tier {row['Tier']}
                            </div>
                        </div>
                        <div style="font-size:1.25rem; font-weight:700;
                                    color:{'#ef4444' if row['Score']>=0.65 else '#f97316' if row['Score']>=0.40 else '#22c55e'};">
                            {row['Score']:.3f}
                        </div>
                        <div style="background:{dec_col}22; color:{dec_col};
                                    border-radius:6px; padding:3px 12px;
                                    font-weight:700; font-size:0.85rem; letter-spacing:0.05em;">
                            {row['Decision']}
                        </div>
                        <div style="flex:1; color:#cbd5e1; font-size:0.85rem;">
                            {row['Action']}
                        </div>
                    </div>
                    <div style="margin-top:0.45rem; color:#475569; font-size:0.78rem;
                                font-style:italic; padding-left:0.2rem;">
                        ↳ {row['Reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Export
            st.markdown("<br>", unsafe_allow_html=True)
            csco_export = pd.DataFrame([
                {k: v for k, v in r.items() if not k.startswith("_") and k != "node"}
                for r in csco_rows
            ])
            st.download_button(
                label     = "⬇️ Export Decision Table CSV",
                data      = csco_export.to_csv(index=False),
                file_name = "csco_decision_table.csv",
                mime      = "text/csv",
            )

        st.markdown("---")

        _brief_source = brief.get("source", "template")
        if _brief_source == "template":
            source_label = "📋 Template Brief"
        elif "gemini" in _brief_source:
            source_label = "🤖 Generated by Gemini 2.5 Flash"
        elif "groq" in _brief_source:
            source_label = "🤖 Generated by Groq GPT-OSS 120B"
        else:
            source_label = f"🤖 Generated by AI"
        confidence = brief.get("confidence", "Medium")
        conf_colour = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}.get(confidence, "#eab308")

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <span style="color:#64748b;font-size:0.85rem;">{source_label}</span>
            <span style="color:{conf_colour};font-weight:600;background:#1e293b;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
                Confidence: {confidence}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Executive Summary
        st.markdown(f"""
        <div class="brief-section">
            <div class="brief-title">📋 Executive Summary</div>
            <div class="brief-content">{brief.get('executive_summary', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Two columns: top risks + actions
        bc1, bc2 = st.columns(2)

        with bc1:
            risks = brief.get("top_risks", [])
            risks_html = "".join(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #1e293b;color:#cbd5e1;font-size:0.9rem;'>"
                f"<span style='color:#ef4444;font-weight:600;'>{i+1}.</span> {r}</div>"
                for i, r in enumerate(risks)
            )
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">⚠️ Top Risks</div>
                {risks_html}
            </div>
            """, unsafe_allow_html=True)

        with bc2:
            actions = brief.get("immediate_actions", [])
            actions_html = "".join(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #1e293b;color:#cbd5e1;font-size:0.9rem;'>"
                f"<span style='color:#22c55e;font-weight:600;'>{i+1}.</span> {a}</div>"
                for i, a in enumerate(actions)
            )
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">🚀 Immediate Actions</div>
                {actions_html}
            </div>
            """, unsafe_allow_html=True)

        # Impact + Timeline
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">💥 Estimated Impact</div>
                <div class="brief-content">{brief.get('estimated_impact', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with ic2:
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">⏱️ Response Timeline</div>
                <div class="brief-content">{brief.get('timeline', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Export brief as JSON
        st.markdown("---")
        brief_json = json.dumps(brief, indent=2)
        st.download_button(
            label     = "⬇️ Download Brief as JSON",
            data      = brief_json,
            file_name = "operations_brief.json",
            mime      = "application/json",
        )

    # ==================================================================
    # TAB 5 — ML Explainability
    # ==================================================================
    with tab5:
        st.markdown('<div class="section-header">ML Explainability — Why Is This City at Delay Risk?</div>', unsafe_allow_html=True)

        # ── What this tab does (always visible) ──────────────────────────
        st.markdown("""
        <div class="risk-card" style="border-left: 3px solid #6366f1; padding: 1rem 1.5rem; margin-bottom:1.2rem;">
            <div style="font-weight:700; font-size:1rem; color:#e2e8f0; margin-bottom:0.5rem;">
                What is ML Explainability?
            </div>
            <div style="color:#94a3b8; font-size:0.88rem; line-height:1.8;">
                Our XGBoost model predicts the probability that shipments passing through each city will be delayed.
                But a prediction alone is not enough — <b style="color:#e2e8f0;">you need to know WHY</b>.<br><br>
                This tab uses <b style="color:#a5b4fc;">SHAP (SHapley Additive exPlanations)</b> — a Nobel-prize-winning
                game-theory method — to decompose each prediction into individual feature contributions.
                Every bar in the charts below answers: <i>"How much did this factor push the delay risk up or down?"</i><br><br>
                <span style="color:#f97316;">Red bars</span> = factors increasing delay risk &nbsp;|&nbsp;
                <span style="color:#22c55e;">Green bars</span> = factors protecting against delay
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not shap_results:
            st.markdown("""
            <div class="brief-section" style="text-align:center; padding: 2.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔍</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">
                    SHAP Library Not Available
                </div>
                <div style="color:#64748b; font-size:0.9rem; max-width:500px; margin:0 auto; line-height:1.6;">
                    Install <code style="background:#0f172a;padding:2px 6px;border-radius:4px;color:#a5b4fc;">shap</code>
                    to enable full ML explainability with feature-level breakdowns.<br><br>
                    The model currently uses <b style="color:#a5b4fc;">13 features</b> to predict shipment delay probability.
                    Below is the feature glossary.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            shap_node_ids = list(shap_results.keys())
            n_explained   = len(shap_results)

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
                <div style="background:#1e293b; border-radius:8px; padding:0.5rem 1rem;">
                    <span style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">Nodes Analysed</span>
                    <div style="color:#a5b4fc; font-weight:700; font-size:1.1rem;">{n_explained} at-risk cities</div>
                </div>
                <div style="background:#1e293b; border-radius:8px; padding:0.5rem 1rem;">
                    <span style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">Method</span>
                    <div style="color:#e2e8f0; font-weight:700; font-size:1.1rem;">XGBoost + SHAP TreeExplainer</div>
                </div>
                <div style="background:#1e293b; border-radius:8px; padding:0.5rem 1rem;">
                    <span style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">Features</span>
                    <div style="color:#e2e8f0; font-weight:700; font-size:1.1rem;">13 logistics signals</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Global Feature Importance ─────────────────────────────────
            st.markdown("#### 📊 Which Factors Matter Most Globally?")
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-bottom:0.5rem;'>"
                "This chart shows the <b style='color:#e2e8f0;'>average influence</b> of each factor across all "
                f"{n_explained} at-risk cities. A higher bar means that factor has a stronger effect on delay "
                "predictions across the entire supply network — it is the most important lever to monitor or fix."
                "</div>",
                unsafe_allow_html=True,
            )
            bar_fig = shap_bar_figure(shap_results)
            st.plotly_chart(bar_fig, width="stretch")

            st.markdown("---")

            # ── Per-Node Waterfall ────────────────────────────────────────
            st.markdown("#### 🌊 Deep Dive: Why Is a Specific City at Risk?")
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-bottom:0.8rem;'>"
                "Select any at-risk city below to see a <b style='color:#e2e8f0;'>step-by-step breakdown</b> of what "
                "is driving its delay probability. Each step shows how one factor moves the risk score up "
                "<span style='color:#ef4444;'>&#9650; (red)</span> or down "
                "<span style='color:#22c55e;'>&#9660; (green)</span> from the starting baseline."
                "</div>",
                unsafe_allow_html=True,
            )

            # Build display names for the dropdown
            node_display = {}
            for nid in shap_node_ids:
                row = risk_df[risk_df["node"] == nid]
                if not row.empty:
                    city    = row.iloc[0]["city_name"]
                    country = row.iloc[0]["country"]
                    score   = row.iloc[0]["risk_score"]
                    level   = row.iloc[0]["risk_level"]
                    node_display[nid] = f"{city}, {country} — Risk Score: {score:.3f}  {level}"
                else:
                    node_display[nid] = nid

            selected_node = st.selectbox(
                "Select city to inspect:",
                options=shap_node_ids,
                format_func=lambda x: node_display.get(x, x),
                index=0,
                key="shap_node_selector",
            )

            sel_row  = risk_df[risk_df["node"] == selected_node]
            sel_name = sel_row.iloc[0]["city_name"] if not sel_row.empty else selected_node
            sel_country = sel_row.iloc[0]["country"] if not sel_row.empty else ""
            sel_score   = sel_row.iloc[0]["risk_score"] if not sel_row.empty else 0.0

            waterfall_fig = shap_waterfall_figure(
                shap_results[selected_node],
                node_name=f"{sel_name}, {sel_country}",
            )
            st.plotly_chart(waterfall_fig, width="stretch")


            # ── Key driver cards ──────────────────────────────────────────
            node_shap      = shap_results[selected_node]
            sorted_drivers = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            top3           = sorted_drivers[:3]

            st.markdown(f"##### Top 3 Factors Driving Delay Risk for {sel_name}")
            driver_cols = st.columns(3)
            for i, (feat, val) in enumerate(top3):
                label = FEATURE_LABELS.get(feat, feat)
                desc  = FEATURE_DESCRIPTIONS.get(feat, "")
                short_desc = desc.split(".")[0]
                direction = "Increases Delay Risk" if val > 0 else "Protects Against Delay"
                col_hex   = "#ef4444" if val > 0 else "#22c55e"
                arrow     = "↑" if val > 0 else "↓"
                with driver_cols[i]:
                    st.markdown(f"""
                    <div class="risk-card" style="text-align:center; padding:1.1rem; min-height:160px;">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Factor #{i+1}</div>
                        <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.5rem;">{label}</div>
                        <div style="font-size:1.4rem;font-weight:700;color:{col_hex};">{arrow} {abs(val):.3f}</div>
                        <div style="font-size:0.75rem;color:{col_hex};margin-top:0.25rem;font-weight:600;">{direction}</div>
                        <div style="font-size:0.72rem;color:#64748b;margin-top:0.5rem;line-height:1.4;">{short_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Plain-English explanation ─────────────────────────────────
            st.markdown(f"#### What This Means for {sel_name}")
            plain_text = shap_to_text(node_shap, sel_name, top_k=6)

            risk_drivers_exist    = any(v > 0 for v in node_shap.values())
            protect_drivers_exist = any(v <= 0 for v in node_shap.values())

            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">
                    Delay Risk Analysis — <b style='color:#a5b4fc;'>{sel_name}, {sel_country}</b>
                    &nbsp; | &nbsp; Predicted Delay Probability: <b style='color:#f97316;'>{sel_score:.1%}</b>
                </div>
                <div class="brief-content" style="font-size:0.87rem; line-height:2.0; white-space:pre-line;">{plain_text}</div>
                <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid #1e293b; color:#64748b; font-size:0.78rem;">
                    These values are computed by the SHAP TreeExplainer running on our XGBoost delay prediction model.
                    Each value represents how much that single factor shifts the delay probability away from the average baseline (0.5).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Glossary ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Feature Glossary — Full explanation of all 13 model inputs", expanded=False):
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.83rem;margin-bottom:0.8rem;'>"
                "These are the 13 signals our XGBoost model uses to predict whether a shipment will be delayed. "
                "SHAP tells us how much each signal contributed to a specific prediction."
                "</div>", unsafe_allow_html=True
            )
            for feat, label in FEATURE_LABELS.items():
                desc = FEATURE_DESCRIPTIONS.get(feat, "")
                st.markdown(f"""
                <div style="padding:0.6rem 0; border-bottom:1px solid #1e293b;">
                    <span style="font-weight:600;color:#a5b4fc;">{label}</span><br>
                    <span style="color:#94a3b8;font-size:0.84rem;line-height:1.6;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

    # ==================================================================
    # TAB 6 — Anomaly Detection
    # ==================================================================
    with tab6:
        st.markdown('<div class="section-header">Anomaly Detection — Unusual Shipment Patterns</div>', unsafe_allow_html=True)

        anomaly_df = results.get("anomaly_df", pd.DataFrame())

        if anomaly_df.empty:
            st.info("Anomaly detection not available.")
        else:
            n_anomalous = anomaly_df["is_anomalous"].sum()
            cascade_nodes = set(results["cascade_result"].keys())

            # Banner
            st.markdown(f"""
            <div class="risk-card" style="display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
                <span style="font-size:1.5rem;">🚨</span>
                <span><b style="color:#e2e8f0;">{n_anomalous}</b> / {len(anomaly_df)} nodes show anomalous shipment patterns</span>
                <span style="color:#94a3b8;font-size:0.85rem;">Isolation Forest — lower score = more unusual behaviour</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # How many anomalies overlap with cascade
            overlap = anomaly_df[anomaly_df["is_anomalous"] & anomaly_df["node"].isin(cascade_nodes)]
            if not overlap.empty:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#450a0a,#7f1d1d);
                            border:1px solid #ef4444;border-radius:10px;
                            padding:0.9rem 1.2rem;margin-bottom:1rem;">
                    <span style="color:#fca5a5;font-weight:700;">
                        ⚠️ {len(overlap)} disrupted node(s) ALSO show anomalous patterns — elevated pre-disruption signal
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Bar chart
            st.plotly_chart(anomaly_bar_figure(anomaly_df, top_n=20), width="stretch")

            # Table
            st.markdown("#### Anomalous Nodes Detail")
            display_df = anomaly_df[anomaly_df["is_anomalous"]][
                ["city_name", "country", "product", "anomaly_score", "anomaly_z", "anomaly_level"]
            ].rename(columns={
                "city_name":     "City",
                "country":       "Country",
                "product":       "Sector",
                "anomaly_score": "Anomaly Score",
                "anomaly_z":     "Z-Score",
                "anomaly_level": "Level",
            })
            st.dataframe(display_df, width="stretch", hide_index=True)

    # ==================================================================
    # TAB 7 — AI Agent
    # ==================================================================
    with tab7:
        st.markdown('<div class="section-header">AI Agent — Autonomous Supply Chain Decision-Maker</div>', unsafe_allow_html=True)

        agent_result = results.get("agent_result")

        if agent_result is None:
            st.info("Agent result not available — re-run the analysis to activate the agent.")
        else:
            action_log      = agent_result.get("action_log", [])
            final_plan      = agent_result.get("final_plan", {})
            approved_routes = agent_result.get("approved_reroutes", [])
            flagged         = agent_result.get("flagged_nodes", [])
            elapsed         = agent_result.get("elapsed_seconds", 0)
            steps_taken     = agent_result.get("steps_taken", 0)
            source          = agent_result.get("source", "template-agent")

            # ── Status banner ───────────────────────────────────────────
            if source == "groq-agent":
                source_label = "🤖 Groq Autonomous Agent (GPT-OSS 120B)"
            elif source == "gemini-enhanced":
                source_label = "🤖 SupplAI Autonomous Agent (Gemini-Enhanced Reasoning)"
            elif source in ("gemini-unavailable", "groq-unavailable"):
                source_label = "📋 SupplAI Autonomous Agent (AI reasoning temporarily unavailable)"
            else:
                source_label = "🤖 SupplAI Autonomous Agent"
            plan_risk = final_plan.get("risk_level", "Unknown")
            risk_colour = {
                "Critical": "#ef4444", "High": "#f97316",
                "Medium": "#eab308",   "Low": "#22c55e",
            }.get(plan_risk, "#94a3b8")

            st.markdown(f"""
            <div class="risk-card" style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
                <span style="font-size:1.5rem;">🧠</span>
                <div>
                    <div style="font-weight:700; font-size:1rem; color:#e2e8f0;">{source_label}</div>
                    <div style="color:#64748b; font-size:0.82rem; margin-top:0.2rem;">
                        Completed <b style="color:#a5b4fc;">{steps_taken} tool calls</b> in {elapsed}s
                        &nbsp;·&nbsp; Approved <b style="color:#22c55e;">{len(approved_routes)} reroute(s)</b>
                        &nbsp;·&nbsp; Flagged <b style="color:#ef4444;">{len(flagged)} supplier(s)</b>
                    </div>
                </div>
                <span style="margin-left:auto; font-size:0.9rem; font-weight:600;
                             color:{risk_colour}; background:#1e293b;
                             padding:4px 14px; border-radius:8px;">
                    {plan_risk} Risk
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Reasoning Trace ─────────────────────────────────────────
            st.markdown("#### 🔍 Agent Reasoning Trace")
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Each step shows the agent's thought, which tool it called, and what the tool returned."
                "</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            TOOL_ICONS = {
                "get_top_risks":          "📊",
                "query_node_risk":        "🔎",
                "get_material_risks":     "📦",
                "get_anomaly_alerts":     "🚨",
                "find_alternate_route":   "🛤️",
                "approve_reroute":        "✅",
                "flag_critical_supplier": "🚩",
                "finalize_action_plan":   "📋",
            }

            for entry in action_log:
                tool    = entry.get("tool")
                thought = entry.get("thought", "")
                result  = entry.get("result") or {}
                args    = entry.get("args") or {}
                step_n  = entry.get("step", "?")
                icon    = TOOL_ICONS.get(tool, "🔧") if tool else "💬"

                if entry.get("type") == "conclusion":
                    st.markdown(f"""
                    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px;
                                padding:1rem 1.2rem; margin-bottom:0.8rem;">
                        <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase;
                                    letter-spacing:0.08em; margin-bottom:0.4rem;">
                            💬 Agent Conclusion
                        </div>
                        <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.6;">{thought}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    continue

                # Render thought bubble
                if thought:
                    st.markdown(f"""
                    <div style="border-left:3px solid #334155; padding:0.4rem 0.8rem;
                                margin-bottom:0.3rem; color:#94a3b8; font-size:0.85rem;
                                font-style:italic;">
                        🧠 {thought}
                    </div>
                    """, unsafe_allow_html=True)

                # Tool call card
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items()) if args else ""
                label    = f"{icon} <b>{tool}</b>({args_str})" if tool else ""

                # Summarise result for display
                result_lines = []
                if isinstance(result, dict):
                    for k, v in list(result.items())[:6]:
                        if isinstance(v, list):
                            result_lines.append(f"  {k}: [{len(v)} item(s)]")
                        else:
                            result_lines.append(f"  {k}: {v}")
                result_preview = "<br>".join(
                    f"<span style='color:#64748b;font-size:0.8rem;'>{l}</span>"
                    for l in result_lines
                )

                st.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e3a5f; border-radius:10px;
                            padding:0.9rem 1.2rem; margin-bottom:0.7rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:0.5rem;">
                        <span style="color:#e2e8f0; font-size:0.9rem;">{label}</span>
                        <span style="color:#475569; font-size:0.75rem;">Step {step_n}</span>
                    </div>
                    <div style="background:#070c16; border-radius:6px; padding:0.5rem 0.8rem;
                                font-family:monospace;">
                        {result_preview}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Final Action Plan ────────────────────────────────────────
            st.markdown("#### 📋 Final Action Plan")

            plan_summary = final_plan.get("summary", "")
            if plan_summary:
                st.markdown(f"""
                <div class="brief-section">
                    <div class="brief-title">🎯 Agent Assessment</div>
                    <div class="brief-content">{plan_summary}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_plan, col_meta = st.columns([3, 1])
            with col_plan:
                actions = final_plan.get("priority_actions", [])
                if actions:
                    actions_html = "".join(
                        f"<div style='padding:0.45rem 0; border-bottom:1px solid #1e293b; "
                        f"color:#cbd5e1; font-size:0.9rem;'>"
                        f"<span style='color:#22c55e; font-weight:700;'>{i+1}.</span> {a}</div>"
                        for i, a in enumerate(actions)
                    )
                    st.markdown(f"""
                    <div class="brief-section">
                        <div class="brief-title">🚀 Autonomous Decisions Made</div>
                        {actions_html}
                    </div>
                    """, unsafe_allow_html=True)

            with col_meta:
                rec_days = final_plan.get("estimated_recovery_days", "?")
                st.markdown(f"""
                <div class="risk-card" style="text-align:center; padding:1.2rem;">
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.08em;">Risk Level</div>
                    <div style="font-size:1.6rem; font-weight:700;
                                color:{risk_colour}; margin:0.4rem 0;">{plan_risk}</div>
                    <hr style="border-color:#1e293b; margin:0.6rem 0;">
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.08em;">Est. Recovery</div>
                    <div style="font-size:1.4rem; font-weight:700;
                                color:#a5b4fc; margin-top:0.4rem;">{rec_days} days</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Approved Reroutes ────────────────────────────────────────
            st.markdown(f"#### ✅ Agent-Approved Reroutes ({len(approved_routes)})")
            if not approved_routes:
                st.info("No reroutes were approved by the agent for this event.")
            else:
                for i, r in enumerate(approved_routes):
                    st.markdown(f"""
                    <div style="background:#0a2218; border:1px solid #166534; border-radius:10px;
                                padding:0.9rem 1.2rem; margin-bottom:0.6rem;">
                        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
                            <div>
                                <span style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">
                                    Approved Route {i+1}
                                </span>
                                <div style="font-weight:700; color:#e2e8f0; margin-top:0.2rem;">
                                    📍 {r['source']} → 📍 {r['destination']}
                                </div>
                            </div>
                            <span style="color:#22c55e; font-size:0.85rem; font-weight:600;">✅ Approved</span>
                        </div>
                        <div style="margin-top:0.5rem; color:#94a3b8; font-size:0.82rem;">
                            {r.get('reason', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Flagged Suppliers ────────────────────────────────────────
            st.markdown(f"#### 🚩 Flagged Critical Suppliers ({len(flagged)})")
            if not flagged:
                st.info("No suppliers were flagged by the agent.")
            else:
                flag_cols = st.columns(min(len(flagged), 3))
                for i, f in enumerate(flagged[:3]):
                    priority_colour = {"high": "#ef4444", "medium": "#f97316", "low": "#eab308"}.get(
                        f.get("priority", "high"), "#ef4444"
                    )
                    with flag_cols[i % 3]:
                        st.markdown(f"""
                        <div class="risk-card" style="text-align:center; padding:1rem;">
                            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                        letter-spacing:0.08em;margin-bottom:0.3rem;">
                                {f.get('priority','high').upper()} PRIORITY
                            </div>
                            <div style="font-size:1.5rem; margin:0.3rem 0;">🚩</div>
                            <div style="font-weight:700; color:#e2e8f0; font-size:0.95rem;
                                        margin-bottom:0.3rem;">{f.get('city', f.get('node_id','?'))}</div>
                            <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">
                                {f.get('reason','')[:120]}
                            </div>
                            <div style="margin-top:0.5rem; display:inline-block;
                                        background:{priority_colour}22; color:{priority_colour};
                                        border-radius:6px; padding:2px 10px; font-size:0.75rem;
                                        font-weight:600;">
                                {f.get('priority','high').title()} Priority
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Human-in-the-Loop Gate ───────────────────────────────────
            st.markdown("#### 👤 Human Approval Gate")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e1a0a,#2d2600);
                        border:1px solid #ca8a04; border-radius:10px;
                        padding:1rem 1.4rem; margin-bottom:1rem;">
                <div style="color:#fef08a; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">
                    ⚠️ Agent has made {len(approved_routes) + len(flagged)} autonomous decision(s)
                </div>
                <div style="color:#fde68a; font-size:0.85rem; line-height:1.6;">
                    The AI agent has independently assessed the disruption, approved alternate routes,
                    and flagged critical suppliers. Review the reasoning trace above and confirm
                    to commit these actions.
                </div>
            </div>
            """, unsafe_allow_html=True)

            approval_key = f"agent_approved_{id(agent_result)}"
            if st.session_state.get(approval_key):
                st.success(
                    f"✅ **All {len(approved_routes) + len(flagged)} agent decisions approved** "
                    f"— Operations team has been notified. Routes are being activated."
                )
                if st.button("🔄 Reset Approval", key="reset_approval"):
                    st.session_state[approval_key] = False
                    st.rerun()
            else:
                col_approve, col_reject = st.columns(2)
                with col_approve:
                    if st.button(
                        f"✅ APPROVE ALL AGENT DECISIONS ({len(approved_routes) + len(flagged)})",
                        type="primary",
                        width="stretch",
                        key="approve_agent",
                    ):
                        st.session_state[approval_key] = True
                        st.rerun()
                with col_reject:
                    if st.button(
                        "❌ REJECT & RESET",
                        width="stretch",
                        key="reject_agent",
                    ):
                        st.warning("Agent decisions rejected. Re-run analysis to generate a new plan.")

    # Premium footer
    render_footer()


def render_footer():
    """Render the footer at the bottom of the main area."""
    st.markdown("""
    <div style="margin-top:3rem; padding:1.2rem 1.5rem;
                background:linear-gradient(135deg,rgba(10,14,26,0.95),rgba(6,11,24,0.98));
                border:1px solid rgba(66,133,244,0.12); border-radius:16px;
                display:flex; align-items:center; justify-content:space-between;
                flex-wrap:wrap; gap:1rem; backdrop-filter:blur(12px);">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.4rem;">🔗</span>
            <div>
                <div style="font-weight:700; font-size:0.92rem;
                            background:linear-gradient(90deg,#4285f4,#34a853,#fbbc04,#ea4335);
                            background-size:200% auto;
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                            animation:shimmer 4s linear infinite;
                            font-family:'Space Grotesk',sans-serif;">SupplAI</div>
                <div style="color:#334155; font-size:0.72rem; margin-top:1px;">
                    AI Supply Chain Disruption Intelligence
                </div>
            </div>
        </div>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
            <span style="background:rgba(66,133,244,0.1); border:1px solid rgba(66,133,244,0.2);
                         border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#4285f4;">
                Google Gemini
            </span>
            <span style="background:rgba(52,168,83,0.1); border:1px solid rgba(52,168,83,0.2);
                         border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#34a853;">
                NetworkX
            </span>
            <span style="background:rgba(251,188,4,0.1); border:1px solid rgba(251,188,4,0.2);
                         border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#fbbc04;">
                XGBoost
            </span>
            <span style="background:rgba(234,67,53,0.1); border:1px solid rgba(234,67,53,0.2);
                         border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#ea4335;">
                SHAP
            </span>
            <span style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.2);
                         border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#8b5cf6;">
                Plotly
            </span>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.72rem; color:#475569; margin-bottom:2px;">
                SupplAI — AI Supply Chain Intelligence
            </div>
            <div style="font-size:0.68rem; color:#1e293b;">
                © 2025
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
