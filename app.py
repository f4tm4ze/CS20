import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import shutil
import tempfile

from model_loader import load_all_models
from feature_extractor import extract_features
import utils

def safe_bool_check(val):
    if isinstance(val, np.ndarray):
        return bool(val.any())
    return bool(val)

st.set_page_config(
    page_title="Android Malware Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg-primary: #0A0E1A;
        --bg-secondary: #0F1629;
        --surface: #131D35;
        --surface-raised: #1A2542;
        --border: rgba(56, 189, 248, 0.12);
        --border-bright: rgba(56, 189, 248, 0.35);
        --text-primary: #E2E8F0;
        --text-secondary: #94A3B8;
        --text-muted: #475569;
        --accent: #38BDF8;
        --accent-dim: rgba(56, 189, 248, 0.1);
        --success: #34D399;
        --success-dim: rgba(52, 211, 153, 0.08);
        --error: #F87171;
        --error-dim: rgba(248, 113, 113, 0.08);
        --warning: #FBBF24;
        --glow-accent: 0 0 20px rgba(56, 189, 248, 0.15);
        --glow-success: 0 0 30px rgba(52, 211, 153, 0.2);
        --glow-error: 0 0 30px rgba(248, 113, 113, 0.2);
    }

    /* ── Global reset ── */
    html, body, .stApp {
        background-color: var(--bg-primary) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div { padding: 1.5rem 1.25rem; }

    .sidebar-logo {
        display: flex; align-items: center; gap: 10px;
        padding: 0.75rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.75rem;
    }
    .sidebar-logo .shield {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, var(--accent), #0EA5E9);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; box-shadow: var(--glow-accent);
    }
    .sidebar-logo .brand {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem; font-weight: 700;
        color: var(--text-secondary); letter-spacing: 0.15em;
        text-transform: uppercase; line-height: 1.3;
    }
    .sidebar-logo .brand span { color: var(--accent); display: block; font-size: 0.85rem; }

    .sidebar-section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem; font-weight: 700;
        color: var(--text-muted); letter-spacing: 0.2em;
        text-transform: uppercase; margin-bottom: 0.75rem;
    }

    .stat-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 0.75rem; margin: 0.75rem 0 1.5rem 0;
    }
    .stat-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.85rem 1rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem; font-weight: 700;
        color: var(--accent); line-height: 1;
    }
    .stat-card .stat-label {
        font-size: 0.68rem; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-top: 4px;
    }

    .model-badge {
        background: var(--surface); border: 1px solid var(--border-bright);
        border-radius: 8px; padding: 0.6rem 1rem;
        font-family: 'Space Mono', monospace; font-size: 0.72rem;
        color: var(--accent); margin-top: 0.5rem;
        display: flex; align-items: center; gap: 8px;
    }
    .model-badge::before {
        content: ''; width: 7px; height: 7px;
        background: var(--success); border-radius: 50%;
        box-shadow: 0 0 6px var(--success); flex-shrink: 0;
    }

    /* ── Main header ── */
    .main-header {
        text-align: center; padding: 3rem 2rem 2.5rem;
        position: relative; overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute; top: 0; left: 50%; transform: translateX(-50%);
        width: 400px; height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }
    .header-eyebrow {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem; font-weight: 700;
        color: var(--accent); letter-spacing: 0.25em;
        text-transform: uppercase; margin-bottom: 1rem;
    }
    .header-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.6rem; font-weight: 600;
        color: var(--text-primary); margin: 0; line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .header-title span { color: var(--accent); }
    .header-sub {
        font-size: 0.95rem; color: var(--text-secondary);
        margin-top: 0.75rem; font-weight: 300;
    }

    /* ── Upload zone ── */
    .upload-section {
        max-width: 680px; margin: 0 auto 2rem auto;
    }
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem; letter-spacing: 0.2em;
        text-transform: uppercase; color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border-bright) !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        box-shadow: var(--glow-accent) !important;
    }
    [data-testid="stFileUploaderDropzoneInput"] { cursor: pointer; }

    /* ── File size info banners ── */
    .stInfo, .stWarning {
        background: var(--surface-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Analyze button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9, #38BDF8) !important;
        color: #0A0E1A !important; font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important; font-weight: 700 !important;
        letter-spacing: 0.12em !important; text-transform: uppercase !important;
        border: none !important; border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.45) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Results ── */
    .results-wrapper {
        max-width: 680px; margin: 0 auto;
    }
    .results-heading {
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem; letter-spacing: 0.2em;
        text-transform: uppercase; color: var(--text-muted);
        text-align: center; margin-bottom: 1.25rem;
    }

    .result-safe {
        background: var(--success-dim);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 14px; padding: 2.25rem 2rem;
        text-align: center;
        box-shadow: var(--glow-success);
        position: relative; overflow: hidden;
    }
    .result-safe::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--success), transparent);
    }
    .result-safe .verdict {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: var(--success); letter-spacing: 0.05em;
    }
    .result-safe .verdict-sub {
        color: var(--text-secondary); font-size: 0.9rem;
        margin-top: 0.4rem; font-weight: 300;
    }
    .result-safe .verdict-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }

    .result-malware {
        background: var(--error-dim);
        border: 1px solid rgba(248, 113, 113, 0.3);
        border-radius: 14px; padding: 2.25rem 2rem;
        text-align: center;
        box-shadow: var(--glow-error);
        position: relative; overflow: hidden;
    }
    .result-malware::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--error), transparent);
    }
    .result-malware .verdict {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: var(--error); letter-spacing: 0.05em;
    }
    .result-malware .verdict-sub {
        color: var(--text-secondary); font-size: 0.9rem;
        margin-top: 0.4rem; font-weight: 300;
    }
    .result-malware .verdict-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] { background: transparent !important; }
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: var(--accent) !important; font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.65rem !important; text-transform: uppercase;
        letter-spacing: 0.08em !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

    /* ── Footer ── */
    .footer-text {
        text-align: center; color: var(--text-muted);
        padding: 2.5rem 0 1rem; font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 0.08em;
        border-top: 1px solid var(--border); margin-top: 3rem;
    }

    /* ── Error/success alerts ── */
    .stAlert { border-radius: 10px !important; }
    [data-baseweb="notification"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-eyebrow">Static Analysis Engine</div>
    <h1 class="header-title">Android <span>Malware</span> Detection</h1>
    <p class="header-sub">Upload an APK to scan for malicious signatures using ensemble ML models</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ────────────────────────────────────────────────────────────
with st.spinner("Initializing detection engine..."):
    models, features = load_all_models()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Detection Model</div>', unsafe_allow_html=True)
    model_names = list(models.keys())
    selected_model = st.selectbox("", model_names, index=0 if model_names else None, label_visibility="collapsed")
    st.markdown(f'<div class="model-badge">{selected_model}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{len(models)}</div>
            <div class="stat-label">Models</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(features)}</div>
            <div class="stat-label">Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Available Models</div>', unsafe_allow_html=True)
    for name in model_names:
        active = "✦ " if name == selected_model else "· "
        color = "var(--accent)" if name == selected_model else "var(--text-muted)"
        st.markdown(f'<p style="font-family:Space Mono,monospace;font-size:0.7rem;color:{color};margin:4px 0;">{active}{name}</p>', unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
_, center, _ = st.columns([1, 3, 1])
with center:
    st.markdown('<div class="section-label">Upload APK File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", type=['apk'],
        help="Select an Android APK file to analyze (up to 700 MB)",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 600:
            st.warning(f"⚠ Very large file: {file_size_mb:.1f} MB — processing will take several minutes.")
        elif file_size_mb > 400:
            st.warning(f"⚠ Large file: {file_size_mb:.1f} MB — processing may take a few minutes.")
        elif file_size_mb > 200:
            st.info(f"ℹ Medium file: {file_size_mb:.1f} MB — processing may take a moment.")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            tmp_path = tmp_file.name

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⟶  Run Analysis", use_container_width=True):
            result = None
            error = None

            with st.spinner("Scanning APK — extracting features and running ensemble models..."):
                try:
                    feature_vector, matches = extract_features(tmp_path, features)
                    model = models[selected_model]
                    prediction = model.predict(feature_vector)[0]

                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(feature_vector)[0]
                        confidence = proba[prediction] if len(proba) > 1 else proba[0]
                    else:
                        confidence = 0.5 + 0.4 * (prediction - 0.5)

                    result = {"prediction": int(prediction), "confidence": float(confidence)}
                except Exception as e:
                    error = e
                finally:
                    utils.safe_file_cleanup(tmp_path)

            if error:
                st.error(f"Analysis failed: {error}")
            elif result:
                prediction = result["prediction"]
                confidence = result["confidence"]

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="results-heading">Analysis Results</div>', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown("""
                    <div class="result-malware">
                        <div class="verdict">MALWARE DETECTED</div>
                        <div class="verdict-sub">Threat level: HIGH — Do not install this APK</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-safe">
                        <div class="verdict">CLEAN</div>
                        <div class="verdict-sub">No malicious signatures detected</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence gauge — dark themed
                gauge_color  = "#F87171" if prediction == 1 else "#34D399"
                gauge_steps  = [
                    {'range': [0, 50],   'color': 'rgba(255,255,255,0.03)'},
                    {'range': [50, 75],  'color': 'rgba(255,255,255,0.05)'},
                    {'range': [75, 100], 'color': 'rgba(56,189,248,0.07)'}
                ]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={'suffix': '%', 'font': {'color': '#E2E8F0', 'size': 42, 'family': 'Space Mono'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Score", 'font': {'color': '#94A3B8', 'size': 13, 'family': 'DM Sans'}},
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickcolor': '#475569',
                            'tickfont': {'color': '#475569', 'size': 10, 'family': 'Space Mono'},
                        },
                        'bar': {'color': "#38BDF8", 'thickness': 0.22},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': gauge_steps,
                        'threshold': {
                            'line': {'color': gauge_color, 'width': 3},
                            'thickness': 0.85,
                            'value': confidence * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=50, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#E2E8F0', 'family': 'DM Sans'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Model + filename summary
                st.markdown(f"""
                <div style="
                    background: var(--surface); border: 1px solid var(--border);
                    border-radius: 10px; padding: 1rem 1.25rem;
                    font-family: Space Mono, monospace; font-size: 0.7rem;
                    color: var(--text-muted); line-height: 1.9;
                ">
                    <span style="color:var(--text-secondary);">FILE</span>&nbsp;&nbsp;&nbsp;{uploaded_file.name}<br>
                    <span style="color:var(--text-secondary);">MODEL</span>&nbsp;&nbsp;{selected_model}<br>
                    <span style="color:var(--text-secondary);">RESULT</span>&nbsp;{"MALWARE" if prediction == 1 else "CLEAN"} · {confidence*100:.1f}% confidence
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Android Malware Detection Engine
</div>
""", unsafe_allow_html=True)
