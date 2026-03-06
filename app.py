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

    .sidebar-header {
        display: flex; align-items: center; gap: 10px;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .shield-icon {
        width: 34px; height: 34px; flex-shrink: 0;
        background: linear-gradient(135deg, #0EA5E9, #38BDF8);
        border-radius: 8px; display: flex; align-items: center;
        justify-content: center; font-size: 16px;
        box-shadow: 0 0 16px rgba(56,189,248,0.25);
    }
    .brand-name {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem; font-weight: 700;
        color: var(--text-muted); letter-spacing: 0.18em;
        text-transform: uppercase; line-height: 1.4;
    }
    .brand-name span { color: var(--accent); display: block; font-size: 0.8rem; }

    .sidebar-section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem; font-weight: 700;
        color: var(--text-muted); letter-spacing: 0.2em;
        text-transform: uppercase; margin-bottom: 0.75rem;
    }

    /* ── Models pill (replaces oversized stat card) ── */
    .models-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 0.5rem 0.85rem;
        margin: 0.5rem 0 1.25rem 0;
    }
    .models-pill .pill-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem; font-weight: 700;
        color: var(--accent); line-height: 1;
    }
    .models-pill .pill-label {
        font-size: 0.7rem; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.08em;
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
        text-align: center; padding: 2.75rem 2rem 2rem;
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

    /* ── Step indicator ── */
    .step-row {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin: 1.5rem auto 2rem auto; max-width: 480px;
    }
    .step-item {
        display: flex; flex-direction: column; align-items: center;
        gap: 5px; flex: 1;
    }
    .step-dot {
        width: 28px; height: 28px; border-radius: 50%;
        border: 1.5px solid var(--border-bright);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Space Mono', monospace; font-size: 0.6rem;
        color: var(--text-muted); background: var(--surface);
    }
    .step-dot.active {
        border-color: var(--accent); color: var(--accent);
        box-shadow: 0 0 10px rgba(56,189,248,0.3);
    }
    .step-dot.done {
        border-color: var(--success); color: var(--success);
        background: rgba(52,211,153,0.08);
        box-shadow: 0 0 8px rgba(52,211,153,0.2);
    }
    .step-label {
        font-family: 'Space Mono', monospace; font-size: 0.52rem;
        color: var(--text-muted); letter-spacing: 0.1em;
        text-transform: uppercase; text-align: center;
    }
    .step-label.active { color: var(--accent); }
    .step-label.done   { color: var(--success); }
    .step-connector {
        height: 1px; flex: 1; max-width: 60px;
        background: var(--border); margin-bottom: 18px;
    }
    .step-connector.done { background: var(--success); opacity: 0.5; }

    /* ── Upload zone ── */
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

    /* ── File info bar (replaces vague Streamlit warnings) ── */
    .file-info-bar {
        display: flex; align-items: center; gap: 10px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 0.65rem 1rem; margin: 0.75rem 0;
    }
    .file-info-bar .fi-icon { font-size: 1rem; flex-shrink: 0; }
    .file-info-bar .fi-name {
        font-family: 'Space Mono', monospace; font-size: 0.68rem;
        color: var(--text-secondary); flex: 1;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .file-info-bar .fi-size {
        font-family: 'Space Mono', monospace; font-size: 0.62rem; flex-shrink: 0;
    }

    /* ── Streamlit warning/info banners ── */
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
        text-align: center; box-shadow: var(--glow-success);
        position: relative; overflow: hidden;
    }
    .result-safe::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--success), transparent);
    }
    .result-safe .verdict-icon { font-size: 2.2rem; margin-bottom: 0.5rem; display: block; }
    .result-safe .verdict {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: var(--success); letter-spacing: 0.05em;
    }
    .result-safe .verdict-sub {
        color: var(--text-secondary); font-size: 0.9rem;
        margin-top: 0.4rem; font-weight: 300;
    }

    .result-malware {
        background: var(--error-dim);
        border: 1px solid rgba(248, 113, 113, 0.3);
        border-radius: 14px; padding: 2.25rem 2rem;
        text-align: center; box-shadow: var(--glow-error);
        position: relative; overflow: hidden;
    }
    .result-malware::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--error), transparent);
    }
    .result-malware .verdict-icon { font-size: 2.2rem; margin-bottom: 0.5rem; display: block; }
    .result-malware .verdict {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: var(--error); letter-spacing: 0.05em;
    }
    .result-malware .verdict-sub {
        color: var(--text-secondary); font-size: 0.9rem;
        margin-top: 0.4rem; font-weight: 300;
    }

    /* ── Summary card ── */
    .summary-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 1rem 1.25rem;
        font-family: 'Space Mono', monospace; font-size: 0.7rem;
        color: var(--text-muted); line-height: 2;
    }
    .summary-card .row { display: flex; gap: 1rem; align-items: baseline; }
    .summary-card .row .key { color: var(--text-secondary); min-width: 52px; }
    .summary-card .row .val { color: var(--text-primary); word-break: break-all; }
    .summary-card .row .val.clean   { color: var(--success); }
    .summary-card .row .val.malware { color: var(--error); }

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

    /* ── Alerts ── */
    .stAlert { border-radius: 10px !important; }
    [data-baseweb="notification"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("step", 1), ("result", None), ("result_filename", ""), ("result_model", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Step renderer ─────────────────────────────────────────────────────────────
def render_steps():
    """
    Each step turns green independently when its condition is met:
      1. Model  → always green (model pre-selected on load)
      2. Upload → green once a file is uploaded
      3. Scan   → green once analysis has completed (step >= 4)
      4. Results→ green once results are stored in session state
    """
    s = st.session_state
    done = {
        1: True,                              # Model always done
        2: s.step >= 2,                       # File uploaded
        3: s.step >= 4,                       # Scan completed
        4: s.result is not None,             # Results ready
    }

    labels = ["Model", "Upload", "Scan", "Results"]
    html = '<div class="step-row">'
    for i, label in enumerate(labels):
        num = i + 1
        if done[num]:
            dot_cls, lbl_cls, inner = "done", "done", "✓"
        else:
            dot_cls, lbl_cls, inner = "", "", str(num)

        html += f"""
        <div class="step-item">
            <div class="step-dot {dot_cls}">{inner}</div>
            <div class="step-label {lbl_cls}">{label}</div>
        </div>"""

        if i < len(labels) - 1:
            conn_cls = "done" if done[num] and done[num + 1] else ""
            html += f'<div class="step-connector {conn_cls}"></div>'

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ── Results renderer ──────────────────────────────────────────────────────────
def render_results(prediction, confidence, filename, model_used):
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

    # Gauge color matches verdict
    bar_color   = "#F87171" if prediction == 1 else "#38BDF8"
    gauge_color = "#F87171" if prediction == 1 else "#34D399"
    gauge_steps = [
        {'range': [0,  50], 'color': 'rgba(255,255,255,0.03)'},
        {'range': [50, 75], 'color': 'rgba(255,255,255,0.05)'},
        {'range': [75,100], 'color': 'rgba(56,189,248,0.07)'},
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
            'bar': {'color': bar_color, 'thickness': 0.22},
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

    # Structured summary card with colored result
    verdict_label = "MALWARE" if prediction == 1 else "CLEAN"
    verdict_class = "malware" if prediction == 1 else "clean"
    st.markdown(f"""
    <div class="summary-card">
        <div class="row"><span class="key">FILE</span><span class="val">{filename}</span></div>
        <div class="row"><span class="key">MODEL</span><span class="val">{model_used}</span></div>
        <div class="row"><span class="key">RESULT</span><span class="val {verdict_class}">{verdict_label} · {confidence*100:.1f}% confidence</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-eyebrow">Static Analysis Engine</div>
    <h1 class="header-title">Android <span>Malware</span> Detection</h1>
    <p class="header-sub">Upload an APK to scan for malicious signatures using ensemble ML models</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
with st.spinner("Initializing detection engine..."):
    models, features = load_all_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="shield-icon">🛡️</div>
        <div class="brand-name">Android<span>MALWARE DETECTOR</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Detection Model</div>', unsafe_allow_html=True)
    model_names    = list(models.keys())
    selected_model = st.selectbox("", model_names, index=0 if model_names else None, label_visibility="collapsed")
    st.markdown(f'<div class="model-badge">{selected_model}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="models-pill">
        <span class="pill-value">{len(models)}</span>
        <span class="pill-label">Models loaded</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Available Models</div>', unsafe_allow_html=True)
    for name in model_names:
        active = "✦ " if name == selected_model else "· "
        color  = "var(--accent)" if name == selected_model else "var(--text-muted)"
        st.markdown(
            f'<p style="font-family:Space Mono,monospace;font-size:0.7rem;color:{color};margin:4px 0;">{active}{name}</p>',
            unsafe_allow_html=True
        )

# ── Main content ──────────────────────────────────────────────────────────────
_, center, _ = st.columns([1, 3, 1])
with center:

    render_steps()

    st.markdown('<div class="section-label">Upload APK File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", type=['apk'],
        help="Select an Android APK file to analyze (up to 700 MB)",
        label_visibility="collapsed"
    )

    # File removed — reset to step 1
    if uploaded_file is None:
        if st.session_state.step >= 2:
            st.session_state.step   = 1
            st.session_state.result = None
            st.rerun()

    if uploaded_file is not None:

        # Advance to step 2 on first upload
        if st.session_state.step < 2:
            st.session_state.step = 2
            st.rerun()

        file_size_mb = uploaded_file.size / (1024 * 1024)
        size_color   = "#F87171" if file_size_mb > 600 else "#FBBF24" if file_size_mb > 200 else "#34D399"
        size_icon    = "⚠" if file_size_mb > 200 else "✓"
        st.markdown(f"""
        <div class="file-info-bar">
            <span class="fi-icon">📦</span>
            <span class="fi-name">{uploaded_file.name}</span>
            <span class="fi-size" style="color:{size_color};">{size_icon} {file_size_mb:.1f} MB</span>
        </div>
        """, unsafe_allow_html=True)

        if file_size_mb > 600:
            st.warning(f"⚠ Very large file: {file_size_mb:.1f} MB — processing will take several minutes.")
        elif file_size_mb > 400:
            st.warning(f"⚠ Large file: {file_size_mb:.1f} MB — processing may take a few minutes.")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            tmp_path = tmp_file.name

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run Analysis", use_container_width=True):
            result = None
            error  = None

            # Advance to step 3 while scanning
            st.session_state.step   = 3
            st.session_state.result = None

            with st.spinner("Scanning APK — extracting features and running ensemble models..."):
                try:
                    feature_vector, matches = extract_features(tmp_path, features)
                    model      = models[selected_model]
                    prediction = model.predict(feature_vector)[0]

                    if hasattr(model, 'predict_proba'):
                        proba      = model.predict_proba(feature_vector)[0]
                        confidence = proba[prediction] if len(proba) > 1 else proba[0]
                    else:
                        confidence = 0.5 + 0.4 * (prediction - 0.5)

                    # Save result and advance to step 4
                    st.session_state.result          = {"prediction": int(prediction), "confidence": float(confidence)}
                    st.session_state.result_filename = uploaded_file.name
                    st.session_state.result_model    = selected_model
                    st.session_state.step            = 4

                except Exception as e:
                    st.session_state.step = 2  # back to upload on error
                    error = e
                finally:
                    utils.safe_file_cleanup(tmp_path)

            if error:
                st.error(f"Analysis failed: {error}")
            else:
                st.rerun()  # re-render so step indicator updates immediately

        # Show persisted results if on step 4
        if st.session_state.step == 4 and st.session_state.result:
            render_results(
                st.session_state.result["prediction"],
                st.session_state.result["confidence"],
                st.session_state.result_filename,
                st.session_state.result_model,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Android Malware Detection Engine
</div>
""", unsafe_allow_html=True)
