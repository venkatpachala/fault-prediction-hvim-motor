"""
MOTOR FAULT DIAGNOSIS DASHBOARD
================================
Dual prediction: ML (Random Forest) + Fuzzy Logic (Mamdani)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Motor Fault Diagnosis",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# LOAD ML MODEL
# ============================================================
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('ml_model.pkl')
        features = joblib.load('features.pkl')
        return model, features
    except Exception as e:
        return None, None

# ============================================================
# LOAD FUZZY SYSTEM
# ============================================================
@st.cache_resource
def load_fuzzy_system():
    try:
        from fuzzy_diagnosis import create_fuzzy_system, predict_fuzzy
        system, _, _, _, _, _ = create_fuzzy_system()
        return system, predict_fuzzy
    except Exception as e:
        st.sidebar.warning(f"Fuzzy system not loaded: {e}")
        return None, None

# ============================================================
# FEATURE EXTRACTION (same 19 features as training)
# ============================================================
def extract_features(df):
    """Extract 19 features matching extract_features.m / retrain_from_csvs.py"""
    ia = df['ia'].values.astype(float)
    ib = df['ib'].values.astype(float)
    ic = df['ic'].values.astype(float)
    Te = df['Te'].values.astype(float)
    wm = df['wm'].values.astype(float)

    # Phase A current features
    ia_rms = np.sqrt(np.mean(ia**2))
    ia_std = np.std(ia)
    n = len(ia)
    ia_mean = np.mean(ia)
    m2 = np.mean((ia - ia_mean)**2)
    m4 = np.mean((ia - ia_mean)**4)
    ia_kurt = (m4 / (m2**2 + 1e-12)) - 3
    ia_crest = np.max(np.abs(ia)) / (ia_rms + 1e-9)

    Y = np.abs(np.fft.fft(ia))
    Yh = Y[:len(Y)//2]
    ia_domFreq = float(np.argmax(Yh))

    # Cross-phase features
    ib_rms = np.sqrt(np.mean(ib**2))
    ic_rms = np.sqrt(np.mean(ic**2))
    current_unbalance = np.std([ib_rms, ic_rms])

    # Torque features
    Te_mean = np.mean(Te)
    Te_std = np.std(Te)
    Te_p2p = np.max(Te) - np.min(Te)

    # Speed features
    wm_mean = np.mean(wm)
    wm_std = np.std(wm)
    wm_p2p = np.max(wm) - np.min(wm)

    # Global features
    energy = np.sum(ia**2)
    fft_energy = np.sum(Y**2)
    torque_ripple = Te_std
    speed_ripple = wm_std
    thd_like = np.sum(Yh[4:]**2) / (np.sum(Yh**2) + 1e-9)

    feature_values = [
        ia_rms, ia_std, ia_kurt, ia_crest, ia_domFreq,
        ib_rms, ic_rms, current_unbalance,
        Te_mean, Te_std, Te_p2p,
        wm_mean, wm_std, wm_p2p,
        energy, fft_energy,
        torque_ripple, speed_ripple, thd_like
    ]

    feature_names = [
        'ia_rms', 'ia_std', 'ia_kurt', 'ia_crest', 'ia_domFreq',
        'ib_rms', 'ic_rms', 'current_unbalance',
        'Te_mean', 'Te_std', 'Te_p2p',
        'wm_mean', 'wm_std', 'wm_p2p',
        'energy', 'fft_energy',
        'torque_ripple', 'speed_ripple', 'thd_like'
    ]

    feat_dict = dict(zip(feature_names, feature_values))
    feature_row = np.array([feature_values])

    return feature_row, feat_dict

# ============================================================
# FAULT CLASS CONFIG
# ============================================================
CLASS_NAMES = ['Healthy', 'Voltage Unbalance', 'Rotor Fault', 'Stator Fault']
CLASS_COLORS = ['#28a745', '#ffc107', '#6f42c1', '#dc3545']
CLASS_ICONS = ['🟢', '🟡', '🟣', '🔴']

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚡ Motor Fault Diagnosis")
st.sidebar.markdown("**Dual AI System**: ML + Fuzzy Logic")
st.sidebar.markdown("---")

# Load models
ml_model, ml_features = load_ml_model()
fuzzy_system, fuzzy_predict = load_fuzzy_system()

st.sidebar.markdown("### Model Status")
if ml_model is not None:
    st.sidebar.success(f"✅ ML Model: {len(ml_features)} features")
else:
    st.sidebar.error("❌ ML Model not loaded")

if fuzzy_system is not None:
    st.sidebar.success("✅ Fuzzy Logic: 4 inputs, 11 rules")
else:
    st.sidebar.error("❌ Fuzzy system not loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.info(
    "1. Upload a motor CSV file\n"
    "2. View signal plots\n"
    "3. Check ML and Fuzzy predictions\n"
    "4. Compare both approaches"
)

# ============================================================
# MAIN AREA
# ============================================================
st.title("⚡ Induction Motor Fault Diagnosis")
st.markdown("**Upload motor data → Get dual AI diagnosis (ML + Fuzzy Logic)**")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "📂 Upload Motor Data (CSV)", type=['csv'],
    help="CSV with columns: t, ia, ib, ic, Te, wm"
)

if uploaded_file is not None:
    # ========================================
    # LOAD CSV
    # ========================================
    with st.spinner('📖 Reading data...'):
        try:
            df_test = pd.read_csv(uploaded_file, nrows=2)
            uploaded_file.seek(0)

            has_headers = any(isinstance(c, str) and not c.replace('.', '').replace('-', '').isnumeric()
                             for c in df_test.columns)

            if has_headers:
                df = pd.read_csv(uploaded_file)
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None)

            df.columns = [str(c).strip() for c in df.columns]

            if 't' in df.columns:
                time_col = df['t'].values
                df = df.drop(columns=['t'])
            elif 'time' in df.columns:
                time_col = df['time'].values
                df = df.drop(columns=['time'])
            elif df.shape[1] == 6:
                df.columns = ['time', 'ia', 'ib', 'ic', 'Te', 'wm']
                time_col = df['time'].values
                df = df.drop(columns=['time'])
            elif df.shape[1] == 5:
                df.columns = ['ia', 'ib', 'ic', 'Te', 'wm']
                time_col = np.arange(len(df)) * 0.0001
            else:
                st.error(f"Unexpected CSV format: {df.shape[1]} columns")
                st.stop()

            required = ['ia', 'ib', 'ic', 'Te', 'wm']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

    st.success(f"✅ Loaded: {uploaded_file.name} — {len(df)} samples, {df.shape[1]} channels")

    # ========================================
    # EXTRACT FEATURES
    # ========================================
    with st.spinner('🔧 Extracting features...'):
        feature_row, feat_dict = extract_features(df)

    # ========================================
    # DUAL PREDICTION
    # ========================================
    st.markdown("## 🎯 Diagnosis Results")

    col_ml, col_fuzzy = st.columns(2)

    # --- ML Prediction ---
    with col_ml:
        st.markdown("### 🤖 Machine Learning")
        if ml_model is not None:
            n_model_features = ml_model.n_features_in_
            if n_model_features != 19:
                st.error(f"Model expects {n_model_features} features, got 19. Retrain!")
            else:
                ml_pred = int(ml_model.predict(feature_row)[0])
                ml_proba = ml_model.predict_proba(feature_row)[0]
                ml_conf = np.max(ml_proba) * 100

                st.markdown(
                    f"<div style='background-color:{CLASS_COLORS[ml_pred]}20; "
                    f"border-left: 5px solid {CLASS_COLORS[ml_pred]}; "
                    f"padding: 20px; border-radius: 8px; margin: 10px 0;'>"
                    f"<h2 style='margin:0;'>{CLASS_ICONS[ml_pred]} {CLASS_NAMES[ml_pred]}</h2>"
                    f"<p style='margin:5px 0 0 0; font-size:18px;'>Confidence: <b>{ml_conf:.1f}%</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Probability bar chart
                fig_ml = go.Figure(go.Bar(
                    x=ml_proba * 100,
                    y=CLASS_NAMES,
                    orientation='h',
                    marker_color=CLASS_COLORS,
                    text=[f'{p:.1f}%' for p in ml_proba * 100],
                    textposition='auto'
                ))
                fig_ml.update_layout(
                    title='Class Probabilities',
                    xaxis_title='Probability (%)',
                    height=250, margin=dict(l=0, r=0, t=35, b=0),
                    xaxis=dict(range=[0, 105])
                )
                st.plotly_chart(fig_ml, use_container_width=True)
        else:
            st.warning("ML model not available")

    # --- Fuzzy Prediction ---
    with col_fuzzy:
        st.markdown("### 🧠 Fuzzy Logic")
        if fuzzy_system is not None and fuzzy_predict is not None:
            fuzzy_cls, fuzzy_name, fuzzy_val = fuzzy_predict(
                fuzzy_system,
                feat_dict['current_unbalance'],
                feat_dict['ib_rms'],
                feat_dict['wm_mean'],
                feat_dict['Te_mean']
            )

            st.markdown(
                f"<div style='background-color:{CLASS_COLORS[fuzzy_cls]}20; "
                f"border-left: 5px solid {CLASS_COLORS[fuzzy_cls]}; "
                f"padding: 20px; border-radius: 8px; margin: 10px 0;'>"
                f"<h2 style='margin:0;'>{CLASS_ICONS[fuzzy_cls]} {fuzzy_name}</h2>"
                f"<p style='margin:5px 0 0 0; font-size:18px;'>"
                f"Defuzzified: <b>{fuzzy_val:.3f}</b> "
                f"(0=Healthy, 1=Unbal, 2=Rotor, 3=Stator)</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Fuzzy inputs display
            fuzzy_inputs = {
                'Current Unbalance': feat_dict['current_unbalance'],
                'Phase B RMS (A)': feat_dict['ib_rms'],
                'Mean Speed (rad/s)': feat_dict['wm_mean'],
                'Mean Torque (Nm)': feat_dict['Te_mean']
            }

            fig_fz = go.Figure(go.Bar(
                x=list(fuzzy_inputs.values()),
                y=list(fuzzy_inputs.keys()),
                orientation='h',
                marker_color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'],
                text=[f'{v:.2f}' for v in fuzzy_inputs.values()],
                textposition='auto'
            ))
            fig_fz.update_layout(
                title='Fuzzy Input Values',
                height=250, margin=dict(l=0, r=0, t=35, b=0)
            )
            st.plotly_chart(fig_fz, use_container_width=True)
        else:
            st.warning("Fuzzy system not available")

    # ========================================
    # AGREEMENT CHECK
    # ========================================
    if ml_model is not None and fuzzy_system is not None:
        st.markdown("---")
        if ml_pred == fuzzy_cls:
            st.success(
                f"✅ **Both systems agree: {CLASS_ICONS[ml_pred]} {CLASS_NAMES[ml_pred]}** — "
                f"High confidence diagnosis"
            )
        else:
            st.warning(
                f"⚠️ **Disagreement!** ML says {CLASS_NAMES[ml_pred]}, "
                f"Fuzzy says {fuzzy_name}. Manual inspection recommended."
            )

    # ========================================
    # MOTOR PARAMETERS
    # ========================================
    st.markdown("---")
    st.markdown("## 📊 Motor Parameters")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Phase A RMS", f"{feat_dict['ia_rms']:.2f} A")
    p2.metric("Avg Torque", f"{feat_dict['Te_mean']:.2f} Nm")
    p3.metric("Avg Speed", f"{feat_dict['wm_mean']:.2f} rad/s")
    p4.metric("Current Unbalance", f"{feat_dict['current_unbalance']:.4f}")

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("Phase B RMS", f"{feat_dict['ib_rms']:.2f} A")
    p6.metric("Phase C RMS", f"{feat_dict['ic_rms']:.2f} A")
    p7.metric("Torque Ripple", f"{feat_dict['torque_ripple']:.2f}")
    p8.metric("Speed Ripple", f"{feat_dict['speed_ripple']:.4f}")

    # ========================================
    # SIGNAL PLOTS
    # ========================================
    st.markdown("---")
    st.markdown("## 📈 Signal Analysis")

    # Downsample for plotting
    step = max(1, len(df) // 2000)
    t_plot = time_col[::step]
    df_plot = df.iloc[::step]

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔌 Phase Currents", "⚙️ Torque", "🔄 Speed", "📊 FFT Spectrum"
    ])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t_plot, y=df_plot['ia'], name='ia',
                                  line=dict(color='#e74c3c', width=1)))
        fig1.add_trace(go.Scatter(x=t_plot, y=df_plot['ib'], name='ib',
                                  line=dict(color='#3498db', width=1)))
        fig1.add_trace(go.Scatter(x=t_plot, y=df_plot['ic'], name='ic',
                                  line=dict(color='#2ecc71', width=1)))
        fig1.update_layout(
            title='Three-Phase Stator Currents',
            xaxis_title='Time (s)', yaxis_title='Current (A)',
            height=450, template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t_plot, y=df_plot['Te'], name='Te',
                                  line=dict(color='#9b59b6', width=1)))
        fig2.update_layout(
            title='Electromagnetic Torque',
            xaxis_title='Time (s)', yaxis_title='Torque (Nm)',
            height=450, template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_plot, y=df_plot['wm'], name='ωm',
                                  line=dict(color='#e67e22', width=1)))
        fig3.update_layout(
            title='Rotor Speed',
            xaxis_title='Time (s)', yaxis_title='Speed (rad/s)',
            height=450, template='plotly_white'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        ia_vals = df['ia'].values.astype(float)
        N = len(ia_vals)
        fs = 1.0 / (time_col[1] - time_col[0]) if len(time_col) > 1 else 10000
        freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
        fft_mag = np.abs(np.fft.fft(ia_vals))[:N//2]
        fft_mag_db = 20 * np.log10(fft_mag + 1e-12)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=freqs, y=fft_mag_db, name='FFT',
                                  line=dict(color='#8e44ad', width=1)))
        fig4.update_layout(
            title='Phase A Current — Frequency Spectrum',
            xaxis_title='Frequency (Hz)', yaxis_title='Magnitude (dB)',
            height=450, template='plotly_white',
            xaxis=dict(range=[0, min(500, freqs[-1])])
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            f"Sampling rate: {fs:.0f} Hz | "
            f"Dominant frequency bin: {int(feat_dict['ia_domFreq'])} | "
            f"THD-like ratio: {feat_dict['thd_like']:.4f}"
        )

    # ========================================
    # ALL FEATURES TABLE
    # ========================================
    with st.expander("🔍 View All 19 Extracted Features"):
        feat_df = pd.DataFrame({
            'Feature': list(feat_dict.keys()),
            'Value': [f"{v:.6f}" for v in feat_dict.values()]
        })
        st.dataframe(feat_df, use_container_width=True, height=500)

    # ========================================
    # METHOD COMPARISON TABLE
    # ========================================
    st.markdown("---")
    st.markdown("## 📋 Method Comparison")

    comp_data = {
        'Aspect': ['Type', 'Features Used', 'Accuracy (Validation)',
                   'Interpretable', 'Training Required', 'Best For'],
        'Random Forest (ML)': ['Black-box', '19', '100.00%',
                               '❌ No', '✅ Yes (800 samples)', 'High accuracy'],
        'Fuzzy Logic': ['Rule-based', '4', '99.88%',
                        '✅ Yes (11 rules)', '❌ No (expert design)', 'Explainability']
    }
    st.table(pd.DataFrame(comp_data))

else:
    # No file uploaded — show instructions
    st.markdown("---")
    st.info(
        "### 👆 Upload a motor CSV file to begin\n\n"
        "**Expected CSV format:**\n"
        "```\n"
        "t, ia, ib, ic, Te, wm\n"
        "0.0000, -0.034, 0.000, 0.063, 0.001, 147.788\n"
        "0.0001, -0.045, 0.000, -0.059, -0.021, 147.729\n"
        "...\n"
        "```\n\n"
        "**Fault Classes:**\n"
        "- 🟢 Healthy — Normal operation\n"
        "- 🟡 Voltage Unbalance — Supply voltage asymmetry\n"
        "- 🟣 Broken Rotor Bar — Rotor cage damage\n"
        "- 🔴 Stator Fault — Stator winding open circuit"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 Random Forest")
        st.markdown(
            "- 150 decision trees\n"
            "- 19 time/frequency features\n"
            "- 100% validation accuracy\n"
            "- Black-box classifier"
        )
    with col2:
        st.markdown("### 🧠 Fuzzy Logic")
        st.markdown(
            "- 11 Mamdani rules\n"
            "- 4 physical inputs\n"
            "- 99.88% validation accuracy\n"
            "- Human-readable rules"
        )