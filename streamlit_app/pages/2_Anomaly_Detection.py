"""
Anomaly Detection Dashboard — Streamlit Page
Visualizes real-time anomaly detection across the fleet using Isolation Forest.
Now uses HTTP calls to the FastAPI backend instead of direct ML imports.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

ML_URL = os.environ.get("ML_ENGINE_URL", "https://logitrackai.onrender.com")

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .anomaly-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .anomaly-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .anomaly-header p { font-size: 1rem; opacity: 0.8; font-weight: 300; }
    
    .alert-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    .alert-card:hover { transform: translateY(-3px); }
    .alert-card h2 { font-size: 2rem; font-weight: 700; margin: 0; }
    .alert-card p { font-size: 0.8rem; opacity: 0.9; margin: 0.2rem 0 0; }
    
    .bg-critical { background: linear-gradient(135deg, #d63031 0%, #e17055 100%); }
    .bg-high { background: linear-gradient(135deg, #e17055 0%, #fdcb6e 100%); }
    .bg-normal { background: linear-gradient(135deg, #00b894 0%, #55efc4 100%); }
    .bg-total { background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); }
    
    .anomaly-row {
        background: #1e1e2f;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        color: #eee;
    }
    .anomaly-row.critical { border-left: 4px solid #d63031; }
    .anomaly-row.high { border-left: 4px solid #e17055; }
    .anomaly-row.medium { border-left: 4px solid #fdcb6e; }
    .anomaly-row.low { border-left: 4px solid #00b894; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="anomaly-header">
    <h1>🔍 Anomaly Detection System</h1>
    <p>Isolation Forest + Z-Score ensemble detecting unusual shipment patterns across Indian logistics corridors</p>
</div>
""", unsafe_allow_html=True)

# ── Controls ──
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
with col_ctrl1:
    fleet_size = st.slider("Fleet Size", 10, 50, 25, help="Number of shipments to analyze")
with col_ctrl2:
    sensitivity = st.select_slider("Sensitivity", options=["Low", "Medium", "High"], value="Medium")

# ── Fetch batch anomaly data from FastAPI ──
@st.cache_data(ttl=120)
def fetch_anomaly_batch(size):
    try:
        r = requests.post(f"{ML_URL}/api/v1/ml/anomaly-detect-batch", json={"fleet_size": size}, timeout=30)
        if r.ok:
            return r.json()
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

batch_result = fetch_anomaly_batch(fleet_size)

if not batch_result:
    st.error("Could not connect to ML engine. Make sure the FastAPI backend is running.")
    st.stop()

summary = batch_result['summary']
risk_dist = batch_result['risk_distribution']
fleet = batch_result.get('fleet', [])

# ── Summary Metrics ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="alert-card bg-total"><h2>{summary['total_shipments']}</h2><p>Total Shipments</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="alert-card bg-critical"><h2>{summary['anomalies_detected']}</h2><p>Anomalies Detected</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="alert-card bg-high"><h2>{summary['critical_alerts'] + summary['high_alerts']}</h2><p>Critical + High Alerts</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="alert-card bg-normal"><h2>{summary['anomaly_rate']:.0%}</h2><p>Anomaly Rate</p></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Risk Distribution Chart ──
col_chart, col_details = st.columns([1, 1])

with col_chart:
    st.markdown("### 📊 Risk Distribution")
    risk_df = pd.DataFrame({
        'Risk Level': list(risk_dist.keys()),
        'Count': list(risk_dist.values()),
    })
    colors = {'LOW': '#00b894', 'MEDIUM': '#fdcb6e', 'HIGH': '#e17055', 'CRITICAL': '#d63031'}
    
    import plotly.express as px
    fig = px.bar(
        risk_df, x='Risk Level', y='Count',
        color='Risk Level',
        color_discrete_map=colors,
        template='plotly_dark',
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter"),
        height=300,
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_details:
    st.markdown("### 📈 Anomaly Score Distribution")
    scores = [r['anomaly_score'] for r in batch_result['results']]
    fig2 = px.histogram(
        x=scores, nbins=15,
        labels={'x': 'Anomaly Score', 'y': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=['#6c5ce7'],
    )
    fig2.add_vline(x=0.4, line_dash="dash", line_color="#d63031", annotation_text="Anomaly Threshold")
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter"),
        height=300,
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Anomaly Alerts Feed ──
st.markdown("### 🚨 Anomaly Alerts")

anomalies = [r for r in batch_result['results'] if r['is_anomaly']]
anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)

# Build a lookup map from fleet data
fleet_map = {s.get('shipment_id', ''): s for s in fleet} if fleet else {}

if not anomalies:
    st.success("✅ No anomalies detected in the current fleet!")
else:
    for r in anomalies:
        level_class = r['risk_level'].lower()
        score_pct = f"{r['anomaly_score']:.0%}"
        reasons_html = "".join([f"<br>⚠️ {reason}" for reason in r['reasons']])
        
        # Get original shipment data from fleet
        shipment = fleet_map.get(r['shipment_id'], {})
        
        st.markdown(f"""
        <div class="anomaly-row {level_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="font-size:1.1rem;">{r['shipment_id']}</strong>
                    <span style="background:{'#d63031' if r['risk_level']=='CRITICAL' else '#e17055' if r['risk_level']=='HIGH' else '#fdcb6e'}; 
                           color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; margin-left:8px;">
                        {r['risk_level']}
                    </span>
                </div>
                <div style="font-size:1.3rem; font-weight:700; color:{'#d63031' if r['anomaly_score']>0.7 else '#e17055'};">
                    {score_pct}
                </div>
            </div>
            <div style="font-size:0.85rem; color:#aaa; margin-top:6px;">
                {shipment.get('region','?').title()} · {shipment.get('weather_condition','?').title()} · 
                {shipment.get('distance_km',0)} km · {shipment.get('package_weight_kg',0)} kg · 
                {shipment.get('delivery_partner','?').title()}
            </div>
            <div style="font-size:0.82rem; margin-top:6px; color:#ddd;">
                {reasons_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Full Fleet Table ──
st.markdown("### 📋 Complete Fleet Analysis")

table_data = []
for r in batch_result['results']:
    shipment = fleet_map.get(r['shipment_id'], {})
    table_data.append({
        'ID': r['shipment_id'],
        'Region': shipment.get('region', '').title(),
        'Weather': shipment.get('weather_condition', '').title(),
        'Distance (km)': shipment.get('distance_km', 0),
        'Weight (kg)': shipment.get('package_weight_kg', 0),
        'Partner': shipment.get('delivery_partner', '').title(),
        'Anomaly Score': r['anomaly_score'],
        'Risk': r['risk_level'],
        'Anomaly': '🔴' if r['is_anomaly'] else '🟢',
    })

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Single Shipment Tester ──
st.markdown("---")
st.markdown("### 🧪 Test Single Shipment")

with st.form("single_test"):
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        test_dist = st.number_input("Distance (km)", 1.0, 5000.0, 350.0)
        test_region = st.selectbox("Region", ["north", "south", "east", "west", "central"])
    with tc2:
        test_weight = st.number_input("Weight (kg)", 0.1, 1000.0, 15.0)
        test_weather = st.selectbox("Weather", ["clear", "rainy", "cold", "foggy", "stormy"])
    with tc3:
        test_partner = st.selectbox("Partner", ["delhivery", "shadowfax", "xpressbees", "dhl", "bluedart"])
        test_vehicle = st.selectbox("Vehicle", ["truck", "ev van", "bike", "three wheeler"])
    with tc4:
        test_package = st.selectbox("Package", ["electronics", "groceries", "automobile parts", "cosmetics", "medicines"])
        test_mode = st.selectbox("Mode", ["same day", "express", "standard", "two day"])
    
    submitted = st.form_submit_button("🔍 Analyze Shipment", use_container_width=True)

if submitted:
    test_payload = {
        'shipment_id': 'TEST-001',
        'distance_km': test_dist,
        'package_weight_kg': test_weight,
        'region': test_region,
        'weather_condition': test_weather,
        'delivery_partner': test_partner,
        'vehicle_type': test_vehicle,
        'package_type': test_package,
        'delivery_mode': test_mode,
    }
    try:
        r = requests.post(f"{ML_URL}/api/v1/ml/anomaly-detect", json=test_payload, timeout=30)
        if r.ok:
            result = r.json()
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Anomaly Score", f"{result['anomaly_score']:.2%}")
            rc2.metric("Risk Level", result['risk_level'])
            rc3.metric("Status", "🔴 ANOMALY" if result['is_anomaly'] else "🟢 NORMAL")
            
            if result.get('reasons'):
                st.warning("**Detected Issues:**\n" + "\n".join([f"- {r}" for r in result['reasons']]))
        else:
            st.error(f"ML Engine returned {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")

st.markdown("---")
st.caption("Anomaly Detection powered by Isolation Forest + Statistical Z-Score Ensemble | LogiTrack AI")
