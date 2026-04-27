"""
What-If Simulator — Streamlit Page
Interactive disruption scenario simulator that shows how disruptions propagate across the fleet.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.whatif_simulator import WhatIfSimulator

st.set_page_config(page_title="What-If Simulator", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .sim-header {
        background: linear-gradient(135deg, #0c0c1d 0%, #1a0a2e 50%, #2d1b69 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(45, 27, 105, 0.4);
    }
    .sim-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .sim-header p { font-size: 1rem; opacity: 0.8; font-weight: 300; }
    
    .impact-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .impact-card h2 { font-size: 2rem; font-weight: 700; margin: 0; }
    .impact-card p { font-size: 0.8rem; opacity: 0.9; margin: 0.2rem 0 0; }
    
    .bg-purple { background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); }
    .bg-red { background: linear-gradient(135deg, #e17055 0%, #d63031 100%); }
    .bg-yellow { background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%); }
    .bg-green { background: linear-gradient(135deg, #00b894 0%, #55efc4 100%); }
    .bg-blue { background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%); }
    
    .shipment-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        color: #eee;
    }
    .shipment-card.at-risk { border-left: 4px solid #d63031; background: #2d1a1a; }
    .shipment-card.newly-at-risk { border-left: 4px solid #fdcb6e; background: #2d2a1a; }
    .shipment-card.safe { border-left: 4px solid #00b894; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="sim-header">
    <h1>🧪 What-If Disruption Simulator</h1>
    <p>Simulate disruption scenarios and see how they propagate across your supply chain fleet in real-time</p>
</div>
""", unsafe_allow_html=True)

# ── Initialize ──
@st.cache_resource
def get_simulator():
    return WhatIfSimulator()

sim = get_simulator()

# ── Scenario Builder ──
st.markdown("### 🎛️ Build Your Scenario")

# Load IMD weather data if available
try:
    imd_data = pd.read_csv("data/processed/imd_district_rainfall.csv")
    weather_risks = imd_data[imd_data['Weather_Risk'].isin(["HIGH (Flood Risk)", "MEDIUM (Heavy Rain)"])]
    has_weather_data = not weather_risks.empty
except:
    has_weather_data = False

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    disruption_type = st.selectbox(
        "Disruption Type",
        ["weather", "highway_closure", "port_congestion", "strike"],
        format_func=lambda x: {
            'weather': '🌧️ Severe Weather (IMD Live)',
            'highway_closure': '🚧 Highway Closure',
            'port_congestion': '🚢 Port Congestion',
            'strike': '✊ Strike / Bandh',
        }[x]
    )

with col_s2:
    if disruption_type == "weather" and has_weather_data:
        # Get top 50 districts by risk
        districts = weather_risks['Location'].tolist()[:50]
        affected_region = st.selectbox(
            "Affected District (Real-Time IMD Data)",
            districts,
            format_func=lambda x: f"📍 {x.title()}"
        )
    else:
        affected_region = st.selectbox(
            "Affected Region",
            ["west", "north", "south", "east", "central"],
            format_func=lambda x: {
                'west': '🏙️ West (Mumbai, Gujarat)',
                'north': '🏔️ North (Delhi, UP, Punjab)',
                'south': '🌴 South (Bangalore, Chennai)',
                'east': '🌿 East (Kolkata, Odisha)',
                'central': '🏛️ Central (MP, Chhattisgarh)',
            }[x]
        )

with col_s3:
    if disruption_type == "weather" and has_weather_data:
        district_risk = weather_risks[weather_risks['Location'] == affected_region]['Weather_Risk'].values[0]
        default_sev = 0.9 if "HIGH" in str(district_risk) else 0.6
        severity = st.slider("Severity", 0.1, 1.0, default_sev, 0.1,
                             help=f"Auto-set based on IMD data: {district_risk}")
    else:
        severity = st.slider("Severity", 0.1, 1.0, 0.7, 0.1,
                             help="0.1 = Minor disruption, 1.0 = Complete shutdown")

with col_s4:
    fleet_size = st.slider("Fleet Size", 10, 50, 25)

# ── Disruption context ──
context_map = {
    'weather': {
        'description': 'Heavy rainfall causing road flooding, reduced visibility, and increased accident risk.',
        'icon': '🌧️',
        'india_example': f'LIVE DATA: {affected_region.title()} is currently facing {district_risk if (has_weather_data and disruption_type=="weather") else "Severe Weather"}'
    },
    'highway_closure': {
        'description': 'Major highway segment closed due to accident/construction, forcing detours.',
        'icon': '🚧',
        'india_example': 'Like NH-44 closure near Dharuhera affecting Delhi-Jaipur traffic',
    },
    'port_congestion': {
        'description': 'Container terminal congestion causing vessel delays and cargo backlog.',
        'icon': '🚢',
        'india_example': 'JNPT/Mundra port congestion during peak shipping season',
    },
    'strike': {
        'description': 'Transport workers strike or Bharat Bandh disrupting all ground movement.',
        'icon': '✊',
        'india_example': 'Like September 2023 truckers strike affecting pan-India logistics',
    },
}

ctx = context_map[disruption_type]
st.info(f"**{ctx['icon']} {disruption_type.replace('_', ' ').title()}:** {ctx['description']}\n\n*India context: {ctx['india_example']}*")

# ── Run Simulation ──
if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
    fleet = sim.generate_sample_fleet(fleet_size, inject_region=affected_region)
    
    with st.spinner("Simulating disruption propagation across fleet..."):
        result = sim.simulate_disruption(
            shipments=fleet,
            disruption_type=disruption_type,
            affected_region=affected_region,
            severity=severity
        )
    
    impact = result['impact_summary']
    
    st.markdown("---")
    st.markdown("### 📊 Impact Assessment")
    
    # ── Impact Metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="impact-card bg-purple"><h2>{impact['total_shipments_analyzed']}</h2><p>Total Analyzed</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="impact-card bg-blue"><h2>{impact['shipments_in_affected_region']}</h2><p>In Affected Zone</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="impact-card bg-red"><h2>{impact['newly_at_risk']}</h2><p>Newly At Risk</p></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="impact-card bg-yellow"><h2>₹{impact['estimated_penalty_inr']:,}</h2><p>Estimated Penalty</p></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="impact-card bg-green"><h2>{impact['avg_delay_prob_increase']:.0%}</h2><p>Avg Risk Increase</p></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Before/After Comparison ──
    col_before, col_after = st.columns(2)
    
    details = result['shipment_details']
    affected_shipments = [d for d in details if d['is_in_affected_region']]
    
    with col_before:
        st.markdown("### 🔵 Before Disruption")
        import plotly.express as px
        
        before_probs = [d['baseline_delay_prob'] for d in details]
        fig_before = px.histogram(
            x=before_probs, nbins=10,
            labels={'x': 'Delay Probability'},
            template='plotly_dark',
            color_discrete_sequence=['#0984e3'],
        )
        fig_before.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=250, margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig_before, use_container_width=True)
    
    with col_after:
        st.markdown("### 🔴 After Disruption")
        after_probs = [d['disrupted_delay_prob'] for d in details]
        fig_after = px.histogram(
            x=after_probs, nbins=10,
            labels={'x': 'Delay Probability'},
            template='plotly_dark',
            color_discrete_sequence=['#d63031'],
        )
        fig_after.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=250, margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig_after, use_container_width=True)
    
    st.markdown("---")
    
    # ── Waterfall: Biggest Movers ──
    st.markdown("### 📈 Risk Change by Shipment")
    
    changes = pd.DataFrame([{
        'Shipment': d['shipment_id'],
        'Before': d['baseline_delay_prob'],
        'After': d['disrupted_delay_prob'],
        'Change': d['prob_increase'],
        'Region': d['region'].title(),
        'In Zone': '🔴 Yes' if d['is_in_affected_region'] else '🟢 No',
        'Status': '⚠️ NEW RISK' if d['newly_at_risk'] else ('🔴 Already Risky' if d['was_at_risk'] else '✅ Safe'),
    } for d in details])
    
    changes = changes.sort_values('Change', ascending=False)
    
    fig_change = px.bar(
        changes.head(15),
        x='Shipment', y='Change',
        color='In Zone',
        color_discrete_map={'🔴 Yes': '#d63031', '🟢 No': '#00b894'},
        template='plotly_dark',
        labels={'Change': 'Delay Probability Change'},
    )
    fig_change.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig_change, use_container_width=True)
    
    # ── Detailed Results Table ──
    st.markdown("### 📋 All Shipment Results")
    
    display = changes.copy()
    display['Before'] = display['Before'].apply(lambda x: f"{x:.0%}")
    display['After'] = display['After'].apply(lambda x: f"{x:.0%}")
    display['Change'] = display['Change'].apply(lambda x: f"+{x:.0%}" if x > 0 else f"{x:.0%}")
    
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    # ── Executive Summary ──
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")
    
    st.markdown(f"""
    **Scenario:** {ctx['icon']} {disruption_type.replace('_', ' ').title()} in {affected_region.title()} region at {severity:.0%} severity
    
    **Key Findings:**
    - **{impact['newly_at_risk']}** out of **{impact['shipments_in_affected_region']}** shipments in the affected region became newly at risk
    - Average delay probability increased by **{impact['avg_delay_prob_increase']:.1%}** for affected shipments  
    - Estimated financial impact: **₹{impact['estimated_penalty_inr']:,}** in potential SLA penalties
    - If rerouted, CO₂ impact: **{impact['reroute_co2_savings_kg']} kg** from additional fuel consumption
    
    **Recommended Actions:**
    1. Trigger automatic rerouting for all newly at-risk shipments
    2. Alert logistics managers in the {affected_region.title()} operations center
    3. Pre-position inventory in unaffected distribution hubs
    4. Activate backup carrier agreements for critical shipments
    """)

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#888;">
        <div style="font-size:3rem; margin-bottom:1rem;">🧪</div>
        <p style="font-size:1.1rem;">Configure a disruption scenario above and click <strong>Run Simulation</strong></p>
        <p style="font-size:0.9rem; margin-top:0.5rem;">The simulator models how disruptions propagate across your fleet using the trained XGBoost delay predictor</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("What-If Simulator powered by XGBoost Delay Prediction | LogiTrack AI")
