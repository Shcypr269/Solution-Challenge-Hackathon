"""
Multi-Modal Transport Optimizer — Streamlit Page
Compare road, rail, air, and waterway options for Indian logistics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.multimodal_optimizer import optimize_transport, TRANSPORT_MODES

st.set_page_config(page_title="Transport Optimizer", page_icon="🚚", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .opt-header {
        background: linear-gradient(135deg, #0a3d0c 0%, #1a5c1f 50%, #2d7d32 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(45, 125, 50, 0.3);
    }
    .opt-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .opt-header p { font-size: 1rem; opacity: 0.8; font-weight: 300; }
    
    .mode-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        color: white;
        transition: transform 0.2s;
    }
    .mode-card:hover { transform: translateY(-2px); }
    .mode-card.recommended {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        border: 2px solid #fff;
        box-shadow: 0 6px 20px rgba(0,184,148,0.4);
    }
    .mode-card.alternative {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        border: 1px solid #555;
    }
    .mode-card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; }
    
    .savings-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .save-cost { background: #dfe6e9; color: #2d3436; }
    .save-green { background: #55efc4; color: #0a3d0c; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="opt-header">
    <h1>🚚 Multi-Modal Transport Optimizer</h1>
    <p>Compare road, rail, air, and waterway options — optimized for Indian logistics corridors with cost, speed, and carbon metrics</p>
</div>
""", unsafe_allow_html=True)

# ── Transport Modes Overview ──
with st.expander("📖 Available Transport Modes", expanded=False):
    modes_data = []
    for mode_id, mode in TRANSPORT_MODES.items():
        modes_data.append({
            'Mode': mode.name,
            'Cost/Tonne-Km (₹)': mode.cost_per_tonne_km,
            'Speed (km/h)': mode.speed_kmh,
            'CO₂/km (kg)': mode.co2_per_km_kg,
            'Fixed Cost (₹)': mode.fixed_cost,
            'Min Distance (km)': mode.min_distance_km,
            'Max Weight (kg)': f"{mode.max_weight_kg:,}",
            'Reliability': f"{mode.reliability:.0%}",
        })
    st.dataframe(pd.DataFrame(modes_data), use_container_width=True, hide_index=True)

# ── Input Form ──
st.markdown("### 📦 Shipment Details")

col_i1, col_i2, col_i3, col_i4 = st.columns(4)

with col_i1:
    distance = st.number_input("Distance (km)", 5.0, 5000.0, 500.0, 10.0, 
                               help="Origin-to-destination distance")
with col_i2:
    weight = st.number_input("Weight (kg)", 0.5, 100000.0, 200.0, 10.0,
                             help="Total shipment weight")
with col_i3:
    deadline = st.number_input("Deadline (hours)", 1.0, 240.0, 48.0, 1.0,
                               help="Maximum allowed delivery time")
with col_i4:
    priority = st.selectbox("Priority", ["balanced", "cost", "speed", "green"],
                           format_func=lambda x: {
                               'balanced': '⚖️ Balanced',
                               'cost': '💰 Lowest Cost',
                               'speed': '⚡ Fastest',
                               'green': '🌱 Greenest',
                           }[x])

col_w1, col_w2 = st.columns([1, 3])
with col_w1:
    weather_severity = st.slider("Weather Severity", 0.0, 1.0, 0.0, 0.1,
                                 help="0 = Clear, 1 = Severe monsoon")

# ── Popular Routes ──
st.markdown("#### 🗺️ Quick Pick — Popular Indian Routes")
route_cols = st.columns(5)
routes = [
    {"label": "Mumbai→Delhi", "dist": 1400, "wt": 500, "dl": 48},
    {"label": "Bangalore→Chennai", "dist": 350, "wt": 50, "dl": 12},
    {"label": "Delhi→Kolkata", "dist": 1500, "wt": 2000, "dl": 72},
    {"label": "Hyderabad→Mumbai", "dist": 700, "wt": 100, "dl": 24},
    {"label": "Local (15km)", "dist": 15, "wt": 5, "dl": 4},
]

selected_route = None
for i, route in enumerate(routes):
    with route_cols[i]:
        if st.button(route["label"], use_container_width=True):
            selected_route = route

# ── Optimize ──
st.markdown("---")

use_dist = selected_route['dist'] if selected_route else distance
use_wt = selected_route['wt'] if selected_route else weight
use_dl = selected_route['dl'] if selected_route else deadline

if selected_route:
    st.info(f"📍 Using route: **{selected_route['label']}** — {use_dist} km, {use_wt} kg, {use_dl}h deadline")

result = optimize_transport(use_dist, use_wt, use_dl, priority, weather_severity)
rec = result['recommended']
alternatives = result['alternatives']
savings = result['savings']

if not rec:
    st.error("No viable transport mode found for this combination. Try adjusting weight or distance.")
else:
    # ── Results ──
    st.markdown("### 🏆 Recommendation")
    
    st.markdown(f"""
    <div class="mode-card recommended">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3>✅ {rec['mode']}</h3>
                <div style="display:flex; gap:2rem; font-size:0.9rem;">
                    <span>💰 ₹{rec['total_cost_inr']:,}</span>
                    <span>⏱️ {rec['travel_time_hrs']}h</span>
                    <span>🌿 {rec['co2_emissions_kg']} kg CO₂</span>
                    <span>📊 {rec['reliability']:.0%} reliable</span>
                </div>
            </div>
            <div style="font-size:0.8rem; text-align:right;">
                <span>{'✅ Meets deadline' if rec['meets_deadline'] else '⚠️ May exceed deadline'}</span>
                <br><span>Score: {rec['score']:.3f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Savings
    if savings['cost_saving_inr'] > 0 or savings['co2_saving_kg'] > 0:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f'<span class="savings-badge save-cost">💰 Saves ₹{savings["cost_saving_inr"]:,} vs most expensive option</span>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<span class="savings-badge save-green">🌱 Saves {savings["co2_saving_kg"]} kg CO₂ vs highest emission option</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── All Options Comparison ──
    st.markdown("### 📊 All Transport Options")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    modes = [a['mode'] for a in alternatives]
    costs = [a['total_cost_inr'] for a in alternatives]
    times = [a['travel_time_hrs'] for a in alternatives]
    co2s = [a['co2_emissions_kg'] for a in alternatives]
    scores = [a['score'] for a in alternatives]
    colors = ['#00b894' if a['mode_id'] == rec['mode_id'] else '#636e72' for a in alternatives]
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("Cost (₹)", "Travel Time (h)", "CO₂ Emissions (kg)", "Overall Score"))
    
    fig.add_trace(go.Bar(x=modes, y=costs, marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=modes, y=times, marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=modes, y=co2s, marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=modes, y=scores, marker_color=colors, showlegend=False), row=2, col=2)
    
    # Add deadline line
    fig.add_hline(y=use_dl, line_dash="dash", line_color="#d63031",
                  annotation_text="Deadline", row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(t=40, b=30),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Alternatives List ──
    st.markdown("### 🔄 Alternative Options")
    
    for alt in alternatives:
        if alt['mode_id'] == rec['mode_id']:
            continue
        
        deadline_status = '✅' if alt['meets_deadline'] else '⚠️ LATE'
        
        st.markdown(f"""
        <div class="mode-card alternative">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3>{alt['mode']}</h3>
                    <div style="display:flex; gap:2rem; font-size:0.85rem; opacity:0.9;">
                        <span>💰 ₹{alt['total_cost_inr']:,}</span>
                        <span>⏱️ {alt['travel_time_hrs']}h</span>
                        <span>🌿 {alt['co2_emissions_kg']} kg CO₂</span>
                        <span>📊 {alt['reliability']:.0%}</span>
                        <span>{deadline_status}</span>
                    </div>
                </div>
                <div style="font-size:1rem; font-weight:600;">{alt['score']:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Detailed Table ──
    st.markdown("### 📋 Detailed Comparison")
    comp_data = [{
        'Mode': a['mode'],
        'Cost (₹)': f"₹{a['total_cost_inr']:,}",
        'Time (hrs)': a['travel_time_hrs'],
        'CO₂ (kg)': a['co2_emissions_kg'],
        'Reliability': f"{a['reliability']:.0%}",
        'Meets Deadline': '✅' if a['meets_deadline'] else '❌',
        'Score': a['score'],
        'Recommended': '⭐' if a['mode_id'] == rec['mode_id'] else '',
    } for a in alternatives]
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Multi-Modal Optimizer calibrated for Indian logistics costs (AITD 2024 data) | LogiTrack AI")
