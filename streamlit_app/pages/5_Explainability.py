"""
Model Explainability Dashboard — Streamlit Page
Transparent AI: see WHY the model predicts delays with feature contribution analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.explainability import DelayExplainer, ETAExplainer

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .xai-header {
        background: linear-gradient(135deg, #0c3547 0%, #1a5276 50%, #2980b9 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(41, 128, 185, 0.3);
    }
    .xai-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .xai-header p { font-size: 1rem; opacity: 0.8; font-weight: 300; }
    
    .pred-box {
        padding: 1.5rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .pred-box h2 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .pred-box p { font-size: 0.85rem; opacity: 0.9; margin: 0.3rem 0 0; }
    
    .pred-delayed { background: linear-gradient(135deg, #d63031 0%, #e17055 100%); box-shadow: 0 6px 20px rgba(214, 48, 49, 0.3); }
    .pred-ontime { background: linear-gradient(135deg, #00b894 0%, #55efc4 100%); box-shadow: 0 6px 20px rgba(0, 184, 148, 0.3); }
    
    .factor-bar {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .factor-name { font-weight: 500; min-width: 150px; color: #ddd; font-size: 0.9rem; }
    .factor-value { color: #aaa; font-size: 0.8rem; min-width: 100px; }
    .factor-importance { font-weight: 700; color: #6c5ce7; font-size: 0.95rem; }
    
    .explanation-box {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
        border-left: 4px solid #6c5ce7;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="xai-header">
    <h1>🧠 Model Explainability</h1>
    <p>Transparent AI — understand WHY our models predict delays and ETAs with per-feature contribution analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Initialize ──
@st.cache_resource
def get_delay_explainer():
    return DelayExplainer()

@st.cache_resource
def get_eta_explainer():
    return ETAExplainer()

delay_exp = get_delay_explainer()
eta_exp = get_eta_explainer()

# ── Tab Selection ──
tab1, tab2, tab3 = st.tabs(["🔮 Delay Risk Explainer", "⏱️ ETA Explainer", "📊 Global Feature Importance"])

# ═══════════════════════════════════════════════════
# TAB 1: Delay Risk Explainer
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown("### Configure Shipment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        partner = st.selectbox("Delivery Partner", ["delhivery", "shadowfax", "xpressbees", "dhl", "bluedart"], key="d_partner")
        package = st.selectbox("Package Type", ["electronics", "groceries", "automobile parts", "cosmetics", "medicines"], key="d_package")
        vehicle = st.selectbox("Vehicle Type", ["truck", "ev van", "bike", "three wheeler"], key="d_vehicle")
        mode = st.selectbox("Delivery Mode", ["same day", "express", "standard", "two day"], key="d_mode")
    
    with col2:
        region = st.selectbox("Region", ["north", "south", "east", "west", "central"], key="d_region")
        weather = st.selectbox("Weather", ["clear", "rainy", "cold", "foggy", "stormy"], key="d_weather")
        distance = st.slider("Distance (km)", 5.0, 1500.0, 350.0, 5.0, key="d_dist")
        weight = st.slider("Weight (kg)", 0.5, 100.0, 15.0, 0.5, key="d_weight")
    
    # Preset scenarios
    st.markdown("#### Quick Scenarios")
    sc1, sc2, sc3 = st.columns(3)
    
    presets = {
        "🔴 High Risk": {
            "delivery_partner": "delhivery", "package_type": "electronics",
            "vehicle_type": "bike", "delivery_mode": "same day",
            "region": "west", "weather_condition": "stormy",
            "distance_km": 350.5, "package_weight_kg": 15.2,
        },
        "🟡 Medium Risk": {
            "delivery_partner": "shadowfax", "package_type": "groceries",
            "vehicle_type": "ev van", "delivery_mode": "express",
            "region": "south", "weather_condition": "rainy",
            "distance_km": 180.0, "package_weight_kg": 8.0,
        },
        "🟢 Low Risk": {
            "delivery_partner": "dhl", "package_type": "cosmetics",
            "vehicle_type": "ev van", "delivery_mode": "two day",
            "region": "south", "weather_condition": "clear",
            "distance_km": 25.0, "package_weight_kg": 2.0,
        },
    }
    
    selected_preset = None
    for i, (label, preset) in enumerate(presets.items()):
        with [sc1, sc2, sc3][i]:
            if st.button(label, use_container_width=True):
                selected_preset = preset
    
    # Build features
    if selected_preset:
        features = selected_preset
    else:
        features = {
            "delivery_partner": partner,
            "package_type": package,
            "vehicle_type": vehicle,
            "delivery_mode": mode,
            "region": region,
            "weather_condition": weather,
            "distance_km": distance,
            "package_weight_kg": weight,
        }
    
    if st.button("🔍 Explain Prediction", use_container_width=True, type="primary", key="delay_btn"):
        result = delay_exp.explain(features)
        
        if 'error' in result:
            st.error(result['error'])
        else:
            st.markdown("---")
            
            # ── Prediction Result ──
            prob = result['probability']
            pred = result['prediction']
            risk = result['risk_level']
            
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                css_class = "pred-delayed" if pred == "Delayed" else "pred-ontime"
                st.markdown(f'<div class="pred-box {css_class}"><h2>{prob:.0%}</h2><p>Delay Probability</p></div>', unsafe_allow_html=True)
            with pc2:
                st.markdown(f'<div class="pred-box {"pred-delayed" if risk in ["HIGH","CRITICAL"] else "pred-ontime"}"><h2>{risk}</h2><p>Risk Level</p></div>', unsafe_allow_html=True)
            with pc3:
                st.markdown(f'<div class="pred-box {"pred-delayed" if pred=="Delayed" else "pred-ontime"}"><h2>{pred}</h2><p>Prediction</p></div>', unsafe_allow_html=True)
            
            # ── Natural Language Explanation ──
            st.markdown(f'<div class="explanation-box">💡 {result["explanation"]}</div>', unsafe_allow_html=True)
            
            # ── Feature Contributions ──
            st.markdown("### 📊 Feature Contributions")
            
            factors = result.get('top_factors', [])
            if factors:
                import plotly.express as px
                
                factor_df = pd.DataFrame(factors)
                fig = px.bar(
                    factor_df, x='importance', y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='RdYlGn_r',
                    template='plotly_dark',
                    labels={'importance': 'Importance (%)', 'feature': 'Feature'},
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=300, margin=dict(t=10, b=30, l=10),
                    font=dict(family="Inter"),
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed factor list
                for f in factors:
                    st.markdown(f"""
                    <div class="factor-bar">
                        <span class="factor-name">{f['feature']}</span>
                        <span class="factor-value">{f['value']}</span>
                        <span class="factor-importance">{f['impact']}</span>
                        <span style="color:{'#d63031' if f['direction']=='increases_risk' else '#00b894'}; font-size:0.8rem;">
                            {'↑ increases risk' if f['direction']=='increases_risk' else '↓ decreases risk'}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant feature contributions detected.")

# ═══════════════════════════════════════════════════
# TAB 2: ETA Explainer
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("### ETA Prediction with Explanation")
    
    ec1, ec2 = st.columns(2)
    with ec1:
        eta_dist = st.slider("Distance (km)", 0.1, 20.0, 3.0, 0.1, key="eta_dist")
        eta_hour = st.slider("Pickup Hour", 0, 23, 14, key="eta_hour")
        eta_city = st.selectbox("City", ["unknown", "city_1", "city_2", "city_3"], key="eta_city")
    with ec2:
        eta_dow = st.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], key="eta_dow")
        eta_pkgs = st.slider("Courier Daily Packages", 5, 80, 30, key="eta_pkgs")
        eta_speed = st.slider("Courier Avg Speed (km/min)", 0.01, 0.15, 0.05, 0.01, key="eta_speed")
    
    if st.button("⏱️ Predict ETA & Explain", use_container_width=True, type="primary", key="eta_btn"):
        eta_features = {
            'delivery_distance_km': eta_dist,
            'log_distance': np.log1p(eta_dist),
            'accept_hour': eta_hour,
            'day_of_week': eta_dow,
            'is_rush_hour': 1 if (8 <= eta_hour <= 10) or (17 <= eta_hour <= 19) else 0,
            'is_weekend': 1 if eta_dow >= 5 else 0,
            'courier_daily_packages': eta_pkgs,
            'courier_avg_speed': eta_speed,
            'distance_x_rush': eta_dist * (1 if (8 <= eta_hour <= 10) or (17 <= eta_hour <= 19) else 0),
            'city': eta_city,
            'time_period': 'morning' if 6<=eta_hour<12 else 'afternoon' if 12<=eta_hour<17 else 'evening' if 17<=eta_hour<21 else 'night',
            'aoi_type': 'other',
        }
        
        result = eta_exp.explain(eta_features)
        
        if 'error' in result:
            st.error(result['error'])
        else:
            st.markdown("---")
            
            eta_val = result['predicted_eta_mins']
            
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f"""
                <div class="pred-box" style="background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%); box-shadow: 0 6px 20px rgba(9,132,227,0.3);">
                    <h2>{eta_val:.0f} min</h2>
                    <p>Predicted Delivery Time</p>
                </div>
                """, unsafe_allow_html=True)
            with ec2:
                st.markdown(f'<div class="explanation-box">💡 {result["explanation"]}</div>', unsafe_allow_html=True)
            
            factors = result.get('top_factors', [])
            if factors:
                import plotly.express as px
                
                fdf = pd.DataFrame(factors)
                fig = px.bar(
                    fdf, x='importance', y='feature',
                    orientation='h',
                    color='impact_mins',
                    color_continuous_scale='Blues',
                    template='plotly_dark',
                    labels={'importance': 'Importance (%)', 'feature': 'Feature', 'impact_mins': 'Impact (min)'},
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=250, margin=dict(t=10, b=30, l=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 3: Global Feature Importance
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown("### 🌍 Global Feature Importance — Delay Predictor")
    st.markdown("Which features matter most across ALL predictions, not just one shipment.")
    
    global_imp = delay_exp.get_global_importance()
    
    if global_imp:
        import plotly.express as px
        
        gdf = pd.DataFrame(global_imp)
        
        fig = px.bar(
            gdf, x='importance', y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Viridis',
            template='plotly_dark',
            labels={'importance': 'Importance (%)', 'feature': 'Feature'},
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=500, margin=dict(t=10, b=30, l=10),
            font=dict(family="Inter"),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(pd.DataFrame(global_imp), use_container_width=True, hide_index=True)
    else:
        st.warning("Model not loaded — could not retrieve global importance.")
    
    st.markdown("---")
    st.markdown("""
    **How to read this:**
    - Higher importance = the model relies more on this feature for predictions
    - Features like `distance_km` and `weather_condition` typically rank highest  
    - Categorical features are shown as one-hot encoded columns (e.g., `weather_condition_stormy`)
    - Use this to understand model behavior and identify potential biases
    """)

st.markdown("---")
st.caption("Model Explainability powered by XGBoost Feature Importance Analysis | Smart Supply Chain AI")
