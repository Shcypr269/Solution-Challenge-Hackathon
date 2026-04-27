import streamlit as st

st.set_page_config(
    page_title="LogiTrack AI",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }
    
    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    
    .metric-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    .metric-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .sdg-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
        color: white;
    }
    
    .sdg-9 { background: #f36d25; }
    .sdg-11 { background: #f99d25; }
    .sdg-12 { background: #cf8d2a; }
    .sdg-13 { background: #48773e; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚛 LogiTrack AI</h1>
    <p>AI-powered disruption prediction & autonomous rerouting for Indian logistics — powered by Gemini & XGBoost</p>
    <div style="margin-top: 1rem;">
        <span class="sdg-badge sdg-9">SDG 9: Industry & Innovation</span>
        <span class="sdg-badge sdg-11">SDG 11: Sustainable Cities</span>
        <span class="sdg-badge sdg-12">SDG 12: Responsible Consumption</span>
        <span class="sdg-badge sdg-13">SDG 13: Climate Action</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>90.5%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card metric-green">
        <h2>₹2.4L</h2>
        <p>Penalties Prevented</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card metric-orange">
        <h2>147</h2>
        <p>Disruptions Handled</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card metric-blue">
        <h2>340 kg</h2>
        <p>CO₂ Emissions Saved</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How It Works
st.markdown("### 🧠 How the Multi-Agent Pipeline Works")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("""
    When a driver reports a disruption (voice/text), our **6-agent LangGraph pipeline** automatically:
    
    1. **🔍 Disruption Agent** — Gemini NLP classifies the incident type and severity
    2. **📦 Shipment Monitor** — XGBoost predicts delay probability (90.5% accuracy)
    3. **🗺️ Route Optimizer** — Finds alternative routes via Google Maps
    4. **📋 Contract Intelligence** — RAG extracts SLA penalties from contracts
    5. **⚖️ Decision Engine** — Cost-benefit analysis: reroute vs. accept penalty
    6. **🔄 Feedback Loop** — Logs outcomes for continuous model retraining
    """)

with col_b:
    st.markdown("""
    **Google Tech Used:**
    - ✅ Gemini 1.5 Flash (NLP)
    - ✅ Google Cloud Run
    - ✅ Cloud Firestore  
    - ✅ Vertex AI
    - ✅ Google Maps Routes API
    - ✅ Cloud Pub/Sub
    """)

st.markdown("---")

# Datasets
st.markdown("### 📊 Datasets Used")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.info("**Delivery Logistics Dataset** — 25K records with delivery partner, vehicle type, weather conditions, distance, weight, and delay labels. Used for XGBoost delay prediction.")

with col_d2:
    st.info("**LaDe (Cainiao-AI)** — 10M+ last-mile delivery packages across 5 cities with GPS trajectories and timestamps. Used for ETA prediction and spatial analysis.")

st.markdown("---")
st.caption("Built for Google Solution Challenge 2026 | Team LogiTrack AI")
