import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Ask LogiTrack AI", page_icon="🤖", layout="wide")

ML_URL = os.environ.get("ML_ENGINE_URL", "https://logitrackai.onrender.com")

# ─── Custom CSS ───
st.markdown("""
<style>
    .chat-bubble {
        padding: 1rem 1.2rem; border-radius: 16px; margin-bottom: 0.6rem;
        line-height: 1.6; font-size: 14px;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; margin-left: 15%; border-bottom-right-radius: 4px;
    }
    .ai-bubble {
        background: #1e1e2e; border: 1px solid #333; color: #e0e0e0;
        margin-right: 15%; border-bottom-left-radius: 4px;
    }
    .tool-badge {
        display: inline-block; background: rgba(102,126,234,0.15); color: #667eea;
        border-radius: 20px; padding: 2px 10px; font-size: 11px; font-weight: 600;
        margin-right: 6px; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Header ───
st.markdown("## 🤖 Ask LogiTrack AI")
st.markdown("Powered by **Gemini 2.0 Flash** + **6 ML Models** — ask anything about your supply chain in plain English.")

# ─── Capabilities ───
with st.expander("💡 What can I ask?", expanded=len(st.session_state.chat_history) == 0):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📦 Delay Prediction**")
        st.markdown("_What's the delay risk for a 500kg shipment from Mumbai to Delhi in rain?_")
        st.markdown("**🔍 Anomaly Detection**")
        st.markdown("_Scan my fleet for anomalies_")
    with col2:
        st.markdown("**🚚 Transport Optimization**")
        st.markdown("_Best transport mode for 1200km, 300kg cargo?_")
        st.markdown("**⏱️ ETA Prediction**")
        st.markdown("_How long for a 350km delivery from Bangalore at 2 PM?_")
    with col3:
        st.markdown("**🧪 What-If Simulation**")
        st.markdown("_What if a port strike hits west India at 80% severity?_")
        st.markdown("**🧠 Explainability**")
        st.markdown("_Why would a stormy express shipment be delayed?_")

st.markdown("---")

# ─── Example Buttons ───
EXAMPLES = [
    "What's the delay risk for a 500kg electronics shipment from Mumbai to Delhi in stormy weather?",
    "Which transport mode is cheapest for 1200km, 300kg cargo?",
    "What happens if a severe weather disruption hits north India at 80% severity?",
    "Scan the fleet for anomalies",
    "Explain why a rainy express shipment from east India gets delayed",
    "Predict ETA for a 350km delivery in Bangalore at 3 PM",
    "Auto-reroute critical shipments in the fleet",
    "What's the best green transport option for 800km, 200kg?",
    "How does distance affect delay probability?",
]

cols = st.columns(3)
for i, q in enumerate(EXAMPLES[:3]):
    if cols[i].button(f"💬 {q[:50]}...", key=f"ex_{i}", use_container_width=True):
        st.session_state["pending_query"] = q

cols2 = st.columns(3)
for i, q in enumerate(EXAMPLES[3:6]):
    if cols2[i].button(f"💬 {q[:50]}...", key=f"ex2_{i}", use_container_width=True):
        st.session_state["pending_query"] = q

cols3 = st.columns(3)
for i, q in enumerate(EXAMPLES[6:9]):
    if cols3[i].button(f"💬 {q[:50]}...", key=f"ex3_{i}", use_container_width=True):
        st.session_state["pending_query"] = q

# ─── Info ───
st.info("""
**🤖 How it works:** Your question → **Gemini AI** understands intent & extracts parameters → 
calls the correct **ML model** (XGBoost/SHAP/Isolation Forest) → Gemini summarizes the result 
in expert-level natural language. Fully autonomous — no coding needed!
""")

# ─── Chat Input ───
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g. What's the delay risk for a 500kg shipment from Mumbai to Delhi in stormy weather?",
        label_visibility="collapsed",
    )
    col_submit, col_clear = st.columns([4, 1])
    with col_submit:
        submitted = st.form_submit_button("🚀 Ask LogiTrack AI", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

# Handle example button clicks
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")
    submitted = True

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

if submitted and user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Call Gemini-powered /chat endpoint
    with st.spinner("🧠 Gemini is analyzing your question and querying ML models..."):
        try:
            url = f"{ML_URL}/api/v1/ml/chat"
            r = requests.post(url, json={"message": user_input}, timeout=60)

            if r.ok:
                data = r.json()
                ai_msg = {
                    "role": "ai",
                    "content": data.get("response", "No response received."),
                    "tool_used": data.get("tool_used"),
                    "params": data.get("params_extracted"),
                    "raw": data.get("ml_data"),
                }
                st.session_state.chat_history.append(ai_msg)
            else:
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": f"❌ ML Engine returned {r.status_code}: {r.text[:200]}",
                    "tool_used": "Error",
                })
        except requests.exceptions.Timeout:
            st.session_state.chat_history.append({
                "role": "ai",
                "content": "⏳ ML Engine is warming up (Render cold start). Please try again in 30 seconds.",
                "tool_used": "Timeout",
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "ai",
                "content": f"❌ Connection error: {str(e)}",
                "tool_used": "Error",
            })
    st.rerun()

# ─── Display Chat History ───
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Tool badge
            tool = msg.get("tool_used")
            if tool:
                st.markdown(f'<span class="tool-badge">🔧 {tool}</span>', unsafe_allow_html=True)

            # Gemini response
            st.markdown(msg["content"])

            # Show extracted params
            if msg.get("params"):
                with st.expander("📡 Parameters Extracted by Gemini"):
                    st.json(msg["params"])

            # Show raw ML data
            if msg.get("raw"):
                with st.expander("📋 Raw ML Engine Response"):
                    st.json(msg["raw"])

            st.markdown("---")

# ─── Footer ───
st.caption("🤖 Ask LogiTrack AI — Gemini 2.0 Flash + XGBoost · LightGBM · Isolation Forest · SHAP | Powered by Google AI")
