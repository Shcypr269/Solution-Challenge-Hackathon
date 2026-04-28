import streamlit as st
import requests
import json
import os
import re

st.set_page_config(page_title="Ask LogiTrack AI", page_icon="🤖", layout="wide")

ML_URL = os.environ.get("ML_ENGINE_URL", "https://logitrackai.onrender.com")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ─── Custom CSS ───
st.markdown("""
<style>
    .chat-container { max-width: 800px; margin: 0 auto; }
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
    .thought-tag {
        display: inline-block; background: rgba(102,126,234,0.15); color: #667eea;
        border-radius: 20px; padding: 2px 10px; font-size: 11px; font-weight: 600;
        margin-right: 6px; margin-bottom: 6px;
    }
    .capability-card {
        background: #1a1a2e; border: 1px solid #333; border-radius: 12px;
        padding: 1rem; cursor: pointer; transition: border-color 0.2s;
    }
    .capability-card:hover { border-color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Header ───
st.markdown("## 🤖 Ask LogiTrack AI")
st.markdown("Chat with our ML engine — ask about delay risks, transport costs, disruptions, anomalies, and more.")

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
        st.markdown("_Best transport mode for 1200km, 300kg, under ₹50k?_")
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
]

cols = st.columns(3)
for i, q in enumerate(EXAMPLES[:3]):
    if cols[i].button(f"💬 {q[:55]}...", key=f"ex_{i}", use_container_width=True):
        st.session_state["pending_query"] = q

cols2 = st.columns(3)
for i, q in enumerate(EXAMPLES[3:]):
    if cols2[i].button(f"💬 {q[:55]}...", key=f"ex2_{i}", use_container_width=True):
        st.session_state["pending_query"] = q

# ─── Chat History ───
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        tags = msg.get("tags", [])
        tags_html = "".join(f'<span class="thought-tag">{t}</span>' for t in tags)
        st.markdown(f'<div class="chat-bubble ai-bubble">{tags_html}<br>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("table"):
            st.markdown(msg["table"])
        if msg.get("raw"):
            with st.expander("📋 Raw API Response"):
                st.json(msg["raw"])

# ─── Parser ───
CITY_COORDS = {
    "mumbai": "west", "delhi": "north", "bangalore": "south", "chennai": "south",
    "kolkata": "east", "hyderabad": "south", "pune": "west", "ahmedabad": "west",
    "jaipur": "north", "lucknow": "north", "bhopal": "central", "patna": "east",
}

def parse_query(query):
    q = query.lower()

    # Extract common parameters
    dist = int(m.group(1)) if (m := re.search(r'(\d+)\s*km', q)) else None
    weight = int(m.group(1)) if (m := re.search(r'(\d+)\s*kg', q)) else None
    severity = int(m.group(1)) / 100 if (m := re.search(r'(\d+)\s*%', q)) else None
    hour = None
    if (m := re.search(r'(\d{1,2})\s*(am|pm)', q, re.I)):
        h = int(m.group(1))
        if m.group(2).lower() == "pm" and h != 12: h += 12
        if m.group(2).lower() == "am" and h == 12: h = 0
        hour = h

    # Detect region from cities
    region = None
    for city, reg in CITY_COORDS.items():
        if city in q:
            region = reg
            break
    for r in ["north", "south", "east", "west", "central"]:
        if r in q:
            region = r
            break

    # Detect weather
    weather = "clear"
    for w in ["stormy", "rainy", "foggy", "cold", "extreme heat", "rain", "storm", "fog"]:
        if w in q:
            weather = w.replace("rain", "rainy").replace("storm", "stormy").replace("fog", "foggy")
            break

    # ─── Route to endpoint ───

    # 1. Delay Prediction
    if any(w in q for w in ["delay risk", "predict delay", "will it be delayed", "delay probability", "delay chance", "risk of delay"]):
        if "mumbai" in q and "delhi" in q: dist = dist or 1400
        if "heavy" in q: weight = max(weight or 0, 500)
        return "predict-delay", {
            "delivery_partner": "delhivery",
            "package_type": "electronics" if "electronic" in q else "general",
            "vehicle_type": "ev van" if "ev" in q else "truck",
            "delivery_mode": "express" if "express" in q else "standard",
            "region": region or "north",
            "weather_condition": weather,
            "distance_km": dist or 500,
            "package_weight_kg": weight or 10,
        }, ["Delay Prediction", f"📏 {dist or 500}km", f"📦 {weight or 10}kg", f"🌤️ {weather}"]

    # 2. ETA
    if any(w in q for w in ["eta", "how long", "delivery time", "estimated time", "predict eta"]):
        city = "unknown"
        for c in CITY_COORDS:
            if c in q: city = c.capitalize(); break
        return "predict-eta", {
            "distance_km": dist or 50, "hour": hour or 14,
            "city": city, "day_of_week": 3,
        }, ["ETA Prediction", f"📏 {dist or 50}km", f"🏙️ {city}"]

    # 3. Transport Optimization
    if any(w in q for w in ["transport mode", "cheapest", "greenest", "optimize transport", "best mode", "road vs", "rail vs", "compare"]):
        priority = "balanced"
        if "green" in q or "eco" in q: priority = "green"
        elif "cheap" in q or "cost" in q or "budget" in q: priority = "cost"
        elif "fast" in q or "speed" in q or "urgent" in q: priority = "speed"
        return "optimize-transport", {
            "distance_km": dist or 1200, "weight_kg": weight or 300,
            "deadline_hours": 48, "priority": priority,
        }, ["Transport Optimizer", f"📏 {dist or 1200}km", f"📦 {weight or 300}kg", f"⚖️ {priority}"]

    # 4. What-If Simulation
    if any(w in q for w in ["what if", "what happens", "disruption", "simulate", "scenario", "strike", "flood", "bandh"]):
        dtype = "weather"
        for d in ["port_congestion", "port congestion", "highway_closure", "highway closure", "strike", "bandh"]:
            if d.replace("_", " ") in q:
                dtype = d.replace(" ", "_"); break
        if "flood" in q or "cyclone" in q: dtype = "weather"
        return "whatif", {
            "disruption_type": dtype, "affected_region": region or "north",
            "severity": severity or 0.7, "fleet_size": 20, "inject_region": True,
        }, ["What-If Simulator", f"💥 {dtype.replace('_',' ')}", f"📍 {region or 'north'}", f"⚡ {severity or 0.7}"]

    # 5. Explainability
    if any(w in q for w in ["explain", "why", "reason", "factors", "shap", "understand"]):
        return "explain-delay", {
            "delivery_partner": "delhivery", "package_type": "electronics",
            "vehicle_type": "truck", "delivery_mode": "express" if "express" in q else "standard",
            "region": region or "north", "weather_condition": weather,
            "distance_km": dist or 1400, "package_weight_kg": weight or 25,
        }, ["Explainability (SHAP)", f"🌤️ {weather}", f"📍 {region or 'north'}"]

    # 6. Anomaly Detection
    if any(w in q for w in ["anomaly", "anomalies", "outlier", "suspicious", "scan", "fleet scan", "fleet health"]):
        return "anomaly-detect-batch", {
            "fleet_size": 25,
        }, ["Fleet Anomaly Scan", "📦 25 shipments"]

    return None, None, []


# ─── Response Formatter ───
def format_response(endpoint, result):
    """Format the ML API response into rich markdown."""
    table = ""
    text = ""

    if endpoint == "predict-delay":
        prob = result.get("delay_probability", result.get("probability", 0))
        risk = result.get("risk_level", "UNKNOWN")
        delayed = result.get("is_delayed", prob > 0.5)
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")

        text = f"**{icon} Delay Prediction: {risk} Risk ({prob:.1%})**\n\n"
        if delayed:
            text += "⚠️ This shipment has a **high chance of being delayed**. Consider switching to a faster mode, rerouting through a less congested corridor, or dispatching earlier."
        else:
            text += "✅ This shipment looks **on track** with acceptable risk levels. No immediate action needed."

        table = f"""
| Metric | Value |
|---|---|
| Prediction | {'🔴 Delayed' if delayed else '🟢 On Time'} |
| Probability | {prob:.1%} |
| Risk Level | {icon} {risk} |
| Confidence | {result.get('confidence', 'High')} |
"""

    elif endpoint == "predict-eta":
        eta = result.get("estimated_time_mins", 0)
        text = f"**⏱️ Estimated Delivery Time: {eta:.0f} minutes** ({eta/60:.1f} hours)\n\n"
        text += f"Based on current traffic patterns and historical data for **{result.get('city', 'the area')}**."
        table = f"""
| Detail | Value |
|---|---|
| ETA | {eta:.0f} mins ({eta/60:.1f} hrs) |
| Distance | {result.get('distance_km', '?')} km |
| City | {result.get('city', '?')} |
| Model | XGBoost Regressor |
"""

    elif endpoint == "optimize-transport":
        rec = result.get("recommended", {})
        savings = result.get("savings", {})
        alts = result.get("alternatives", [])
        mode = rec.get("mode", "N/A")
        cost = rec.get("total_cost_inr", 0)
        co2 = rec.get("co2_emissions_kg", 0)
        time = rec.get("travel_time_hrs", "?")

        text = f"**✅ Best Option: {mode}** — ₹{cost:,} · {time}h · {co2}kg CO₂\n\n"
        text += f"💰 **Saves ₹{savings.get('cost_saving_inr', 0):,}** vs air cargo | 🌿 **Saves {savings.get('co2_saving_kg', 0)} kg CO₂**"

        table = "| Mode | Cost (₹) | Time | CO₂ | Reliable | Deadline | Score |\n|---|---|---|---|---|---|---|\n"
        for a in alts[:6]:
            star = "⭐ " if a.get("mode_id") == rec.get("mode_id") else ""
            table += f"| {star}{a['mode']} | ₹{a['total_cost_inr']:,} | {a['travel_time_hrs']}h | {a['co2_emissions_kg']}kg | {round(a.get('reliability',0)*100)}% | {'✅' if a.get('meets_deadline') else '❌'} | {a.get('score',0):.3f} |\n"

    elif endpoint == "whatif":
        imp = result.get("impact_summary", {})
        text = f"**💥 Disruption Simulation Complete**\n\n"
        newly = imp.get("newly_at_risk", 0)
        penalty = imp.get("estimated_penalty_inr", 0)
        text += f"**{newly} shipments** would become newly at risk, with an estimated penalty of **₹{penalty:,}**."
        if newly > 5:
            text += " ⚠️ This is a significant impact — pre-position backup inventory and alert drivers in the affected zone."
        else:
            text += " Current fleet resilience is adequate."

        table = f"""
| Metric | Value |
|---|---|
| Total Analyzed | {imp.get('total_shipments_analyzed', 0)} |
| In Affected Region | {imp.get('shipments_in_affected_region', 0)} |
| Newly At Risk | {newly} |
| Avg Risk Increase | {imp.get('avg_delay_prob_increase', 0):.1%} |
| Est. Penalty | ₹{penalty:,} |
"""

    elif endpoint == "explain-delay":
        prob = result.get("probability", 0)
        factors = result.get("top_factors", [])
        explanation = result.get("explanation", "")

        icon = "🔴" if prob > 0.6 else "🟡" if prob > 0.3 else "🟢"
        text = f"**{icon} Prediction: {result.get('prediction', 'N/A')}** ({prob:.1%} probability)\n\n"
        text += f"💡 {explanation}" if explanation else ""

        if factors:
            table = "| Factor | Importance | Direction |\n|---|---|---|\n"
            for f in factors:
                arrow = "↑ Increases Risk" if f.get("direction") == "increases_risk" else "↓ Decreases Risk"
                color = "🔴" if f.get("direction") == "increases_risk" else "🟢"
                table += f"| {f['feature']} | {f.get('importance', 0)}% | {color} {arrow} |\n"

    elif endpoint == "anomaly-detect-batch":
        s = result.get("summary", {})
        text = f"**🔍 Fleet Scan Complete** — {s.get('total_shipments', 0)} shipments analyzed\n\n"
        anom = s.get("anomalies_detected", 0)
        rate = s.get("anomaly_rate", 0)
        if anom > 0:
            text += f"⚠️ Found **{anom} anomalies** ({rate:.1%} rate). {s.get('critical_alerts', 0)} critical + {s.get('high_alerts', 0)} high priority."
        else:
            text += "✅ Fleet is clean — no anomalies detected!"

        dist = result.get("risk_distribution", {})
        table = f"""
| Metric | Value |
|---|---|
| Total Scanned | {s.get('total_shipments', 0)} |
| Anomalies | {anom} |
| Anomaly Rate | {rate:.1%} |
| Critical | {dist.get('CRITICAL', 0)} |
| High | {dist.get('HIGH', 0)} |
| Medium | {dist.get('MEDIUM', 0)} |
| Low | {dist.get('LOW', 0)} |
"""

    else:
        text = "Here's the raw response from the ML engine:"
        table = f"```json\n{json.dumps(result, indent=2)}\n```"

    return text, table


# ─── Chat Input ───
user_input = st.chat_input("Ask LogiTrack AI anything about your supply chain...")

# Handle example button clicks
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="chat-bubble user-bubble">🧑 {user_input}</div>', unsafe_allow_html=True)

    # Parse and route
    endpoint, payload, tags = parse_query(user_input)

    if endpoint is None:
        ai_msg = {
            "role": "ai",
            "content": "🤔 I couldn't understand that question. I can help with:\n\n"
                       "• **Delay prediction** — _\"What's the delay risk for...\"_\n"
                       "• **ETA estimation** — _\"How long will a 350km delivery take?\"_\n"
                       "• **Transport optimization** — _\"Best mode for 1200km, 300kg\"_\n"
                       "• **Disruption simulation** — _\"What if a strike hits west India?\"_\n"
                       "• **Explainability** — _\"Why would this shipment be delayed?\"_\n"
                       "• **Anomaly detection** — _\"Scan the fleet for anomalies\"_",
            "tags": ["Unrecognized"],
        }
        st.session_state.chat_history.append(ai_msg)
        tags_html = "".join(f'<span class="thought-tag">{t}</span>' for t in ai_msg["tags"])
        st.markdown(f'<div class="chat-bubble ai-bubble">{tags_html}<br>{ai_msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.spinner(f"🧠 Querying ML engine → `{endpoint}`..."):
            try:
                url = f"{ML_URL}/api/v1/ml/{endpoint}"
                r = requests.post(url, json=payload, timeout=45)

                if r.ok:
                    result = r.json()
                    text, table = format_response(endpoint, result)

                    ai_msg = {"role": "ai", "content": text, "table": table, "tags": tags, "raw": result}
                    st.session_state.chat_history.append(ai_msg)

                    tags_html = "".join(f'<span class="thought-tag">{t}</span>' for t in tags)
                    st.markdown(f'<div class="chat-bubble ai-bubble">{tags_html}<br>{text}</div>', unsafe_allow_html=True)
                    if table:
                        st.markdown(table)
                    with st.expander("📋 Raw API Response"):
                        st.json(result)
                    with st.expander("📡 Request Details"):
                        st.code(f"POST {url}\n\n{json.dumps(payload, indent=2)}", language="json")
                else:
                    st.error(f"ML Engine returned {r.status_code}: {r.text[:300]}")
            except requests.exceptions.Timeout:
                st.warning("⏳ ML Engine is warming up (Render cold start). Please try again in 30 seconds.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# ─── Footer ───
st.markdown("---")
cols = st.columns([3, 1])
with cols[0]:
    st.caption("🤖 Ask LogiTrack AI — Natural language interface to 6 ML models | XGBoost · LightGBM · Isolation Forest · SHAP")
with cols[1]:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
