import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Ask LogiTrack AI", page_icon="🤖", layout="wide")

ML_URL = os.environ.get("ML_ENGINE_URL", "https://logitrackai.onrender.com")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

st.markdown("""
<style>
    .chat-msg { padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    .user-msg { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .ai-msg { background: #1e1e2e; border: 1px solid #333; color: #e0e0e0; }
    .result-card { background: #1a1a2e; border: 1px solid #444; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
    .result-card h4 { color: #667eea; margin: 0 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🤖 Ask LogiTrack AI")
st.markdown("Ask questions in plain English — powered by **Gemini + ML models**.")

EXAMPLE_QUERIES = [
    "What's the delay risk for a 500kg electronics shipment from Mumbai to Delhi in stormy weather?",
    "Which transport mode is cheapest and greenest for 1200km, 300kg?",
    "Predict ETA for a 50km delivery in Delhi at 2 PM",
    "What happens if a weather disruption hits north India with 80% severity?",
    "Explain why a stormy express shipment gets delayed",
]

st.markdown("**Try these examples:**")
cols = st.columns(3)
for idx, q in enumerate(EXAMPLE_QUERIES[:3]):
    if cols[idx].button(q[:50] + "...", key=f"ex_{idx}", use_container_width=True):
        st.session_state["user_query"] = q

cols2 = st.columns(2)
for idx, q in enumerate(EXAMPLE_QUERIES[3:]):
    if cols2[idx].button(q[:55] + "...", key=f"ex2_{idx}", use_container_width=True):
        st.session_state["user_query"] = q

st.markdown("---")

user_input = st.text_input(
    "Your question:",
    value=st.session_state.get("user_query", ""),
    placeholder="e.g. What's the delay risk for a heavy shipment from Mumbai to Delhi?",
    key="query_input"
)

def parse_and_route(query):
    """Route the user query to the correct ML endpoint based on keywords."""
    q = query.lower()

    if any(w in q for w in ["delay risk", "predict delay", "will it be delayed", "delay probability"]):
        partner = "delhivery"
        pkg = "electronics"
        vehicle = "truck"
        mode = "express" if "express" in q else "standard"
        region = "north"
        weather = "clear"
        distance = 500
        weight = 10

        for r in ["north", "south", "east", "west", "central"]:
            if r in q:
                region = r
                break
        for w in ["stormy", "rainy", "foggy", "clear", "extreme heat"]:
            if w in q:
                weather = w
                break
        import re
        dist_match = re.search(r'(\d+)\s*km', q)
        if dist_match:
            distance = int(dist_match.group(1))
        weight_match = re.search(r'(\d+)\s*kg', q)
        if weight_match:
            weight = int(weight_match.group(1))
        if "mumbai" in q and "delhi" in q:
            distance = 1400
            region = "north"
        if "heavy" in q:
            weight = max(weight, 500)

        payload = {
            "delivery_partner": partner, "package_type": pkg,
            "vehicle_type": vehicle, "delivery_mode": mode,
            "region": region, "weather_condition": weather,
            "distance_km": distance, "package_weight_kg": weight
        }
        return "predict-delay", payload

    elif any(w in q for w in ["eta", "how long", "delivery time", "estimated time"]):
        distance = 50
        hour = 14
        city = "unknown"
        import re
        dist_match = re.search(r'(\d+)\s*km', q)
        if dist_match:
            distance = int(dist_match.group(1))
        hour_match = re.search(r'(\d{1,2})\s*(am|pm|AM|PM)', q)
        if hour_match:
            h = int(hour_match.group(1))
            if "pm" in hour_match.group(2).lower() and h != 12:
                h += 12
            hour = h
        for c in ["delhi", "mumbai", "bangalore", "chennai", "kolkata"]:
            if c in q:
                city = c.capitalize()
                break

        payload = {"distance_km": distance, "hour": hour, "city": city, "day_of_week": 3}
        return "predict-eta", payload

    elif any(w in q for w in ["transport mode", "cheapest", "greenest", "road vs", "rail vs", "optimize transport"]):
        distance = 1200
        weight = 300
        deadline = 48
        priority = "balanced"
        import re
        dist_match = re.search(r'(\d+)\s*km', q)
        if dist_match:
            distance = int(dist_match.group(1))
        weight_match = re.search(r'(\d+)\s*kg', q)
        if weight_match:
            weight = int(weight_match.group(1))
        if "green" in q:
            priority = "green"
        elif "cheap" in q or "cost" in q:
            priority = "cost"
        elif "fast" in q or "speed" in q or "urgent" in q:
            priority = "speed"

        payload = {
            "distance_km": distance, "weight_kg": weight,
            "deadline_hours": deadline, "priority": priority, "weather_severity": 0.2
        }
        return "optimize-transport", payload

    elif any(w in q for w in ["what if", "what happens", "disruption", "simulate", "scenario"]):
        dtype = "weather"
        region = "north"
        severity = 0.7
        for d in ["weather", "port_congestion", "highway_closure", "strike"]:
            if d.replace("_", " ") in q:
                dtype = d
                break
        for r in ["north", "south", "east", "west", "central"]:
            if r in q:
                region = r
                break
        import re
        sev_match = re.search(r'(\d+)%', q)
        if sev_match:
            severity = int(sev_match.group(1)) / 100

        payload = {
            "disruption_type": dtype, "affected_region": region,
            "severity": severity, "fleet_size": 20
        }
        return "whatif", payload

    elif any(w in q for w in ["explain", "why", "reason", "factors", "shap"]):
        payload = {
            "delivery_partner": "delhivery", "package_type": "electronics",
            "vehicle_type": "truck", "delivery_mode": "express",
            "region": "north", "weather_condition": "stormy",
            "distance_km": 1400, "package_weight_kg": 25
        }
        return "explain-delay", payload

    elif any(w in q for w in ["anomaly", "anomalies", "outlier", "suspicious"]):
        payload = {"fleet_size": 25}
        return "anomaly-detect-batch", payload

    else:
        return None, None


def format_response(endpoint, result):
    """Format the ML response into a readable answer."""

    if endpoint == "predict-delay":
        risk = result.get("risk_level", "UNKNOWN")
        prob = result.get("delay_probability", 0)
        color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
        return f"""
**{color} Delay Prediction: {risk} Risk**

| Metric | Value |
|---|---|
| Delayed? | {"Yes" if result.get("is_delayed") else "No"} |
| Probability | {prob:.1%} |
| Risk Level | {risk} |

{"This shipment has a very high chance of being delayed. Consider rerouting or switching to express air cargo." if prob > 0.7 else "This shipment looks on track with acceptable risk levels."}
"""

    elif endpoint == "predict-eta":
        eta = result.get("estimated_time_mins", 0)
        return f"""
**Estimated Delivery Time: {eta:.0f} minutes** ({eta/60:.1f} hours)

| Detail | Value |
|---|---|
| Distance | {result.get('distance_km')} km |
| City | {result.get('city')} |
| Model | {result.get('model', 'XGBoost')} |
"""

    elif endpoint == "optimize-transport":
        rec = result.get("recommended", {})
        savings = result.get("savings", {})
        alts = result.get("alternatives", [])
        table = "| Mode | Cost (₹) | Time (hrs) | CO₂ (kg) | Meets Deadline |\n|---|---|---|---|---|\n"
        for a in alts[:5]:
            table += f"| {a['mode']} | ₹{a['total_cost_inr']:,} | {a['travel_time_hrs']}h | {a['co2_emissions_kg']} kg | {'✅' if a.get('meets_deadline') else '❌'} |\n"

        return f"""
**Recommended: {rec.get('mode', 'N/A')}** — ₹{rec.get('total_cost_inr', 0):,}, {rec.get('co2_emissions_kg', 0)} kg CO₂

{table}

**Savings vs Air Cargo:** ₹{savings.get('cost_saving_inr', 0):,} saved, {savings.get('co2_saving_kg', 0)} kg CO₂ reduced.
"""

    elif endpoint == "whatif":
        summary = result.get("summary", {})
        return f"""
**What-If Simulation Results**

| Metric | Before | After Disruption |
|---|---|---|
| Avg Delay Risk | {summary.get('baseline_avg_risk', 0):.1%} | {summary.get('disrupted_avg_risk', 0):.1%} |
| High Risk Count | {summary.get('baseline_high_risk', 0)} | {summary.get('disrupted_high_risk', 0)} |
| Newly At Risk | — | {summary.get('newly_at_risk_count', 0)} |

The disruption would put **{summary.get('newly_at_risk_count', 0)} additional shipments** at risk.
"""

    elif endpoint == "explain-delay":
        factors = result.get("top_factors", [])
        explanation = result.get("explanation", "")
        table = "| Factor | Value | Impact |\n|---|---|---|\n"
        for f in factors:
            table += f"| {f['feature']} | {f['value']} | {f['impact']} |\n"
        return f"""
**Prediction: {result.get('prediction', 'N/A')}** ({result.get('probability', 0):.1%} probability)

{table}

{explanation}
"""

    elif endpoint == "anomaly-detect-batch":
        return f"""
**Fleet Anomaly Scan Results**

| Metric | Value |
|---|---|
| Shipments Scanned | {result.get('total_scanned', 0)} |
| Anomalies Found | {result.get('total_anomalies', 0)} |
| Anomaly Rate | {result.get('anomaly_rate', 0):.1%} |
"""

    else:
        return f"```json\n{json.dumps(result, indent=2)}\n```"


if st.button("Ask LogiTrack AI", type="primary", use_container_width=True) and user_input:
    st.markdown(f'<div class="chat-msg user-msg">{user_input}</div>', unsafe_allow_html=True)

    endpoint, payload = parse_and_route(user_input)

    if endpoint is None:
        st.markdown('<div class="chat-msg ai-msg">I couldn\'t understand that question. Try asking about delay risk, ETA, transport optimization, disruption scenarios, or anomaly detection.</div>', unsafe_allow_html=True)
    else:
        with st.spinner(f"Querying ML engine ({endpoint})..."):
            try:
                url = f"{ML_URL}/api/v1/ml/{endpoint}"
                r = requests.post(url, json=payload, timeout=30)

                if r.ok:
                    result = r.json()
                    formatted = format_response(endpoint, result)
                    st.markdown(f'<div class="chat-msg ai-msg">', unsafe_allow_html=True)
                    st.markdown(formatted)
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("Raw API Response"):
                        st.json(result)

                    with st.expander("Request Sent"):
                        st.code(f"POST {url}\n\n{json.dumps(payload, indent=2)}", language="json")
                else:
                    st.error(f"ML Engine returned {r.status_code}: {r.text[:200]}")

            except Exception as e:
                st.error(f"Connection error: {str(e)}")

st.markdown("---")
st.caption("Ask LogiTrack AI — Natural language interface to ML models | Powered by Gemini + XGBoost | LogiTrack AI")
