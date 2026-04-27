import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Smart Alerts", page_icon="🚨", layout="wide")

ML_URL = os.environ.get("ML_ENGINE_URL", "https://logitrackai.onrender.com")

st.markdown("""
<style>
    .alert-critical { background: linear-gradient(135deg, #d63031, #e17055); color: white; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    .alert-high { background: linear-gradient(135deg, #e17055, #fdcb6e); color: white; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    .alert-medium { background: linear-gradient(135deg, #fdcb6e, #00b894); color: #333; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    .alert-low { background: linear-gradient(135deg, #00b894, #00cec9); color: white; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    .alert-header { font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
    .alert-detail { font-size: 0.9rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚨 Smart Alerts Dashboard")
st.markdown("Auto-generated alerts from ML models — anomalies, disruptions, and route risks detected in real-time.")

@st.cache_data(ttl=120)
def run_alert_scan():
    """Run all ML models and generate alerts."""
    alerts = []
    timestamp = datetime.now().strftime("%H:%M:%S")

    # 1. Fleet anomaly scan
    try:
        r = requests.post(f"{ML_URL}/api/v1/ml/anomaly-detect-batch", json={"fleet_size": 30}, timeout=20)
        if r.ok:
            data = r.json()
            total = data.get("total_anomalies", 0)
            rate = data.get("anomaly_rate", 0)
            if total > 0:
                anomalies = data.get("anomalies", [])
                for anom in anomalies[:5]:
                    alerts.append({
                        "time": timestamp,
                        "type": "ANOMALY",
                        "severity": "HIGH" if anom.get("anomaly_score", 0) < -0.3 else "MEDIUM",
                        "title": f"Anomaly: {anom.get('shipment_id', 'Unknown')}",
                        "detail": f"Score: {anom.get('anomaly_score', 0):.3f} | Risk: {anom.get('risk_level', 'N/A')} | Reasons: {', '.join(anom.get('anomaly_reasons', [])[:2])}",
                        "source": "Isolation Forest"
                    })
                alerts.append({
                    "time": timestamp,
                    "type": "FLEET",
                    "severity": "CRITICAL" if rate > 0.3 else "HIGH" if rate > 0.15 else "MEDIUM",
                    "title": f"Fleet Alert: {total} anomalies in {data.get('total_scanned', 30)} shipments ({rate:.0%})",
                    "detail": f"Anomaly rate is {'CRITICAL — immediate attention needed' if rate > 0.3 else 'elevated — review flagged shipments'}",
                    "source": "Batch Anomaly Detector"
                })
    except Exception:
        pass

    # 2. Disruption impact scan
    regions = ["north", "south", "east", "west"]
    for region in regions:
        try:
            r = requests.post(f"{ML_URL}/api/v1/ml/whatif", json={
                "disruption_type": "weather", "affected_region": region,
                "severity": 0.7, "fleet_size": 15
            }, timeout=15)
            if r.ok:
                data = r.json()
                summary = data.get("summary", {})
                newly = summary.get("newly_at_risk_count", 0)
                if newly > 0:
                    alerts.append({
                        "time": timestamp,
                        "type": "DISRUPTION",
                        "severity": "CRITICAL" if newly > 5 else "HIGH" if newly > 2 else "MEDIUM",
                        "title": f"Weather Disruption: {region.upper()} India — {newly} shipments newly at risk",
                        "detail": f"Avg risk increased from {summary.get('baseline_avg_risk', 0):.0%} → {summary.get('disrupted_avg_risk', 0):.0%}",
                        "source": "What-If Simulator"
                    })
        except Exception:
            pass

    # 3. High-risk route check
    test_routes = [
        {"delivery_partner": "delhivery", "package_type": "electronics", "vehicle_type": "truck",
         "delivery_mode": "express", "region": "north", "weather_condition": "stormy",
         "distance_km": 1400, "package_weight_kg": 500},
        {"delivery_partner": "shadowfax", "package_type": "fragile", "vehicle_type": "ev_van",
         "delivery_mode": "standard", "region": "south", "weather_condition": "rainy",
         "distance_km": 800, "package_weight_kg": 50},
    ]
    for route in test_routes:
        try:
            r = requests.post(f"{ML_URL}/api/v1/ml/predict-delay", json=route, timeout=10)
            if r.ok:
                data = r.json()
                if data.get("delay_probability", 0) > 0.7:
                    alerts.append({
                        "time": timestamp,
                        "type": "DELAY",
                        "severity": data.get("risk_level", "HIGH"),
                        "title": f"High Delay Risk: {route['region'].upper()} {route['delivery_mode']} shipment ({data.get('delay_probability', 0):.0%})",
                        "detail": f"Partner: {route['delivery_partner']} | Weather: {route['weather_condition']} | Distance: {route['distance_km']}km",
                        "source": "XGBoost Delay Predictor"
                    })
        except Exception:
            pass

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 9))

    return alerts


with st.spinner("🔍 Scanning fleet with ML models..."):
    alerts = run_alert_scan()

# Summary bar
col1, col2, col3, col4 = st.columns(4)
critical = len([a for a in alerts if a["severity"] == "CRITICAL"])
high = len([a for a in alerts if a["severity"] == "HIGH"])
medium = len([a for a in alerts if a["severity"] == "MEDIUM"])
col1.metric("Total Alerts", len(alerts))
col2.metric("🔴 Critical", critical)
col3.metric("🟠 High", high)
col4.metric("🟡 Medium", medium)

st.markdown("---")

# Filter
severity_filter = st.multiselect("Filter by severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
type_filter = st.multiselect("Filter by type", ["ANOMALY", "FLEET", "DISRUPTION", "DELAY"], default=["ANOMALY", "FLEET", "DISRUPTION", "DELAY"])

filtered = [a for a in alerts if a["severity"] in severity_filter and a["type"] in type_filter]

# Display alerts
if not filtered:
    st.success("✅ No alerts matching your filters. All systems nominal.")
else:
    for alert in filtered:
        css_class = f"alert-{alert['severity'].lower()}"
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(alert["severity"], "⚪")
        type_icon = {"ANOMALY": "🔍", "FLEET": "🚛", "DISRUPTION": "⛈️", "DELAY": "⏱️"}.get(alert["type"], "📋")

        st.markdown(f"""
        <div class="{css_class}">
            <div class="alert-header">{icon} {type_icon} [{alert['type']}] {alert['title']}</div>
            <div class="alert-detail">{alert['detail']}</div>
            <div class="alert-detail" style="margin-top: 0.3rem; font-size: 0.8rem;">
                Source: {alert['source']} | {alert['time']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Alert table
if filtered:
    st.markdown("### 📋 Alert Log")
    df = pd.DataFrame(filtered)
    st.dataframe(df[["time", "severity", "type", "title", "source"]], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Smart Alerts powered by XGBoost + Isolation Forest + What-If Simulator | LogiTrack AI")
