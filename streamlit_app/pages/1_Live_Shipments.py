import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import random

st.set_page_config(page_title="Live Shipments", page_icon="🗺️", layout="wide")

st.markdown("## 🗺️ Live Shipment Tracking")
st.markdown("Real-time view of active shipments across India with ML-predicted risk levels.")

# Simulated shipment data representing India logistics
shipments = pd.DataFrame([
    {"id": "SHP-001", "origin": "Mumbai", "destination": "Delhi", "lat": 23.26, "lng": 77.41, "status": "IN_TRANSIT", 
     "partner": "Delhivery", "vehicle": "Truck", "weather": "Clear", "distance_km": 1400, "weight_kg": 25.0,
     "delay_risk": 0.82, "risk_level": "HIGH", "eta": "Apr 22, 10:00 AM"},
    {"id": "SHP-002", "origin": "Bangalore", "destination": "Chennai", "lat": 12.97, "lng": 77.59, "status": "IN_TRANSIT",
     "partner": "Shadowfax", "vehicle": "EV Van", "weather": "Rainy", "distance_km": 350, "weight_kg": 8.0,
     "delay_risk": 0.91, "risk_level": "HIGH", "eta": "Apr 21, 6:00 PM"},
    {"id": "SHP-003", "origin": "Kolkata", "destination": "Patna", "lat": 23.81, "lng": 86.44, "status": "IN_TRANSIT",
     "partner": "XpressBees", "vehicle": "Bike", "weather": "Clear", "distance_km": 580, "weight_kg": 3.0,
     "delay_risk": 0.15, "risk_level": "LOW", "eta": "Apr 21, 2:00 PM"},
    {"id": "SHP-004", "origin": "Jaipur", "destination": "Lucknow", "lat": 27.02, "lng": 76.36, "status": "DELAYED",
     "partner": "DHL", "vehicle": "Truck", "weather": "Stormy", "distance_km": 560, "weight_kg": 45.0,
     "delay_risk": 0.97, "risk_level": "CRITICAL", "eta": "Apr 22, 8:00 PM"},
    {"id": "SHP-005", "origin": "Hyderabad", "destination": "Vizag", "lat": 16.50, "lng": 79.52, "status": "IN_TRANSIT",
     "partner": "Delhivery", "vehicle": "EV Van", "weather": "Clear", "distance_km": 620, "weight_kg": 12.0,
     "delay_risk": 0.33, "risk_level": "LOW", "eta": "Apr 21, 4:00 PM"},
    {"id": "SHP-006", "origin": "Indore", "destination": "Bhopal", "lat": 23.17, "lng": 75.81, "status": "IN_TRANSIT",
     "partner": "Shadowfax", "vehicle": "Bike", "weather": "Clear", "distance_km": 190, "weight_kg": 2.5,
     "delay_risk": 0.08, "risk_level": "LOW", "eta": "Apr 21, 11:00 AM"},
    {"id": "SHP-007", "origin": "Guwahati", "destination": "Silchar", "lat": 25.59, "lng": 93.17, "status": "IN_TRANSIT",
     "partner": "DHL", "vehicle": "Truck", "weather": "Rainy", "distance_km": 340, "weight_kg": 30.0,
     "delay_risk": 0.72, "risk_level": "MEDIUM", "eta": "Apr 22, 1:00 PM"},
    {"id": "SHP-008", "origin": "Coimbatore", "destination": "Madurai", "lat": 10.36, "lng": 77.00, "status": "IN_TRANSIT",
     "partner": "XpressBees", "vehicle": "EV Van", "weather": "Clear", "distance_km": 210, "weight_kg": 6.0,
     "delay_risk": 0.12, "risk_level": "LOW", "eta": "Apr 21, 12:00 PM"},
])

# Sidebar filters
st.sidebar.markdown("### Filters")
risk_filter = st.sidebar.multiselect("Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
partner_filter = st.sidebar.multiselect("Delivery Partner", shipments["partner"].unique(), default=shipments["partner"].unique())

filtered = shipments[shipments["risk_level"].isin(risk_filter) & shipments["partner"].isin(partner_filter)]

# Summary stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shipments", len(filtered))
col2.metric("🔴 High/Critical Risk", len(filtered[filtered["risk_level"].isin(["HIGH", "CRITICAL"])]))
col3.metric("🟢 On Track", len(filtered[filtered["risk_level"] == "LOW"]))
col4.metric("Avg Delay Risk", f"{filtered['delay_risk'].mean():.0%}")

# Map
st.markdown("### India Logistics Map")
m = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles="CartoDB dark_matter")

color_map = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}
icon_map = {"LOW": "ok-sign", "MEDIUM": "warning-sign", "HIGH": "alert", "CRITICAL": "fire"}

for _, row in filtered.iterrows():
    color = color_map.get(row["risk_level"], "blue")
    popup_html = f"""
    <div style="font-family: Inter, sans-serif; width: 220px;">
        <b style="font-size:14px;">{row['id']}</b><br>
        <b>{row['origin']}</b> → <b>{row['destination']}</b><br>
        <hr style="margin:4px 0;">
        Partner: {row['partner']}<br>
        Vehicle: {row['vehicle']}<br>
        Weather: {row['weather']}<br>
        Distance: {row['distance_km']} km<br>
        <hr style="margin:4px 0;">
        <span style="color: {color}; font-weight:bold;">
            Delay Risk: {row['delay_risk']:.0%} ({row['risk_level']})
        </span><br>
        ETA: {row['eta']}
    </div>
    """
    folium.Marker(
        [row["lat"], row["lng"]],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color=color, icon="truck", prefix="fa"),
    ).add_to(m)

st_folium(m, width=None, height=500, use_container_width=True)

# Shipment Table
st.markdown("### 📋 Shipment Details")

def color_risk(val):
    colors = {"LOW": "#38ef7d", "MEDIUM": "#f5a623", "HIGH": "#f5576c", "CRITICAL": "#d63031"}
    return f"background-color: {colors.get(val, '#fff')}; color: white; font-weight: bold; border-radius: 4px; padding: 2px 8px;"

display_df = filtered[["id", "origin", "destination", "partner", "vehicle", "weather", "distance_km", "delay_risk", "risk_level", "eta"]].copy()
display_df["delay_risk"] = display_df["delay_risk"].apply(lambda x: f"{x:.0%}")
display_df.columns = ["ID", "Origin", "Destination", "Partner", "Vehicle", "Weather", "Distance (km)", "Delay Risk", "Risk", "ETA"]

st.dataframe(display_df, use_container_width=True, hide_index=True)
