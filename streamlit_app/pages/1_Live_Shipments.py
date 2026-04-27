import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import random

st.set_page_config(page_title="Live Shipments", page_icon="🗺️", layout="wide")

st.markdown("## 🗺️ Live Shipment Tracking")
st.markdown("Real-time view of active shipments across India with ML-predicted risk levels.")

# Generate 250 realistic live shipments across India dynamically
@st.cache_data(ttl=300) # Cache for 5 mins to simulate "live" updating
def generate_live_shipments(n=250):
    import numpy as np
    
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Indore", "Guwahati", "Coimbatore"]
    partners = ["Delhivery", "Shadowfax", "XpressBees", "DHL", "BlueDart", "Amazon Shipping"]
    vehicles = ["Truck", "EV Van", "Heavy Trailer", "Bike", "Train Freight"]
    weather_conds = ["Clear", "Rainy", "Stormy", "Foggy", "Extreme Heat"]
    
    data = []
    for i in range(n):
        orig = random.choice(cities)
        dest = random.choice([c for c in cities if c != orig])
        
        # Approximate India Lat/Lng bounds (very rough box)
        lat = np.random.uniform(10.0, 28.0)
        lng = np.random.uniform(70.0, 88.0)
        
        distance = int(np.random.uniform(50, 2500))
        weight = round(np.random.uniform(1.0, 5000.0), 1)
        weather = random.choices(weather_conds, weights=[60, 20, 5, 10, 5])[0]
        
        # Dynamic risk calculation heuristic
        risk = np.random.beta(a=2, b=5) # Skew towards lower risk naturally
        if weather in ["Stormy", "Foggy"]:
            risk += np.random.uniform(0.2, 0.5)
        if distance > 1500:
            risk += np.random.uniform(0.1, 0.3)
            
        risk = min(max(risk, 0.01), 0.99)
        
        if risk < 0.3:
            risk_lvl = "LOW"
        elif risk < 0.6:
            risk_lvl = "MEDIUM"
        elif risk < 0.85:
            risk_lvl = "HIGH"
        else:
            risk_lvl = "CRITICAL"
            
        status = "IN_TRANSIT" if risk_lvl in ["LOW", "MEDIUM"] else random.choice(["IN_TRANSIT", "DELAYED"])
        
        data.append({
            "id": f"SHP-{random.randint(10000, 99999)}",
            "origin": orig,
            "destination": dest,
            "lat": lat,
            "lng": lng,
            "status": status,
            "partner": random.choice(partners),
            "vehicle": random.choice(vehicles),
            "weather": weather,
            "distance_km": distance,
            "weight_kg": weight,
            "delay_risk": risk,
            "risk_level": risk_lvl,
            "eta": f"Apr {random.randint(21, 26)}, {random.randint(8,20)}:00"
        })
    return pd.DataFrame(data)

shipments = generate_live_shipments(250)

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
