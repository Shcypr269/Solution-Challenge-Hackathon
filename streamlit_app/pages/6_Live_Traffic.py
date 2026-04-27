"""
Real-Time Traffic Monitor — Streamlit Page
Live traffic conditions across Indian logistics corridors using TomTom API.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.tomtom_traffic import (
    get_traffic_flow, calculate_route, get_traffic_incidents,
    scan_corridor_traffic, INDIA_CORRIDORS, API_KEY
)

st.set_page_config(page_title="Live Traffic", page_icon="🚦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .traffic-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #e94560 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.3);
    }
    .traffic-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .traffic-header p { font-size: 1rem; opacity: 0.8; font-weight: 300; }
    
    .corridor-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        color: #eee;
        transition: transform 0.2s;
    }
    .corridor-card:hover { transform: translateY(-2px); }
    .corridor-card.free { border-left: 4px solid #00b894; }
    .corridor-card.light { border-left: 4px solid #fdcb6e; }
    .corridor-card.moderate { border-left: 4px solid #e17055; }
    .corridor-card.severe { border-left: 4px solid #d63031; }
    .corridor-card.closed { border-left: 4px solid #6c5ce7; }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .mini-metric {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
    }
    .mini-metric span { font-weight: 600; }
    
    .route-box {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="traffic-header">
    <h1>🚦 Real-Time Traffic Monitor</h1>
    <p>Live traffic conditions across Indian logistics corridors — powered by TomTom Traffic APIs</p>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("TomTom API key not found. Set `TOMTOM_API_KEY` in `backend/.env`")
    st.stop()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🛣️ Corridor Monitor", "🗺️ Route Calculator", "⚠️ Live Incidents"])

# ═══════════════════════════════════════════
# TAB 1: Corridor Traffic Monitor
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### Indian Logistics Corridors — Live Status")
    
    col_ctrl, _ = st.columns([1, 2])
    with col_ctrl:
        selected_corridors = st.multiselect(
            "Select corridors to monitor",
            list(INDIA_CORRIDORS.keys()),
            default=["mumbai_delhi", "chennai_bangalore", "delhi_jaipur"],
            format_func=lambda x: INDIA_CORRIDORS[x]["name"]
        )
    
    if st.button("🔄 Refresh Traffic Data", use_container_width=True, type="primary"):
        for cid in selected_corridors:
            corridor = INDIA_CORRIDORS[cid]
            with st.spinner(f"Scanning {corridor['name']}..."):
                # Get traffic at origin and destination
                origin_flow = get_traffic_flow(corridor["origin"][0], corridor["origin"][1])
                dest_flow = get_traffic_flow(corridor["destination"][0], corridor["destination"][1])
                
                # Get route info
                route = calculate_route(
                    corridor["origin"],
                    corridor["destination"],
                    via=corridor.get("via") or None,
                    traffic=True
                )
            
            # Determine congestion level
            if origin_flow and dest_flow:
                avg_congestion = (origin_flow.congestion_ratio + dest_flow.congestion_ratio) / 2
                level = (
                    "closed" if origin_flow.road_closure or dest_flow.road_closure else
                    "severe" if avg_congestion > 0.7 else
                    "moderate" if avg_congestion > 0.4 else
                    "light" if avg_congestion > 0.15 else
                    "free"
                )
            else:
                avg_congestion = 0
                level = "free"
            
            st.markdown(f"""
            <div class="corridor-card {level}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="font-size:1.1rem;">{corridor['name']}</strong>
                        <span style="background:{'#00b894' if level=='free' else '#fdcb6e' if level=='light' else '#e17055' if level=='moderate' else '#d63031'};
                               color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; margin-left:8px;">
                            {level.upper()}
                        </span>
                    </div>
                </div>
                <div class="metric-row">
                    {'<div class="mini-metric">Distance: <span>' + str(route.distance_km) + ' km</span></div>' if route else ''}
                    {'<div class="mini-metric">ETA: <span>' + str(int(route.travel_time_mins)) + ' min</span></div>' if route else ''}
                    {'<div class="mini-metric">Traffic Delay: <span>+' + str(int(route.traffic_delay_mins)) + ' min</span></div>' if route else ''}
                    {'<div class="mini-metric">Origin Speed: <span>' + str(origin_flow.current_speed_kmh) + ' km/h</span></div>' if origin_flow else ''}
                    {'<div class="mini-metric">Dest Speed: <span>' + str(dest_flow.current_speed_kmh) + ' km/h</span></div>' if dest_flow else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click **Refresh Traffic Data** to fetch live conditions from TomTom API")
        
        # Show corridor list
        st.markdown("#### Available Corridors")
        for cid, corridor in INDIA_CORRIDORS.items():
            st.markdown(f"- **{corridor['name']}**: `{corridor['origin']}` → `{corridor['destination']}`")

# ═══════════════════════════════════════════
# TAB 2: Route Calculator
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Traffic-Aware Route Calculator")
    st.markdown("Calculate optimal truck routes with live traffic conditions")
    
    col_o, col_d = st.columns(2)
    
    # Indian city coordinates
    cities = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Kolkata": (22.5726, 88.3639),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Ahmedabad": (23.0225, 72.5714),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462),
        "Visakhapatnam": (17.6868, 83.2185),
        "Indore": (22.7196, 75.8577),
        "JNPT Port": (18.9543, 72.9486),
        "Mundra Port": (22.8386, 69.7193),
    }
    
    with col_o:
        origin_city = st.selectbox("Origin City", list(cities.keys()), index=0, key="route_origin")
    with col_d:
        dest_city = st.selectbox("Destination City", list(cities.keys()), index=6, key="route_dest")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_traffic = st.checkbox("Use live traffic", value=True)
    with col_opt2:
        vehicle_weight = st.number_input("Vehicle weight (kg)", 0, 50000, 0, help="0 = car mode, >0 = truck mode")
    
    if st.button("🚚 Calculate Route", use_container_width=True, type="primary", key="calc_route"):
        origin = cities[origin_city]
        dest = cities[dest_city]
        
        with st.spinner(f"Calculating route {origin_city} → {dest_city}..."):
            route = calculate_route(
                origin, dest,
                traffic=use_traffic,
                vehicle_weight_kg=vehicle_weight if vehicle_weight > 0 else None
            )
        
        if route:
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Distance", f"{route.distance_km} km")
            rc2.metric("Travel Time", f"{route.travel_time_mins:.0f} min")
            rc3.metric("Traffic Delay", f"+{route.traffic_delay_mins:.0f} min")
            rc4.metric("Avg Speed", f"{route.distance_km / max(route.travel_time_mins/60, 0.1):.0f} km/h")
            
            st.markdown(f"""
            <div class="route-box">
                <strong>{origin_city} → {dest_city}</strong><br>
                <span style="font-size:0.9rem; opacity:0.8;">{route.summary}</span><br>
                <span style="font-size:0.85rem; opacity:0.7;">
                    Departure: {route.departure_time[:19] if route.departure_time else 'Now'} | 
                    Arrival: {route.arrival_time[:19] if route.arrival_time else 'N/A'}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show route on map
            if route.points:
                import folium
                from streamlit_folium import st_folium
                
                mid_lat = (origin[0] + dest[0]) / 2
                mid_lng = (origin[1] + dest[1]) / 2
                
                m = folium.Map(location=[mid_lat, mid_lng], zoom_start=6, tiles="CartoDB dark_matter")
                
                # Route line
                route_coords = [(p["lat"], p["lng"]) for p in route.points]
                folium.PolyLine(route_coords, color="#e94560", weight=4, opacity=0.8).add_to(m)
                
                # Markers
                folium.Marker(
                    [origin[0], origin[1]],
                    popup=f"Origin: {origin_city}",
                    icon=folium.Icon(color="green", icon="play", prefix="fa")
                ).add_to(m)
                folium.Marker(
                    [dest[0], dest[1]],
                    popup=f"Destination: {dest_city}",
                    icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
                ).add_to(m)
                
                st_folium(m, width=None, height=400, use_container_width=True)
        else:
            st.error("Could not calculate route. Check API key and try again.")

# ═══════════════════════════════════════════
# TAB 3: Live Incidents
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ Live Traffic Incidents")
    st.markdown("Real-time disruptions detected across Indian logistics regions")
    
    region_boxes = {
        "Mumbai Region": (18.5, 72.5, 19.5, 73.5),
        "Delhi NCR": (28.0, 76.5, 29.0, 77.5),
        "Bangalore": (12.5, 77.0, 13.5, 78.0),
        "Chennai": (12.5, 79.5, 13.5, 80.5),
        "Kolkata": (22.0, 87.5, 23.0, 89.0),
        "Hyderabad": (17.0, 78.0, 18.0, 79.0),
        "Pune": (18.0, 73.5, 19.0, 74.5),
        "Ahmedabad": (22.5, 72.0, 23.5, 73.0),
        "All India": (8.0, 68.0, 37.0, 97.0),
    }
    
    selected_region = st.selectbox("Region", list(region_boxes.keys()), index=0)
    
    if st.button("🔍 Fetch Incidents", use_container_width=True, type="primary", key="fetch_incidents"):
        bbox = region_boxes[selected_region]
        
        with st.spinner(f"Fetching incidents in {selected_region}..."):
            incidents = get_traffic_incidents(bbox)
        
        if incidents:
            st.success(f"Found **{len(incidents)}** active incidents in {selected_region}")
            
            # Category breakdown
            categories = {}
            for inc in incidents:
                categories[inc.category] = categories.get(inc.category, 0) + 1
            
            if categories:
                import plotly.express as px
                fig = px.pie(
                    names=list(categories.keys()),
                    values=list(categories.values()),
                    template='plotly_dark',
                    title="Incident Categories",
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Incident list
            for inc in incidents[:20]:
                severity_colors = {0: '#888', 1: '#fdcb6e', 2: '#e17055', 3: '#d63031', 4: '#6c5ce7'}
                color = severity_colors.get(inc.severity, '#888')
                
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #333; border-left:4px solid {color}; 
                            border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem; color:#eee;">
                    <strong>[{inc.category}]</strong> {inc.description}
                    <br><span style="font-size:0.8rem; color:#aaa;">
                        {inc.from_location} → {inc.to_location} | 
                        Delay: {inc.delay_seconds//60} min | 
                        Length: {inc.length_meters}m
                        {' | Roads: ' + ', '.join(inc.road_numbers) if inc.road_numbers else ''}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No active incidents found in {selected_region}. Roads are clear!")
    else:
        st.info("Select a region and click **Fetch Incidents** to see live disruptions")

st.markdown("---")
st.caption("Real-time traffic data powered by TomTom Traffic APIs | LogiTrack AI")
