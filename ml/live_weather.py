
import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

if not os.getenv("OPENWEATHER_API_KEY"):
    load_dotenv('.env')

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

LOGISTICS_HUBS = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "delhi": {"lat": 28.7041, "lon": 77.1025},
    "bangalore": {"lat": 12.9716, "lon": 77.5946},
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "kolkata": {"lat": 22.5726, "lon": 88.3639},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "pune": {"lat": 18.5204, "lon": 73.8567},
}

def get_live_weather(hub_name: str) -> dict:

    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY not configured"}

    hub = hub_name.lower()
    if hub not in LOGISTICS_HUBS:
        return {"error": f"Hub {hub_name} not found in key logistics nodes"}

    coords = LOGISTICS_HUBS[hub]

    try:
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        severity = 0.0
        risk_type = "Clear"

        weather_main = data['weather'][0]['main'].lower()
        wind_speed = data['wind']['speed']

        if weather_main in ['thunderstorm', 'tornado', 'squall']:
            severity = 0.9
            risk_type = "Severe Storm"
        elif weather_main == 'rain':
            rain_1h = data.get('rain', {}).get('1h', 0)
            if rain_1h > 10:
                severity = 0.8
                risk_type = "Heavy Rain / Flooding"
            else:
                severity = 0.4
                risk_type = "Moderate Rain"
        elif weather_main in ['fog', 'mist', 'haze', 'dust', 'sand', 'ash']:
            severity = 0.6
            risk_type = "Low Visibility"
        elif weather_main == 'snow':
            severity = 0.7
            risk_type = "Snow / Ice"

        if wind_speed > 15:
            severity = max(severity, 0.5)
            risk_type += " + High Winds"

        return {
            "hub": hub_name,
            "condition": data['weather'][0]['description'].title(),
            "temp_c": data['main']['temp'],
            "wind_speed_ms": wind_speed,
            "visibility_m": data.get('visibility', 10000),
            "disruption_severity": min(1.0, severity),
            "risk_type": risk_type,
            "live": True
        }

    except Exception as e:
        return {"error": str(e), "live": False}

def check_all_hubs():

    results = {}
    for hub in LOGISTICS_HUBS.keys():
        results[hub] = get_live_weather(hub)
    return results

if __name__ == "__main__":
    print("Fetching Live Weather for India Logistics Hubs...")
    results = check_all_hubs()
    for hub, data in results.items():
        if "error" in data:
            print(f"[{hub.title()}] Error: {data['error']}")
        else:
            print(f"[{hub.title()}] {data['condition']} | Temp: {data['temp_c']}°C | "
                  f"Wind: {data['wind_speed_ms']} m/s | Risk: {data['risk_type']} (Severity: {data['disruption_severity']})")
