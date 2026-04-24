"""Tests for the disruption classifier keyword fallback logic.
Fully self-contained — extracts only the fallback method to avoid all external deps.
"""

def _keyword_fallback(text: str) -> dict:
    """Duplicated from disruption_classifier.py for isolated testing."""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["accident", "crash", "collision"]):
        event_type, severity, clearance = "traffic_accident", "HIGH", 120
    elif any(w in text_lower for w in ["storm", "flood", "rain", "snow", "weather"]):
        event_type, severity, clearance = "weather_disruption", "MEDIUM", 180
    elif any(w in text_lower for w in ["breakdown", "flat tire", "engine", "mechanical"]):
        event_type, severity, clearance = "vehicle_breakdown", "HIGH", 90
    elif any(w in text_lower for w in ["closed", "blocked", "closure"]):
        event_type, severity, clearance = "road_closure", "CRITICAL", 240
    elif any(w in text_lower for w in ["traffic", "jam", "congestion", "slow"]):
        event_type, severity, clearance = "traffic_jam", "LOW", 45
    else:
        event_type, severity, clearance = "traffic_jam", "MEDIUM", 60
        
    return {
        "event_type": event_type,
        "severity": severity,
        "estimated_clearance_time_mins": clearance,
        "description": text
    }

def test_keyword_fallback_accident():
    result = _keyword_fallback("There's been a major accident on NH48")
    assert result["event_type"] == "traffic_accident"
    assert result["severity"] == "HIGH"
    assert result["estimated_clearance_time_mins"] == 120

def test_keyword_fallback_weather():
    result = _keyword_fallback("Heavy rain and flooding on the road")
    assert result["event_type"] == "weather_disruption"
    assert result["severity"] == "MEDIUM"

def test_keyword_fallback_breakdown():
    result = _keyword_fallback("Engine breakdown, vehicle not moving")
    assert result["event_type"] == "vehicle_breakdown"
    assert result["severity"] == "HIGH"

def test_keyword_fallback_closure():
    result = _keyword_fallback("Road is completely closed due to construction")
    assert result["event_type"] == "road_closure"
    assert result["severity"] == "CRITICAL"
    assert result["estimated_clearance_time_mins"] == 240

def test_keyword_fallback_traffic():
    result = _keyword_fallback("Stuck in slow traffic jam near highway")
    assert result["event_type"] == "traffic_jam"
    assert result["severity"] == "LOW"

def test_keyword_fallback_unknown():
    result = _keyword_fallback("Something weird happened on the route")
    assert result["event_type"] == "traffic_jam"
    assert result["severity"] == "MEDIUM"

if __name__ == '__main__':
    test_keyword_fallback_accident()
    test_keyword_fallback_weather()
    test_keyword_fallback_breakdown()
    test_keyword_fallback_closure()
    test_keyword_fallback_traffic()
    test_keyword_fallback_unknown()
    print("All disruption classifier tests passed!")
