import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings
from app.core.circuit_breaker import circuit_breaker

class DisruptionClassifier:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            temperature=0.0
        )
        
    @circuit_breaker(max_attempts=3, min_wait=1, max_wait=10)
    async def _call_llm(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content
        
    async def classify_text(self, text: str) -> Dict[str, Any]:
        prompt = f"""Analyze the following disruption report from a logistics driver.
Extract structured JSON with these exact fields:
- event_type: one of [traffic_accident, weather_disruption, vehicle_breakdown, road_closure, traffic_jam]
- severity: one of [LOW, MEDIUM, HIGH, CRITICAL]
- estimated_clearance_time_mins: integer estimate
- description: brief summary

Report: {text}

Respond with valid JSON only, no markdown formatting."""

        try:
            raw = await self._call_llm(prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            
            result = json.loads(cleaned.strip())
            # Validate required fields exist
            required = ["event_type", "severity", "estimated_clearance_time_mins", "description"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")
            return result
            
        except Exception as e:
            print(f"LLM classification failed ({e}), using keyword fallback")
            return self._keyword_fallback(text)
    
    def _keyword_fallback(self, text: str) -> Dict[str, Any]:
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
