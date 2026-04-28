# LogiTrack AI - Smart Supply Chain Platform

An AI-powered supply chain optimization system built for the Google Solution Challenge 2026. The system continuously monitors shipments, preemptively detects disruptions using machine learning, and autonomously reroutes critical shipments before delays cascade -- powered by Gemini, XGBoost, Isolation Forest, SHAP, and Google Cloud.

## UN Sustainable Development Goals

- SDG 9: Industry, Innovation, and Infrastructure
- SDG 11: Sustainable Cities and Communities (reduced traffic and emissions through optimized routing)
- SDG 12: Responsible Consumption and Production (reduced fuel waste via multi-modal transport)
- SDG 13: Climate Action (lower carbon footprint through route and mode optimization)

## Problem Statement

Design a scalable system capable of continuously analyzing multifaceted transit data to preemptively detect and flag potential supply chain disruptions. Formulate dynamic mechanisms that instantly execute or recommend highly optimized route adjustments before localized bottlenecks cascade into broader delays.

## Solution Overview

LogiTrack AI addresses every aspect of the problem statement through a three-layer architecture:

1. Preemptive Detection Layer: Isolation Forest anomaly detection and XGBoost delay prediction continuously scan the fleet to catch disruptions before they impact delivery.
2. Dynamic Response Layer: When critical anomalies are found, the auto-reroute engine instantly optimizes affected shipments using a multi-modal transport optimizer comparing road, rail, air, waterway, and EV options.
3. Intelligence Layer: Gemini 2.0 Flash powers a natural language interface so logistics managers can query the entire ML engine conversationally, while SHAP explainability ensures every prediction is transparent.

## Machine Learning Models

### Model 1: XGBoost Delay Predictor
- Dataset: Delivery Logistics (25,000 rows, India-specific)
- Performance: 90.5% accuracy, 96.9% AUC
- Purpose: Predicts whether a specific shipment will be delayed based on partner, vehicle, weather, region, distance, and weight
- Features: 8 input features with label encoding for categorical variables

### Model 2: DataCo Supply Chain Delay Predictor
- Dataset: DataCo Smart Supply Chain (172,000 rows, global)
- Performance: 70.6% accuracy, 77.7% AUC
- Purpose: Broader supply chain delay patterns across international logistics

### Model 3: XGBoost ETA Predictor
- Dataset: Cainiao-AI LaDe (421,000 real last-mile deliveries)
- Performance: Low MAE on delivery time estimation
- Purpose: Predicts estimated delivery time in minutes based on distance, time of day, city, and day of week

### Model 4: Isolation Forest Anomaly Detector
- Type: Unsupervised ensemble (Isolation Forest + Z-Score)
- Purpose: Detects anomalous shipments in the fleet that deviate from normal patterns
- Output: Anomaly score, risk level (LOW, MEDIUM, HIGH, CRITICAL), and human-readable reasons

### Model 5: What-If Disruption Simulator
- Type: Scenario-based simulation engine
- Purpose: Models how disruptions (weather, port congestion, highway closure, strike) propagate across a fleet
- Output: Number of newly at-risk shipments, estimated penalty in INR, average delay probability increase

### Model 6: Multi-Modal Transport Optimizer
- Type: Rule-based optimization with Indian freight cost tables
- Purpose: Compares 5 transport modes (Road, Rail, Air Cargo, Waterway, EV Fleet) on cost, time, CO2 emissions, reliability, and deadline feasibility
- Output: Recommended mode with savings comparison versus air cargo baseline

## Key Features

### Autonomous Rerouting (The Killer Feature)
The system scans the persistent fleet of 30 shipments using the Isolation Forest model. When CRITICAL or HIGH risk anomalies are detected, it automatically triggers the transport optimizer for each affected shipment. The result is a before/after comparison showing the original mode versus the ML-recommended mode, with computed cost savings, time improvements, and CO2 reductions. This directly addresses the problem statement requirement of "dynamic mechanisms that instantly execute route adjustments."

### Gemini-Powered Intelligent Chatbot
A natural language interface backed by Gemini 2.0 Flash. The flow is:
1. User asks a question in plain English (e.g., "What is the delay risk for a 500kg shipment from Mumbai to Delhi in stormy weather?")
2. Gemini understands the intent and extracts structured parameters (region, weather, distance, weight)
3. The backend calls the appropriate ML model with extracted parameters
4. Gemini summarizes the ML result in expert-level natural language for the logistics manager

The chatbot can route to all 7 ML tools: delay prediction, ETA estimation, transport optimization, what-if simulation, explainability, anomaly scanning, and auto-rerouting.

### SHAP Explainability
Every delay prediction comes with a SHAP-based explanation showing which features pushed the prediction toward or away from "delayed." The system generates human-readable explanations like "Weather condition (stormy) and high package weight are the primary drivers of delay risk for this shipment."

### Real-Time Impact Metrics Dashboard
The dashboard computes and displays measurable business impact:
- Penalties prevented (INR) based on anomalies caught early
- CO2 saved (kg) through route optimization
- Disruptions caught before they impacted delivery
- On-time delivery rate improvement
- Model accuracy metrics

### Persistent Fleet Monitoring
Instead of generating random data on each request, the system maintains a persistent fleet of 30 realistic Indian shipments. All pages (Dashboard, Live Map, Anomaly Detection, Auto-Reroute) operate on the same fleet, providing a coherent view of the supply chain state.

### Live Map with Leaflet
An interactive dark-themed map showing all Indian logistics hubs, major transport corridors, and fleet positions color-coded by ML-calculated risk level. City labels, corridor lines, and shipment popups with anomaly details are all rendered in real time.

## Trained Model Files

| File | Size | Algorithm | Training Data |
|---|---|---|---|
| dataco_delay_predictor.joblib | 664 KB | XGBoost | DataCo Supply Chain (180K orders) |
| delay_predictor_v2.joblib | 192 KB | XGBoost | Indian Logistics Dataset |
| eta_predictor_v1.joblib | 1.7 MB | XGBoost | Cainiao LaDe Dataset |
| eta_predictor_v2.joblib | 2.4 MB | XGBoost | Cainiao LaDe Dataset (improved) |
| anomaly_detector.joblib | 3.8 MB | Isolation Forest | DataCo Supply Chain |

## Datasets Used

| Dataset | Size | Source | Use |
|---|---|---|---|
| Delivery_Logistics.csv | 25K rows | India logistics | Primary delay prediction |
| DataCo Supply Chain | 180K rows | Kaggle | Global delay patterns |
| LaDe (Cainiao-AI) | 421K deliveries | Research dataset | Last-mile ETA prediction |
| India OSM PBF | 1.67 GB | OpenStreetMap | Road network graph |
| UPPLY Seaports | 97 India ports | UPPLY | Port coordinates |
| FASTag Toll Data | 5K transactions | Government | Highway movement patterns |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /api/v1/ml/predict-delay | Predict delay probability for a shipment |
| POST | /api/v1/ml/predict-eta | Predict estimated delivery time |
| POST | /api/v1/ml/anomaly-detect | Detect anomaly in single shipment |
| POST | /api/v1/ml/anomaly-detect-batch | Batch fleet anomaly detection |
| POST | /api/v1/ml/explain-delay | Explainable delay prediction with SHAP |
| POST | /api/v1/ml/explain-eta | Explainable ETA prediction |
| POST | /api/v1/ml/whatif | What-if disruption simulation |
| POST | /api/v1/ml/optimize-transport | Multi-modal transport optimizer |
| POST | /api/v1/ml/chat | Gemini-powered intelligent chatbot |
| GET | /api/v1/ml/fleet | Get persistent fleet data |
| GET | /api/v1/ml/fleet-scan | Scan fleet for anomalies |
| GET | /api/v1/ml/auto-reroute | Detect and auto-optimize critical shipments |
| GET | /api/v1/ml/impact-metrics | Real-time impact dashboard metrics |
| GET | /api/v1/ml/global-importance | Global feature importance from XGBoost |
| GET | /api/v1/ml/transport-modes | Available transport mode specifications |

## Technology Stack

### Backend
- Python 3.10, FastAPI, Uvicorn
- XGBoost, LightGBM, scikit-learn for ML
- SHAP for model explainability
- Isolation Forest for anomaly detection
- Google Gemini 2.0 Flash for NLU and NLG
- LangGraph and LangChain for multi-agent orchestration
- TomTom Traffic API for live corridor data

### Frontend
- Standalone Web Dashboard (HTML, CSS, JavaScript) deployed on Vercel
- Streamlit Dashboard with 7 interactive pages
- Leaflet.js for live map visualization
- Flutter mobile application

### Infrastructure
- Render (FastAPI ML engine hosting)
- Vercel (Web dashboard hosting)
- GitHub (Version control and CI/CD)
- Google Cloud Platform (Pub/Sub, Cloud Run, Firestore)

## Deployment Architecture

```
Standalone Web Dashboard (Vercel)
        |
        v (HTTP POST)
FastAPI ML Engine (Render) <--- Streamlit Dashboard (Cloud/Local)
        |                           |
        v                           v
  5 Trained ML Models         Gemini 2.0 Flash API
  (.joblib files)             (Google AI Studio)
        |
        v
  TomTom Traffic API
  (Live corridor data)
```

## Project Structure

```
backend/                    FastAPI backend and ML API
  app/
    agents/                 LangGraph agent nodes
    api/routes/             REST endpoints (15+ routes)
    ml/                     Inference wrappers
    models/                 Pydantic schemas
    core/                   Firestore, PubSub, circuit breaker
  fleet.json                Persistent fleet (30 shipments)
ml/                         ML training pipeline
  train.py                  XGBoost delay predictor
  train_dataco.py           DataCo supply chain model
  eta_predictor.py          ETA predictor (LaDe dataset)
  anomaly_detector.py       Isolation Forest ensemble
  explainability.py         SHAP feature contribution
  whatif_simulator.py       Disruption scenario engine
  multimodal_optimizer.py   Multi-modal transport optimizer
  tomtom_traffic.py         TomTom API integration
  live_weather.py           IMD weather data
models/                     Saved model artifacts (.joblib)
streamlit_app/              Streamlit dashboard
  Home.py                   Dashboard home
  pages/
    1_Live_Shipments.py     India map with tracking
    2_Anomaly_Detection.py  Fleet anomaly detection
    3_What_If_Simulator.py  Disruption scenario tester
    4_Transport_Optimizer.py Multi-modal comparison
    5_Explainability.py     Model transparency
    6_Fleet_Monitor.py      Fleet health dashboard
    7_Ask_LogiTrack_AI.py   Gemini chatbot
solutionchallenge_web/      Standalone web dashboard
  index.html                SPA with 7 pages
  styles.css                Glassmorphism dark theme
  app.js                    ML API integration
requirements.txt            Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Virtual environment
- Gemini API key (for chatbot)
- TomTom API key (for live traffic, optional)

### Local Setup
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt

# Train ML models
python ml/train.py
python ml/train_dataco.py
python ml/eta_predictor.py
python ml/anomaly_detector.py
```

### Run Streamlit Dashboard
```bash
streamlit run streamlit_app/Home.py
```

### Run FastAPI Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Environment Variables
```
GOOGLE_API_KEY=your_gemini_api_key
TOMTOM_API_KEY=your_tomtom_key (optional)
```

---

Built for Google Solution Challenge 2026 | Team LogiTrack AI
