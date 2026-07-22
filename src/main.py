"""
=============================================================================
 src/main.py
 Vehicle Predictive Maintenance — FastAPI Backend
=============================================================================
 Exposes the prediction pipeline as a REST API.
 Streamlit apps call this instead of importing modules directly.
 
 Endpoints:
   GET  /health                      — liveness check
   GET  /api/v1/plates               — list all plate numbers
   POST /api/v1/track1/diagnose      — fault classification + LLM brief
   POST /api/v1/track2/alert         — risk detection + Bahasa Indonesia alert
 
 Run:
   uvicorn src.main:app --reload --port 8010
 
 Streamlit apps connect to:
   http://localhost:8010
=============================================================================
"""
 
import logging
import sys
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
 
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from database import (
    get_all_plates,
    get_track1_avg_sensors, get_track1_record,
    get_track2_avg_sensors, get_track2_record,
    save_track1_prediction, save_track2_prediction, log_query,
    row_to_track1_sensors, row_to_track2_sensors,
)
from rag_pipeline import build_vectorstore
from llm_chain import build_chains, run_track1, run_track2, FAULT_METADATA, RISK_METADATA
from monitoring import record_llm_query, record_track1_query, record_track2_query, start_metrics_server
from src.safety import validate_request, validate_brief, validate_alert
from src.logger import log_request
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
TRACK1_FEATURES = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
    "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
    "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
    "BATTERY VOLTAGE V", "FUEL TEMP C",
]
TRACK2_FEATURES = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT",
    "COOLANT TEMP C", "OIL PRESSURE PSI", "BATTERY VOLTAGE V",
    "TPMS PSI", "AMBIENT TEMP C", "CABIN HUMIDITY PCT",
    "FUEL LEVEL PCT", "BRAKE PEDAL EVENTS", "SPEED KMH",
]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# APP STATE — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────
 
class AppState:
    track1_model = None
    track2_model = None
    vectorstore  = None
    chains       = None
 
 
state = AppState()
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and vector store at startup. Clean up on shutdown."""
    log.info("Starting PredictiveCare API ...")
 
    # Prometheus metrics server
    try:
        start_metrics_server(port=8000)
    except Exception:
        pass
 
    # ML models
    try:
        state.track1_model = joblib.load("ml_models/track1_fault_classifier.pkl")
        state.track2_model = joblib.load("ml_models/track2_risk_classifier.pkl")
        log.info("ML models loaded.")
    except Exception as e:
        log.error(f"ML models not found: {e}")
 
    # RAG vector store + LLM chains
    try:
        state.vectorstore = build_vectorstore(force_rebuild=False)
        state.chains      = build_chains(state.vectorstore)
        log.info("RAG + LLM chains ready.")
    except Exception as e:
        log.error(f"RAG/LLM init failed: {e}")
 
    yield
 
    log.info("PredictiveCare API shutdown.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="PredictiveCare — Predictive Maintenance API",
    description="AI-powered vehicle fault detection and owner alerts for connected vehicles.",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to VM IP in production
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
 
class DiagnoseRequest(BaseModel):
    plate_number: str
 
    @field_validator("plate_number")
    @classmethod
    def clean_plate(cls, v):
        return v.strip().upper()
 
 
class DiagnoseResponse(BaseModel):
    plate_number:     str
    test_id:          Optional[str]
    owner_name:       Optional[str]
    car_model:        Optional[str]
    car_year:         Optional[int]
    fault_class:      int
    fault_label:      str
    priority:         str
    is_anomaly:       bool
    brief:            Optional[str]
    true_fault_class: Optional[int]
    true_fault_label: Optional[str]
    response_time_ms: int
    context_chunks:   int
    safety_passed:    bool
    output_valid:     bool
 
 
class AlertRequest(BaseModel):
    plate_number: str
 
    @field_validator("plate_number")
    @classmethod
    def clean_plate(cls, v):
        return v.strip().upper()
 
 
class AlertResponse(BaseModel):
    plate_number:    str
    test_id:         Optional[str]
    owner_name:      Optional[str]
    car_model:       Optional[str]
    car_year:        Optional[int]
    risk_class:      int
    risk_label:      str
    risk_label_id:   str
    alert:           Optional[str]
    true_risk_class: Optional[int]
    true_risk_label: Optional[str]
    response_time_ms: int
    context_chunks:   int
    safety_passed:    bool
    output_valid:     bool
    # Sensor readings — for display in owner app sensor pills
    coolant_temp_c:     Optional[float]
    oil_pressure_psi:   Optional[float]
    battery_voltage_v:  Optional[float]
    tpms_psi:           Optional[float]
    fuel_level_pct:     Optional[float]
    speed_kmh:          Optional[float]
 
 # ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/health", tags=["System"])
async def health():
    """Liveness check — returns system component status."""
    return {
        "status":          "ok",
        "track1_model":    state.track1_model is not None,
        "track2_model":    state.track2_model is not None,
        "vectorstore":     state.vectorstore  is not None,
        "llm_chains":      state.chains       is not None,
    }
 
 
@app.get("/api/v1/plates", tags=["Data"])
async def list_plates(track: int = Query(default=1, ge=1, le=2)):
    """
    List all available plate numbers for the given track.
    Used to populate the dropdown in the Streamlit apps.
    """
    try:
        plates = get_all_plates(track=track)
        return {"track": track, "plates": plates, "count": len(plates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/api/v1/track1/diagnose", response_model=DiagnoseResponse, tags=["Track 1"])
async def diagnose(request: DiagnoseRequest):
    """
    Track 1 — Fault Classification (Technician).
 
    1. Load sensor data from DB by plate number
    2. Validate inputs (safety filter)
    3. Run ML classifier → fault class 0–7
    4. If anomaly: run RAG + LLM → Technician Fault Brief
    5. Log to JSONL + Prometheus + PostgreSQL query_log
    """
    plate = request.plate_number
 
    # ── Fetch 30-day telemetry and compute average ────────────────────────────
    # The ML classifier was trained on aggregated 30-day sensor data.
    # get_track1_avg_sensors() queries all 30 daily rows and returns their
    # mean — this is the correct input for classification.
    record = get_track1_avg_sensors(plate)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No Track 1 telemetry found for plate: {plate}",
        )
 
    # Sensor dict is already keyed correctly by get_track1_avg_sensors()
    sensors = {k: record[k] for k in [
        "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
        "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
        "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
        "BATTERY VOLTAGE V", "FUEL TEMP C",
    ]}
 
    # ── Safety validation ─────────────────────────────────────────────────────
    ok, err = validate_request(plate, sensors, track=1)
    if not ok:
        log_request(
            track=1, plate_number=plate,
            predicted_class=-1, predicted_label="INVALID",
            response_time_ms=0, context_chunks=0,
            llm_called=False, test_id=record.get("test_id"),
            safety_passed=False, status="validation_error", error=err,
        )
        raise HTTPException(status_code=422, detail=err)
 
    # ── ML classification ─────────────────────────────────────────────────────
    if state.track1_model is None:
        raise HTTPException(status_code=503, detail="Track 1 ML model not loaded.")
 
    X = np.array([[sensors[f] for f in TRACK1_FEATURES]])
    fault_class = int(state.track1_model.predict(X)[0])
    fault_meta  = FAULT_METADATA.get(fault_class, FAULT_METADATA[0])
    fault_label = fault_meta["label"]
    priority    = fault_meta["priority"]
    is_anomaly  = fault_class != 0
 
    # ── LLM chain (anomalies only) ────────────────────────────────────────────
    brief        = None
    resp_ms      = 0
    chunks_used  = 0
    llm_called   = False
    output_valid = True
 
    if is_anomaly and state.chains is not None:
        result = run_track1(state.chains, fault_class=fault_class, sensor_readings=sensors)
        brief       = result["brief"]
        resp_ms     = result["response_time_ms"]
        chunks_used = result["context_chunks"]
        llm_called  = True
 
        # Output guardrail
        valid_ok, valid_msg = validate_brief(brief)
        output_valid = valid_ok
        if not valid_ok:
            log.warning(f"Track 1 output validation: {valid_msg}")
 
        record_llm_query(
            track=1,
            result=result,
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            endpoint="/api/v1/track1/diagnose",
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            rag_retrieval_ms=result.get("rag_retrieval_ms", 0),
            output_valid=output_valid,
        )
 
    # ── Persist prediction ────────────────────────────────────────────────────
    try:
        save_track1_prediction(record["test_id"], fault_class, fault_label)
    except Exception:
        pass
 
    # ── Log ───────────────────────────────────────────────────────────────────
    log_request(
        track=1, plate_number=plate,
        predicted_class=fault_class, predicted_label=fault_label,
        response_time_ms=resp_ms, context_chunks=chunks_used,
        llm_called=llm_called, test_id=record.get("test_id"),
        true_class=int(record["true_fault_class"]),
        true_label=record["true_fault_label"],
        output_valid=output_valid,
    )
    try:
        log_query(
            track=1, plate_number=plate, test_id=record["test_id"],
            predicted_class=fault_class, predicted_label=fault_label,
            response_time_ms=resp_ms, context_chunks=chunks_used,
            llm_called=llm_called,
        )
    except Exception:
        pass
 
    return DiagnoseResponse(
        plate_number=plate,
        test_id=record.get("test_id"),
        owner_name=record.get("owner_name"),
        car_model=record.get("car_model"),
        car_year=record.get("car_year"),
        fault_class=fault_class,
        fault_label=fault_label,
        priority=priority,
        is_anomaly=is_anomaly,
        brief=brief,
        true_fault_class=int(record["true_fault_class"]),
        true_fault_label=record["true_fault_label"],
        response_time_ms=resp_ms,
        context_chunks=chunks_used,
        safety_passed=True,
        output_valid=output_valid,
    )
 
 
@app.post("/api/v1/track2/alert", response_model=AlertResponse, tags=["Track 2"])
async def alert(request: AlertRequest):
    """
    Track 2 — Risk Detection (Owner).
 
    1. Load sensor data from DB by plate number
    2. Validate inputs (safety filter)
    3. Run ML classifier → risk class 0–3
    4. If risk > 0: run RAG + LLM → Bahasa Indonesia Push Alert
    5. Validate alert output (no sensor values, correct language)
    6. Log to JSONL + Prometheus + PostgreSQL query_log
    """
    plate = request.plate_number
 
    # ── Fetch daily readings and compute average ─────────────────────────────
    # The ML risk classifier receives the 7-reading daily average
    # (one per 2-hour interval, 07:00–19:00) — not a single snapshot.
    record = get_track2_avg_sensors(plate)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No Track 2 telemetry found for plate: {plate}",
        )
 
    # Sensor dict already keyed correctly by get_track2_avg_sensors()
    sensors = {k: record[k] for k in [
        "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT",
        "COOLANT TEMP C", "OIL PRESSURE PSI", "BATTERY VOLTAGE V",
        "TPMS PSI", "AMBIENT TEMP C", "CABIN HUMIDITY PCT",
        "FUEL LEVEL PCT", "BRAKE PEDAL EVENTS", "SPEED KMH",
    ]}
 
    ok, err = validate_request(plate, sensors, track=2)
    if not ok:
        log_request(
            track=2, plate_number=plate,
            predicted_class=-1, predicted_label="INVALID",
            response_time_ms=0, context_chunks=0,
            llm_called=False, test_id=record.get("test_id"),
            safety_passed=False, status="validation_error", error=err,
        )
        raise HTTPException(status_code=422, detail=err)
 
    if state.track2_model is None:
        raise HTTPException(status_code=503, detail="Track 2 ML model not loaded.")

 
    X = np.array([[sensors[f] for f in TRACK2_FEATURES]])
    risk_class  = int(state.track2_model.predict(X)[0])
    risk_meta   = RISK_METADATA.get(risk_class, RISK_METADATA[0])
    risk_label  = risk_meta["label"]
    risk_label_id = risk_meta["id"]
 
    push_alert   = None
    resp_ms      = 0
    chunks_used  = 0
    llm_called   = False
    output_valid = True
 
    if risk_class > 0 and state.chains is not None:
        result = run_track2(state.chains, risk_class=risk_class, sensor_readings=sensors)
        push_alert  = result["alert"]
        resp_ms     = result["response_time_ms"]
        chunks_used = result["context_chunks"]
        llm_called  = True
 
        valid_ok, valid_msg = validate_alert(push_alert or "")
        output_valid = valid_ok
        if not valid_ok:
            log.warning(f"Track 2 output validation: {valid_msg}")
 
        record_llm_query(
            track=2,
            result=result,
            model=os.getenv("GOOGLE_MODEL"),
            endpoint="/api/v1/track2/alert",
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            rag_retrieval_ms=result.get("rag_retrieval_ms", 0),
            output_valid=output_valid,
        )
    else:
        record_llm_query(
            track=2,
            result={"risk_class": risk_class, "risk_label": risk_label,
                    "response_time_ms": 0, "context_chunks": 0},
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            endpoint="/api/v1/track2/alert",
        )
 
    try:
        save_track2_prediction(record["test_id"], risk_class, risk_label)
    except Exception:
        pass
 
    log_request(
        track=2, plate_number=plate,
        predicted_class=risk_class, predicted_label=risk_label,
        response_time_ms=resp_ms, context_chunks=chunks_used,
        llm_called=llm_called, test_id=record.get("test_id"),
        true_class=int(record["true_risk_class"]),
        true_label=record["true_risk_label"],
        output_valid=output_valid,
    )
    try:
        log_query(
            track=2, plate_number=plate, test_id=record["test_id"],
            predicted_class=risk_class, predicted_label=risk_label,
            response_time_ms=resp_ms, context_chunks=chunks_used,
            llm_called=llm_called,
        )
    except Exception:
        pass
 
    def _f(key):
        # record from get_track2_avg_sensors already has lowercase sensor keys
        v = record.get(key)
        return float(v) if v is not None else None
 
    return AlertResponse(
        plate_number=plate,
        test_id=record.get("test_id"),
        owner_name=record.get("owner_name"),
        car_model=record.get("car_model"),
        car_year=record.get("car_year"),
        risk_class=risk_class,
        risk_label=risk_label,
        risk_label_id=risk_label_id,
        alert=push_alert,
        true_risk_class=int(record["true_risk_class"]),
        true_risk_label=record["true_risk_label"],
        response_time_ms=resp_ms,
        context_chunks=chunks_used,
        safety_passed=True,
        output_valid=output_valid,
        coolant_temp_c=_f("coolant_temp_c"),
        oil_pressure_psi=_f("oil_pressure_psi"),
        battery_voltage_v=_f("battery_voltage_v"),
        tpms_psi=_f("tpms_psi"),
        fuel_level_pct=_f("fuel_level_pct"),
        speed_kmh=_f("speed_kmh"),
    )
