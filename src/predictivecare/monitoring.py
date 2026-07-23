"""
=============================================================================
 monitoring.py
 Vehicle Predictive Maintenance — LLM-Focused Prometheus Metrics
=============================================================================
 Exposes metrics at http://localhost:8000/metrics
 Scraped by Prometheus every 15s, visualised in Grafana.

 LLM-specific metrics tracked:
   vpm_llm_requests_total       Counter   — LLM calls by track, model, endpoint
   vpm_llm_response_time_ms     Histogram — LLM latency in milliseconds
   vpm_llm_input_tokens_total   Counter   — cumulative input tokens by model
   vpm_llm_output_tokens_total  Counter   — cumulative output tokens by model
   vpm_llm_cost_usd_total       Counter   — cumulative USD cost by model/track
   vpm_llm_skipped_total        Counter   — requests that skipped LLM (no anomaly)
   vpm_rag_chunks_retrieved     Histogram — RAG docs retrieved per query
   vpm_rag_retrieval_time_ms    Histogram — RAG retrieval latency
   vpm_fault_class_total        Counter   — fault class distribution (Track 1)
   vpm_risk_class_total         Counter   — risk class distribution (Track 2)
   vpm_active_sessions          Gauge     — active Streamlit sessions per app
   vpm_output_valid_total       Counter   — LLM output passed/failed guardrails

 Usage:
   from predictivecare.monitoring import record_llm_query, set_active_sessions, start_metrics_server
   start_metrics_server(port=8000)
   record_llm_query(track=1, result=result, model="gemini-2.0-flash", ...)
=============================================================================
"""

import logging
import threading
from typing import Dict, Any, Optional

from prometheus_client import Counter, Histogram, Gauge, start_http_server

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM REQUEST METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Total LLM calls — by track, model name, and endpoint
LLM_REQUESTS_TOTAL = Counter(
    name="vpm_llm_requests_total",
    documentation="Total LLM API calls made, labelled by track, model, and endpoint",
    labelnames=["track", "model", "endpoint"],
)

# LLM response time in milliseconds
# Buckets: 500ms → 30s (Gemini Flash typically 2–8s)
LLM_RESPONSE_TIME_MS = Histogram(
    name="vpm_llm_response_time_ms",
    documentation="LLM chain end-to-end response time in milliseconds",
    labelnames=["track", "model"],
    buckets=[500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000],
)

# Cumulative input tokens by model
LLM_INPUT_TOKENS_TOTAL = Counter(
    name="vpm_llm_input_tokens_total",
    documentation="Cumulative LLM prompt input tokens consumed, by model",
    labelnames=["track", "model"],
)

# Cumulative output tokens by model
LLM_OUTPUT_TOKENS_TOTAL = Counter(
    name="vpm_llm_output_tokens_total",
    documentation="Cumulative LLM response output tokens generated, by model",
    labelnames=["track", "model"],
)

# Cumulative USD cost — broken down by track and model for billing analysis
LLM_COST_USD_TOTAL = Counter(
    name="vpm_llm_cost_usd_total",
    documentation="Cumulative estimated USD cost of LLM API calls",
    labelnames=["track", "model"],
)

# Total cumulative cost across ALL tracks and models — single number for dashboard
LLM_COST_TOTAL = Counter(
    name="vpm_llm_cost_total_usd",
    documentation="Total cumulative USD cost of all LLM API calls (all tracks + models combined)",
    labelnames=[],
)

# Skipped LLM calls — Class 0 / No Risk — vehicle healthy, no generation needed
LLM_SKIPPED_TOTAL = Counter(
    name="vpm_llm_skipped_total",
    documentation="Requests skipped LLM call because result was Normal/No Risk",
    labelnames=["track"],
)

# LLM output guardrail results — did the output pass validation?
OUTPUT_VALID_TOTAL = Counter(
    name="vpm_output_valid_total",
    documentation="LLM output validation results (passed vs failed guardrails)",
    labelnames=["track", "status"],   # status: passed | failed
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Number of RAG chunks retrieved per query — reflects retrieval quality
RAG_CHUNKS_RETRIEVED = Histogram(
    name="vpm_rag_chunks_retrieved",
    documentation="Number of SOP context chunks retrieved per LLM query (MMR k=4)",
    labelnames=["track"],
    buckets=[1, 2, 3, 4, 5, 6, 8],
)

# RAG retrieval latency — time to embed query + search vector store
RAG_RETRIEVAL_TIME_MS = Histogram(
    name="vpm_rag_retrieval_time_ms",
    documentation="ChromaDB MMR retrieval latency in milliseconds",
    labelnames=["track"],
    buckets=[50, 100, 200, 300, 500, 800, 1000, 2000],
)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Fault class distribution — Track 1
FAULT_CLASS_TOTAL = Counter(
    name="vpm_fault_class_total",
    documentation="Track 1: ML fault class predictions (0=Normal, 1-7=fault types)",
    labelnames=["fault_class", "fault_label"],
)

# Risk class distribution — Track 2
RISK_CLASS_TOTAL = Counter(
    name="vpm_risk_class_total",
    documentation="Track 2: ML risk class predictions (0=No Risk, 1-3=risk levels)",
    labelnames=["risk_class", "risk_label"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge(
    name="vpm_active_sessions",
    documentation="Currently active Streamlit sessions per app",
    labelnames=["app"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PRICING CONSTANTS (Gemini API — May 2026)
# ─────────────────────────────────────────────────────────────────────────────

PRICING = {
    "gemini-2.0-flash": {
        "input_per_1m":  0.10,
        "output_per_1m": 0.40,
    },
    "gemini-3.1-flash-lite": {
        "input_per_1m":  0.075,
        "output_per_1m": 0.30,
    },
    "default": {
        "input_per_1m":  0.10,
        "output_per_1m": 0.40,
    },
}

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate estimated USD cost for a single LLM call."""
    pricing      = PRICING.get(model, PRICING["default"])
    input_cost   = (input_tokens  / 1_000_000) * pricing["input_per_1m"]
    output_cost  = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return round(input_cost + output_cost, 8)


# ─────────────────────────────────────────────────────────────────────────────
# RECORD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def record_llm_query(
    track: int,
    result: Dict[str, Any],
    model: str             = "gemini-2.0-flash",
    endpoint: str          = "",
    input_tokens: int      = 0,
    output_tokens: int     = 0,
    rag_retrieval_ms: int  = 0,
    output_valid: bool     = True,
) -> None:
    """
    Record all LLM-related metrics for one completed query.

    Args:
        track:            1 (Technician) or 2 (Owner)
        result:           Dict from llm_chain.run_track1() or run_track2()
        model:            LLM model name (from GOOGLE_MODEL env var)
        endpoint:         API endpoint that triggered the call
        input_tokens:     Prompt tokens sent to the LLM
        output_tokens:    Completion tokens received from the LLM
        rag_retrieval_ms: Time spent on RAG retrieval in milliseconds
        output_valid:     Whether the LLM output passed guardrail validation
    """
    resp_ms    = result.get("response_time_ms", 0)
    chunks     = result.get("context_chunks", 0)
    llm_called = result.get("llm_called", resp_ms > 0)

    # ── Classification metrics ────────────────────────────────────────────────
    if track == 1:
        fault_class = str(result.get("fault_class", "unknown"))
        fault_label = result.get("fault_label", "unknown")
        FAULT_CLASS_TOTAL.labels(
            fault_class=fault_class,
            fault_label=fault_label,
        ).inc()
    else:
        risk_class = str(result.get("risk_class", "unknown"))
        risk_label = result.get("risk_label", "unknown")
        RISK_CLASS_TOTAL.labels(
            risk_class=risk_class,
            risk_label=risk_label,
        ).inc()

    # ── LLM was called ────────────────────────────────────────────────────────
    if llm_called and resp_ms > 0:
        LLM_REQUESTS_TOTAL.labels(
            track=str(track),
            model=model,
            endpoint=endpoint or f"/api/v1/track{track}",
        ).inc()

        LLM_RESPONSE_TIME_MS.labels(
            track=str(track),
            model=model,
        ).observe(resp_ms)

        if input_tokens > 0:
            LLM_INPUT_TOKENS_TOTAL.labels(
                track=str(track), model=model
            ).inc(input_tokens)

        if output_tokens > 0:
            LLM_OUTPUT_TOKENS_TOTAL.labels(
                track=str(track), model=model
            ).inc(output_tokens)

        cost = calculate_cost(model, input_tokens, output_tokens)
        if cost > 0:
            LLM_COST_USD_TOTAL.labels(
                track=str(track), model=model
            ).inc(cost)
            # Also increment the grand total counter
            LLM_COST_TOTAL.inc(cost)

        OUTPUT_VALID_TOTAL.labels(
            track=str(track),
            status="passed" if output_valid else "failed",
        ).inc()

    else:
        # LLM was skipped — vehicle healthy / no risk
        LLM_SKIPPED_TOTAL.labels(track=str(track)).inc()

    # ── RAG metrics ───────────────────────────────────────────────────────────
    if chunks > 0:
        RAG_CHUNKS_RETRIEVED.labels(track=str(track)).observe(chunks)

    if rag_retrieval_ms > 0:
        RAG_RETRIEVAL_TIME_MS.labels(track=str(track)).observe(rag_retrieval_ms)

    log.debug(
        f"[Metrics T{track}] model={model} resp={resp_ms}ms "
        f"in={input_tokens} out={output_tokens} chunks={chunks} "
        f"valid={output_valid}"
    )


# Backward-compatible wrappers so existing code doesn't break
def record_track1_query(result: Dict[str, Any], **kwargs) -> None:
    """Backward-compatible wrapper — calls record_llm_query(track=1)."""
    import os
    record_llm_query(
        track=1,
        result=result,
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        **kwargs,
    )

def record_track2_query(result: Dict[str, Any], **kwargs) -> None:
    """Backward-compatible wrapper — calls record_llm_query(track=2)."""
    import os
    record_llm_query(
        track=2,
        result=result,
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        **kwargs,
    )


def set_active_sessions(app: str, count: int) -> None:
    """Update active session gauge for 'technician' or 'owner' app."""
    ACTIVE_SESSIONS.labels(app=app).set(count)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS SERVER
# ─────────────────────────────────────────────────────────────────────────────

_server_started = False
_server_lock    = threading.Lock()

def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus HTTP metrics server. Safe to call multiple times.
    Metrics exposed at: http://localhost:{port}/metrics
    """
    global _server_started
    with _server_lock:
        if _server_started:
            return
        try:
            start_http_server(port)
            _server_started = True
            log.info(f"Prometheus metrics server started on :{port}/metrics")
        except OSError as e:
            log.warning(f"Could not start metrics server on :{port} — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFANA DASHBOARD REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

GRAFANA_PANELS = {
    "Total API Cost (All Time)": {
        "type":  "stat",
        "query": 'vpm_llm_cost_total_usd_total',
        "note":  "Grand total USD spent on Gemini API — all tracks combined",
    },
    "API Cost by Track": {
        "type":  "stat",
        "query": 'sum by (track) (vpm_llm_cost_usd_total)',
        "note":  "USD cost split between Track 1 (Technician) and Track 2 (Owner)",
    },
    "LLM Requests Total": {
        "type":  "stat",
        "query": 'sum(vpm_llm_requests_total)',
        "note":  "Total LLM API calls made across both tracks",
    },
    "LLM Response Time p50 / p95 / p99": {
        "type":  "gauge",
        "query": 'histogram_quantile(0.95, sum(rate(vpm_llm_response_time_ms_bucket[5m])) by (le, model))',
        "note":  "95th percentile LLM latency in ms — key SLA metric",
    },
    "LLM Response Time Distribution": {
        "type":  "timeseries",
        "query": 'histogram_quantile(0.50, sum(rate(vpm_llm_response_time_ms_bucket[5m])) by (le))',
        "note":  "Median LLM latency over time",
    },
    "Input Tokens per Minute": {
        "type":  "timeseries",
        "query": 'rate(vpm_llm_input_tokens_total[1m])',
        "note":  "Token consumption rate — useful for quota monitoring",
    },
    "Output Tokens per Minute": {
        "type":  "timeseries",
        "query": 'rate(vpm_llm_output_tokens_total[1m])',
        "note":  "Output token rate — affects cost per minute",
    },
    "Cumulative Cost USD by Track": {
        "type":  "stat",
        "query": 'sum by (track) (vpm_llm_cost_usd_total)',
        "note":  "Running API cost split between Track 1 and Track 2",
    },
    "Cost Rate (USD per minute)": {
        "type":  "timeseries",
        "query": 'rate(vpm_llm_cost_usd_total[1m])',
        "note":  "Real-time spend rate",
    },
    "RAG Chunks Retrieved": {
        "type":  "stat",
        "query": 'histogram_quantile(0.95, sum(rate(vpm_rag_chunks_retrieved_bucket[5m])) by (le))',
        "note":  "p95 RAG chunks per query — should be consistently 4 (MMR k=4)",
    },
    "RAG Retrieval Latency p95": {
        "type":  "gauge",
        "query": 'histogram_quantile(0.95, sum(rate(vpm_rag_retrieval_time_ms_bucket[5m])) by (le))',
        "note":  "ChromaDB MMR search latency at p95",
    },
    "LLM Skipped (Healthy Vehicles)": {
        "type":  "stat",
        "query": 'sum(vpm_llm_skipped_total)',
        "note":  "Queries where no LLM call was made — vehicle healthy",
    },
    "Output Guardrail Pass Rate": {
        "type":  "gauge",
        "query": 'sum(vpm_output_valid_total{status="passed"}) / sum(vpm_output_valid_total)',
        "note":  "% of LLM outputs that passed safety/completeness validation",
    },
    "Fault Class Distribution": {
        "type":  "piechart",
        "query": 'vpm_fault_class_total',
        "note":  "ML fault classification breakdown — Track 1",
    },
    "Risk Class Distribution": {
        "type":  "piechart",
        "query": 'vpm_risk_class_total',
        "note":  "ML risk level breakdown — Track 2",
    },
    "Active Sessions": {
        "type":  "stat",
        "query": 'vpm_active_sessions',
        "note":  "Live Streamlit sessions per app",
    },
}


def print_grafana_reference() -> None:
    """Print all Grafana panel PromQL queries for manual dashboard setup."""
    print("\n" + "=" * 65)
    print("  Grafana Dashboard — LLM Monitoring Panels")
    print("=" * 65)
    for panel, cfg in GRAFANA_PANELS.items():
        print(f"\n  Panel : {panel}")
        print(f"  Type  : {cfg['type']}")
        print(f"  Query : {cfg['query']}")
        print(f"  Note  : {cfg['note']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n" + "=" * 60)
    print("  Monitoring — Self Test (LLM Metrics)")
    print("=" * 60)

    start_metrics_server(port=8000)

    model = "gemini-2.0-flash"

    print("\nSimulating Track 1 (Technician) queries ...")
    t1_tests = [
        {"fault_class": 0, "fault_label": "Normal",               "response_time_ms": 0,    "context_chunks": 0},
        {"fault_class": 6, "fault_label": "Oil Pressure Issue",   "response_time_ms": 3200, "context_chunks": 4},
        {"fault_class": 3, "fault_label": "Cooling System Problem","response_time_ms": 2800, "context_chunks": 4},
        {"fault_class": 4, "fault_label": "Engine Misfire",       "response_time_ms": 3100, "context_chunks": 4},
    ]
    for r in t1_tests:
        record_llm_query(
            track=1, result=r, model=model,
            endpoint="/api/v1/track1/diagnose",
            input_tokens=820 if r["response_time_ms"] > 0 else 0,
            output_tokens=390 if r["response_time_ms"] > 0 else 0,
            rag_retrieval_ms=180 if r["context_chunks"] > 0 else 0,
            output_valid=True,
        )
        print(f"  T1: {r['fault_label']} — resp={r['response_time_ms']}ms")

    print("\nSimulating Track 2 (Owner) queries ...")
    t2_tests = [
        {"risk_class": 0, "risk_label": "No Risk",    "response_time_ms": 0,    "context_chunks": 0},
        {"risk_class": 3, "risk_label": "High Risk",  "response_time_ms": 2900, "context_chunks": 4},
        {"risk_class": 2, "risk_label": "Medium Risk","response_time_ms": 2600, "context_chunks": 4},
        {"risk_class": 1, "risk_label": "Low Risk",   "response_time_ms": 2200, "context_chunks": 4},
    ]
    for r in t2_tests:
        record_llm_query(
            track=2, result=r, model=model,
            endpoint="/api/v1/track2/alert",
            input_tokens=640 if r["response_time_ms"] > 0 else 0,
            output_tokens=280 if r["response_time_ms"] > 0 else 0,
            rag_retrieval_ms=160 if r["context_chunks"] > 0 else 0,
            output_valid=True,
        )
        print(f"  T2: {r['risk_label']} — resp={r['response_time_ms']}ms")

    set_active_sessions("technician", 2)
    set_active_sessions("owner", 5)

    cost = calculate_cost("gemini-2.0-flash", 820, 390)
    print(f"\n  Cost per T1 call (820 in + 390 out): ${cost:.6f}")

    print_grafana_reference()
    print(f"\nMetrics at: http://localhost:8000/metrics")
    print("Sleeping 10s ...")
    time.sleep(10)
    print("Self-test complete.")
