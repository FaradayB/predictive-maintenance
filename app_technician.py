"""
=============================================================================
 app_technician.py
 Vehicle Predictive Maintenance — Technician Workshop App
 Run: streamlit run app_technician.py --server.port 8501
=============================================================================
"""
 
import os, logging, time
import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
logging.basicConfig(level=logging.INFO)
 
# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiveCare — Workshop",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── API helpers (calls FastAPI instead of importing directly) ──────────────
API_URL = os.getenv("API_URL", "http://localhost:8010")
 
def api_diagnose(plate: str) -> dict:
    resp = requests.post(
        f"{API_URL}/api/v1/track1/diagnose",
        json={"plate_number": plate}, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()
 
def api_plates(track: int = 1) -> list:
    resp = requests.get(
        f"{API_URL}/api/v1/plates",
        params={"track": track}, timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("plates", [])
 
def api_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {}
 
# ── Fallback direct imports if API not available ───────────────────────────
try:
    from database import get_track1_history, get_query_log
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False
 
try:
    from rag_pipeline import build_vectorstore
    from llm_chain import build_chains
    from monitoring import start_metrics_server, set_active_sessions
    LLM_AVAILABLE = True
    try:
        start_metrics_server(port=8000)
    except Exception:
        pass
except Exception:
    LLM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
 
:root {
    --bg:        #0d0f14;
    --surface:   #161a22;
    --surface2:  #1e2330;
    --border:    #2a3040;
    --accent:    #e8531a;
    --accent2:   #f0952a;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --ok:        #22c55e;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --info:      #3b82f6;
}
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1 { font-size: 1.8rem !important; font-weight: 800 !important; letter-spacing: -0.03em; color: var(--text) !important; }
h2 { font-size: 1.2rem !important; font-weight: 700 !important; color: var(--text) !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.08em; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; }
.card-accent { border-left: 3px solid var(--accent); }
.badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.05em; }
.badge-ok      { background: #14532d33; color: var(--ok);    border: 1px solid #14532d; }
.badge-warn    { background: #78350f33; color: var(--warn);  border: 1px solid #78350f; }
.badge-danger  { background: #7f1d1d33; color: var(--danger);border: 1px solid #7f1d1d; }
.badge-info    { background: #1e3a5f33; color: var(--info);  border: 1px solid #1e3a5f; }
.sensor-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.88rem; }
.sensor-name { color: var(--muted); font-family: 'JetBrains Mono', monospace; }
.sensor-val  { font-family: 'JetBrains Mono', monospace; font-weight: 500; }
.brief-block { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.7; white-space: pre-wrap; color: var(--text); }
.priority-critical { background: #7f1d1d; border: 1px solid var(--danger); border-radius: 8px; padding: 12px 16px; color: #fca5a5; font-weight: 700; font-size: 0.9rem; }
.priority-high     { background: #78350f; border: 1px solid var(--warn);   border-radius: 8px; padding: 12px 16px; color: #fcd34d; font-weight: 700; font-size: 0.9rem; }
.priority-medium   { background: #1e3a5f; border: 1px solid var(--info);   border-radius: 8px; padding: 12px 16px; color: #93c5fd; font-weight: 700; font-size: 0.9rem; }
.priority-normal   { background: #14532d; border: 1px solid var(--ok);     border-radius: 8px; padding: 12px 16px; color: #86efac; font-weight: 700; font-size: 0.9rem; }
 
/* ── Chat UI ── */
.chat-container { margin-top: 8px; }
.chat-msg-user {
    display: flex; justify-content: flex-end; margin-bottom: 10px;
}
.chat-msg-user .bubble {
    background: var(--accent); color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; max-width: 75%;
    font-size: 0.88rem; line-height: 1.5;
}
.chat-msg-ai {
    display: flex; justify-content: flex-start; margin-bottom: 10px;
}
.chat-msg-ai .bubble {
    background: var(--surface2); color: var(--text);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 10px 16px; max-width: 80%;
    font-size: 0.88rem; line-height: 1.6;
    white-space: pre-wrap;
}
.chat-label { font-size: 0.7rem; color: var(--muted); margin-bottom: 3px; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.07em; }
 
[data-testid="stSelectbox"] > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; }
[data-testid="stButton"] button { background: var(--accent) !important; color: white !important; border: none !important; border-radius: 6px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.03em !important; padding: 10px 24px !important; width: 100%; }
[data-testid="stButton"] button:hover { background: var(--accent2) !important; }
[data-testid="stMetric"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.5rem !important; }
.stSpinner > div { border-color: var(--accent) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
TRACK1_FEATURES = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
    "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
    "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
    "BATTERY VOLTAGE V", "FUEL TEMP C",
]
 
FAULT_META = {
    0: ("Normal",                 "normal",   "✅ NORMAL — No action required"),
    1: ("Battery Degradation",    "medium",   "🔵 MEDIUM — Schedule within 14 days"),
    2: ("Brake System Issue",     "high",     "🟡 HIGH — Inspect within 3 days"),
    3: ("Cooling System Problem", "high",     "🟡 HIGH — Inspect within 3 days"),
    4: ("Engine Misfire",         "high",     "🟡 HIGH — Inspect within 3 days"),
    5: ("Alternator Failure",     "medium",   "🔵 MEDIUM — Schedule within 7 days"),
    6: ("Oil Pressure Issue",     "critical", "🔴 CRITICAL — Do not drive. Inspect immediately"),
    7: ("Transmission Problem",   "high",     "🟡 HIGH — Inspect within 3 days"),
}
 
THRESHOLDS = {
    "O2 SENSOR V":       (0.10, 0.50, 0.65),
    "MAF G PER S":       (5.0,  7.0,  10.0),
    "THROTTLE POS PCT":  (10.0, 20.0, 30.0),
    "CRANK RPM":         (700,  900,  1100),
    "CAM ADVANCE DEG":   (9.0,  12.0, 18.0),
    "KNOCK COUNT 30D":   (0,    1,    5),
    "COOLANT TEMP C":    (85.0, 95.0, 105.0),
    "OIL PRESSURE PSI":  (35.0, 50.0, 20.0),
    "MAP KPA":           (30.0, 40.0, 48.0),
    "EGR DUTY PCT":      (15.0, 25.0, 40.0),
    "BATTERY VOLTAGE V": (13.8, 14.5, 12.0),
    "FUEL TEMP C":       (25.0, 45.0, 58.0),
}
 
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
 
for key in ["result", "record", "chat_history", "vectorstore", "chains"]:
    if key not in st.session_state:
        st.session_state[key] = None
 
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []   # list of {"role": "user"|"ai", "content": str}
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def sensor_status(name, value):
    if name not in THRESHOLDS:
        return "ok"
    lo, hi, crit = THRESHOLDS[name]
    if name in ("OIL PRESSURE PSI", "BATTERY VOLTAGE V"):
        if value <= crit: return "danger"
        if value <= lo:   return "warn"
        return "ok"
    if value >= crit: return "danger"
    if value >= hi:   return "warn"
    return "ok"
 
def priority_html(level):
    labels = {
        "critical": ("🔴", "CRITICAL — Do not drive. Inspect immediately", "priority-critical"),
        "high":     ("🟡", "HIGH — Inspect within 3 days",                 "priority-high"),
        "medium":   ("🔵", "MEDIUM — Schedule within 7–14 days",           "priority-medium"),
        "normal":   ("✅", "NORMAL — No action required",                  "priority-normal"),
    }
    emoji, text, cls = labels.get(level, labels["normal"])
    return f'<div class="{cls}">{emoji} {text}</div>'

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    from rag_pipeline import build_vectorstore
    return build_vectorstore(force_rebuild=False)
 
@st.cache_resource(show_spinner=False)
def load_chains(_vs):
    from llm_chain import build_chains
    return build_chains(_vs)
 
 
def run_chat_query(question: str, result: dict) -> str:
    """
    Send a follow-up question to the LLM, grounded in:
    - The fault brief already generated
    - The sensor readings
    - The SOP context via RAG retrieval
    """
    try:
        # ── Safety: sanitise query before hitting the LLM ─────────────────
        try:
            from src.safety import sanitise_chat_query
            ok, cleaned = sanitise_chat_query(question, track=1)
            if not ok:
                return f"⚠️ {cleaned}"
            question = cleaned
        except ImportError:
            pass  # safety module not available — proceed without filtering
 
 
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = load_vectorstore()
        if st.session_state.chains is None:
            st.session_state.chains = load_chains(st.session_state.vectorstore)
 
        from rag_pipeline import get_retriever, format_context
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
 
        # Retrieve relevant SOP context for this follow-up question
        retriever = get_retriever(st.session_state.vectorstore, k=3)
        docs      = retriever.invoke(question)
        context   = format_context(docs)
 
        # Build the chat system prompt with full context
        fault_label = result.get("predicted_label", "Unknown")
        brief       = result.get("brief", "Not available")
        sensors_str = "\n".join(
            f"  {k}: {v}" for k, v in result.get("sensors", {}).items()
        )
 
        system = f"""You are an expert automotive diagnostic assistant helping a workshop technician.
 
The vehicle has already been diagnosed with: {fault_label}
 
The full fault brief for this vehicle:
{brief}
 
Sensor readings:
{sensors_str}
 
Relevant SOP context:
{context}
 
Answer the technician's follow-up question using ONLY the information above.
Be concise and actionable. Do not introduce information not present in the brief or SOP context.
"""
 
        llm    = st.session_state.chains["llm"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",  "{question}"),
        ])
        chain  = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
 
    except Exception as e:
        return f"Unable to answer: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.markdown("## 🔧 PredictiveCare Workshop")
    st.markdown('<div style="color:#6b7280;font-size:0.8rem;margin-bottom:20px;">Technician Diagnostic System</div>', unsafe_allow_html=True)
    st.divider()
 
    st.markdown("### Vehicle")
    try:
        plates = api_plates(track=1)
    except Exception:
        plates = []
 
    if plates:
        selected_plate = st.selectbox("Plate Number", options=plates, label_visibility="collapsed")
    else:
        selected_plate = st.text_input("Plate Number", value="B 1234 ABC")
 
    load_clicked = st.button("Load Vehicle Data", use_container_width=True)
 
    st.divider()
 
    # API health
    st.markdown("### System")
    health = api_health()
    c1, c2 = st.columns(2)
    with c1:
        ok = health.get("track1_model", False)
        st.markdown(f'<span class="badge badge-{"ok" if ok else "danger"}">{"ML ✓" if ok else "ML ✗"}</span>', unsafe_allow_html=True)
    with c2:
        ok = health.get("llm_chains", False)
        st.markdown(f'<span class="badge badge-{"ok" if ok else "danger"}">{"LLM ✓" if ok else "LLM ✗"}</span>', unsafe_allow_html=True)
 
    st.divider()
 
    if DB_AVAILABLE:
        st.markdown("### Recent Queries")
        try:
            logs = get_query_log(limit=5)
            for entry in logs:
                label = entry.get("predicted_label") or "—"
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#6b7280;padding:3px 0;">'
                    f'T1 · {entry["plate_number"]} · <span style="color:#e8eaf0">{label}</span></div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.caption("No queries yet.")
 
    st.divider()
    st.markdown('<div style="color:#6b7280;font-size:0.72rem;">API: :8010/docs<br>Metrics: :8000/metrics<br>Grafana: :3000</div>', unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown("# 🔧 Technician Fault Diagnostic")
st.markdown('<div style="color:#6b7280;font-size:0.9rem;margin-bottom:24px;">30-day OBD telemetry · ML anomaly detection · RAG-grounded fault brief</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & RUN (via FastAPI)
# ─────────────────────────────────────────────────────────────────────────────
 
if load_clicked and selected_plate:
    # Clear previous chat when loading a new vehicle
    st.session_state.chat_history = []
 
    with st.spinner("Running diagnosis via API ..."):
        try:
            data = api_diagnose(selected_plate)
            st.session_state.result = {
                "record":            data,
                "sensors":           {},          # sensors not returned by API — display from record
                "predicted_class":   data.get("fault_class", 0),
                "predicted_label":   data.get("fault_label", "Normal"),
                "brief":             data.get("brief"),
                "resp_ms":           data.get("response_time_ms", 0),
                "chunks_used":       data.get("context_chunks", 0),
            }
        except Exception as e:
            st.error(f"API error: {e}. Check that `uvicorn src.main:app --port 8010` is running.")
 
# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
 
if st.session_state.result:
    r       = st.session_state.result
    record  = r["record"]
    p_class = r["predicted_class"]
    p_label = r["predicted_label"]
    brief   = r["brief"]
    priority_level = FAULT_META.get(p_class, FAULT_META[0])[1]
 
    # ── Vehicle identity ──────────────────────────────────────────────────
    # Vehicle card — use native columns instead of HTML wrapper
    # (wrapping st.columns inside HTML divs breaks Streamlit rendering)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Plate",  record.get("plate_number", "—"))
    with c2: st.metric("Owner",  record.get("owner_name",   "—"))
    with c3: st.metric("Model",  record.get("car_model",    "—"))
    with c4: st.metric("Year",   str(record.get("car_year", "—")))
 
    # ── Priority banner ───────────────────────────────────────────────────
    st.markdown(priority_html(priority_level), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Fault brief ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.4], gap="large")
 
    with col_left:
        st.markdown("## Sensor Classification")
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            f'<span style="color:#6b7280;font-size:0.8rem;">ML Prediction</span>'
            f'<span class="badge badge-{"ok" if p_class==0 else "danger"}">{p_class} — {p_label}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:#6b7280;font-size:0.8rem;">Ground Truth</span>'
            f'<span class="badge badge-info">{record.get("true_fault_class","?")} — {record.get("true_fault_label","?")}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if r["resp_ms"] > 0:
            st.markdown(
                f'<div style="color:#6b7280;font-size:0.75rem;">'
                f'⏱ {r["resp_ms"]}ms &nbsp;·&nbsp; 📄 {r["chunks_used"]} chunks</div>',
                unsafe_allow_html=True,
            )
 
    with col_right:
        st.markdown("## Technician Fault Brief")
        if p_class == 0:
            st.markdown(
                '<div class="card" style="border-color:#22c55e;">'
                '<div style="color:#22c55e;font-size:1.1rem;font-weight:700;margin-bottom:8px;">✅ Vehicle Operating Normally</div>'
                '<div style="color:#6b7280;font-size:0.88rem;">All sensor readings within normal SOP thresholds. No action required.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif brief:
            st.markdown(f'<div class="brief-block">{brief}</div>', unsafe_allow_html=True)
        else:
            st.info("LLM not available. ML classification above is still valid.")
 
    # ─────────────────────────────────────────────────────────────────────
    # FOLLOW-UP CHAT
    # Only shown after a diagnosis has been run
    # ─────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💬 Ask a Follow-Up Question")
    st.markdown(
        '<div style="color:#6b7280;font-size:0.85rem;margin-bottom:16px;">'
        'Ask anything about this vehicle\'s diagnosis. Answers are grounded in the SOP and fault brief above.'
        '</div>',
        unsafe_allow_html=True,
    )
 
    # Suggested questions
    suggestions = [
        "What tools do I need for this inspection?",
        "How long should this repair take?",
        "Can the vehicle be driven to the workshop?",
        "What parts may need to be replaced?",
        "What could cause this fault to reoccur?",
    ]
    st.markdown("**Suggested questions:**")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": suggestion}
                )
                with st.spinner("Thinking ..."):
                    answer = run_chat_query(suggestion, r)
                st.session_state.chat_history.append(
                    {"role": "ai", "content": answer}
                )
 
    # Chat input
    user_input = st.chat_input(
        "Ask the diagnostic assistant ...",
        key="tech_chat_input",
    )
    if user_input and user_input.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input.strip()}
        )
        with st.spinner("Thinking ..."):
            answer = run_chat_query(user_input.strip(), r)
        st.session_state.chat_history.append(
            {"role": "ai", "content": answer}
        )
 
    # Render chat history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            content_safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="chat-msg-user">'
                    f'<div><div class="chat-label" style="text-align:right;">You</div>'
                    f'<div class="bubble">{content_safe}</div></div>'
                    f'</div>'
                )
            else:
                chat_html += (
                    f'<div class="chat-msg-ai">'
                    f'<div><div class="chat-label">PredictiveCare Assistant</div>'
                    f'<div class="bubble">{content_safe}</div></div>'
                    f'</div>'
                )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
 
        # Clear chat button
        if st.button("Clear conversation", key="clear_chat_tech"):
            st.session_state.chat_history = []
            st.rerun()
 
    # ── 30-day history ────────────────────────────────────────────────────
    if DB_AVAILABLE:
        st.divider()
        st.markdown("## 30-Day Sensor History")
        with st.expander("View all telemetry records for this vehicle"):
            try:
                history = get_track1_history(selected_plate)
                if history:
                    df = pd.DataFrame(history)
                    display_cols = ["day_in_window", "coolant_temp_c", "oil_pressure_psi",
                                    "battery_voltage_v", "crank_rpm", "true_fault_label"]
                    st.dataframe(df[[c for c in display_cols if c in df.columns]],
                                 use_container_width=True)
            except Exception:
                st.caption("History not available.")
 
else:
    st.markdown(
        '<div class="card" style="text-align:center;padding:48px;">'
        '<div style="font-size:3rem;margin-bottom:16px;">🔧</div>'
        '<div style="font-size:1.1rem;font-weight:700;margin-bottom:8px;">Select a Vehicle</div>'
        '<div style="color:#6b7280;">Choose a plate number from the sidebar and click Load Vehicle Data.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

