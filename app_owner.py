"""
=============================================================================
 app_owner.py
 Vehicle Predictive Maintenance — Car Owner Mobile App
 Run: streamlit run app_owner.py --server.port 8502
=============================================================================
"""
 
import os, logging
import requests
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
logging.basicConfig(level=logging.INFO)
 
# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiveCare — My Car",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ── API helpers ────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8010")
 
def api_alert(plate: str) -> dict:
    resp = requests.post(
        f"{API_URL}/api/v1/track2/alert",
        json={"plate_number": plate}, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()
 
def api_plates(track: int = 2) -> list:
    resp = requests.get(
        f"{API_URL}/api/v1/plates",
        params={"track": track}, timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("plates", [])
 
# ── Monitoring ─────────────────────────────────────────────────────────────
try:
    from monitoring import start_metrics_server, set_active_sessions
    try:
        start_metrics_server(port=8000)
    except Exception:
        pass
    MONITORING = True
except Exception:
    MONITORING = False
 
# ─────────────────────────────────────────────────────────────────────────────
# STYLING — mobile-first, warm & approachable
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');
 
:root {
    --bg:       #f5f4f0;
    --surface:  #ffffff;
    --border:   #e2e0d8;
    --accent:   #c8381a;
    --accent2:  #e8531a;
    --text:     #1a1a1a;
    --muted:    #8a8578;
    --ok:       #16803c;
    --ok-bg:    #dcfce7;
    --warn:     #b45309;
    --warn-bg:  #fef9c3;
    --med:      #c2410c;
    --med-bg:   #ffedd5;
    --danger:   #b91c1c;
    --danger-bg:#fee2e2;
}
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    max-width: 480px; margin: 0 auto;
}
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stHeader"], footer { display: none !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 16px 16px 80px !important; max-width: 480px !important; }
h1 { font-size: 1.5rem !important; font-weight: 800 !important; color: var(--text) !important; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--text) !important; }
h3 { font-size: 0.8rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.07em; }
 
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 14px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
.topbar-logo { font-size: 1.1rem; font-weight: 800; color: var(--accent); }
.topbar-sub  { font-size: 0.75rem; color: var(--muted); }
 
.vehicle-card { background: var(--text); color: white; border-radius: 16px; padding: 20px; margin-bottom: 20px; position: relative; overflow: hidden; }
.vehicle-card::before { content: ''; position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; background: var(--accent); border-radius: 50%; opacity: 0.15; }
.vehicle-plate { font-family: 'DM Mono', monospace; font-size: 1.4rem; font-weight: 500; letter-spacing: 0.05em; margin-bottom: 4px; }
.vehicle-name  { font-size: 0.85rem; opacity: 0.7; }
.vehicle-owner { font-size: 1rem; font-weight: 700; margin-bottom: 2px; }
 
.risk-card   { border-radius: 16px; padding: 20px; margin-bottom: 16px; }
.risk-none   { background: var(--ok-bg);     border: 1.5px solid #bbf7d0; }
.risk-low    { background: var(--warn-bg);   border: 1.5px solid #fde68a; }
.risk-medium { background: var(--med-bg);    border: 1.5px solid #fed7aa; }
.risk-high   { background: var(--danger-bg); border: 1.5px solid #fecaca; }
.risk-title  { font-size: 1.1rem; font-weight: 800; margin-bottom: 4px; }
.risk-none   .risk-title { color: var(--ok);     }
.risk-low    .risk-title { color: var(--warn);   }
.risk-medium .risk-title { color: var(--med);    }
.risk-high   .risk-title { color: var(--danger); }
 
.alert-bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; border-top-left-radius: 4px; padding: 16px; margin-bottom: 16px; font-size: 0.9rem; line-height: 1.65; white-space: pre-wrap; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
 
.sensor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
.sensor-pill { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.sensor-pill-label { font-size: 0.7rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
.sensor-pill-value { font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500; }
.sensor-pill.ok     { border-color: #bbf7d0; }
.sensor-pill.warn   { border-color: #fde68a; background: #fefce8; }
.sensor-pill.danger { border-color: #fecaca; background: #fff5f5; }
 
/* ── Chat UI ── */
.owner-chat-wrap { margin-top: 4px; }
.owner-msg-user {
    display: flex; justify-content: flex-end; margin-bottom: 10px;
}
.owner-msg-user .bubble {
    background: var(--accent); color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 14px; max-width: 80%;
    font-size: 0.88rem; line-height: 1.5;
}
.owner-msg-ai {
    display: flex; justify-content: flex-start; margin-bottom: 10px;
    align-items: flex-end; gap: 8px;
}
.owner-avatar { width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; flex-shrink: 0; }
.owner-msg-ai .bubble {
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 10px 14px; max-width: 80%;
    font-size: 0.88rem; line-height: 1.6;
    white-space: pre-wrap;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.chat-section-title { font-size: 0.85rem; font-weight: 700; color: var(--text); margin: 20px 0 12px; }
.chat-divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
 
[data-testid="stButton"] button { background: var(--accent) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; padding: 12px 24px !important; width: 100%; }
[data-testid="stButton"] button:hover { background: var(--accent2) !important; }
[data-testid="stSelectbox"] > div > div { background: var(--surface) !important; border: 1.5px solid var(--border) !important; border-radius: 10px !important; font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--text) !important; }
[data-testid="stSelectbox"] span { color: var(--text) !important; }
[data-testid="stSelectbox"] svg { fill: var(--muted) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] { background: var(--surface) !important; }
[data-testid="stSelectbox"] [data-baseweb="popover"] li { color: var(--text) !important; background: var(--surface) !important; }
[data-testid="stSelectbox"] [data-baseweb="popover"] li:hover { background: var(--bg) !important; }
.stSpinner > div { border-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
RISK_META = {
    0: ("Tidak Ada Risiko", "none",   "🟢", "Kendaraan Anda dalam kondisi baik"),
    1: ("Risiko Rendah",    "low",    "🟡", "Pantau kondisi kendaraan"),
    2: ("Risiko Sedang",    "medium", "🟠", "Jadwalkan servis segera"),
    3: ("Risiko Tinggi",    "high",   "🔴", "Inspeksi segera diperlukan!"),
}
 
SENSOR_DISPLAY = {
    "coolant_temp_c":     ("Suhu Mesin",   "°C"),
    "oil_pressure_psi":   ("Tekanan Oli",  "PSI"),
    "battery_voltage_v":  ("Daya Baterai", "V"),
    "tpms_psi":           ("Tekanan Ban",  "PSI"),
    "fuel_level_pct":     ("Bahan Bakar",  "%"),
    "speed_kmh":          ("Kecepatan",    "km/h"),
}
 
SENSOR_THRESHOLDS_T2 = {
    "coolant_temp_c":    (95.0,  105.0),
    "oil_pressure_psi":  (35.0,  25.0),
    "battery_voltage_v": (13.8,  12.5),
    "tpms_psi":          (30.0,  27.0),
    "fuel_level_pct":    (25.0,  10.0),
    "speed_kmh":         (60.0,  90.0),
}
 
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
 
for key in ["result", "chat_history", "vectorstore", "chains"]:
    if key not in st.session_state:
        st.session_state[key] = None
 
if st.session_state.chat_history is None:
    st.session_state.chat_history = []
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def sensor_status_t2(key, value):
    if key not in SENSOR_THRESHOLDS_T2:
        return "ok"
    warn_th, danger_th = SENSOR_THRESHOLDS_T2[key]
    if key in ("oil_pressure_psi", "battery_voltage_v", "tpms_psi", "fuel_level_pct"):
        if value <= danger_th: return "danger"
        if value <= warn_th:   return "warn"
        return "ok"
    if value >= danger_th: return "danger"
    if value >= warn_th:   return "warn"
    return "ok"
 
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    from rag_pipeline import build_vectorstore
    return build_vectorstore(force_rebuild=False)
 
@st.cache_resource(show_spinner=False)
def load_chains(_vs):
    from llm_chain import build_chains
    return build_chains(_vs)
 
 
def run_owner_chat(question: str, result: dict) -> str:
    """
    Follow-up chat for the owner — answers in Bahasa Indonesia,
    using the alert and SOP context as grounding.
    Never reveals raw sensor values or technical terms.
    """
    try:
        # ── Safety: sanitise query before hitting the LLM ─────────────────
        try:
            from src.safety import sanitise_chat_query
            ok, cleaned = sanitise_chat_query(question, track=2)
            if not ok:
                return f"⚠️ Maaf, pertanyaan Anda tidak dapat diproses: {cleaned}"
            question = cleaned
        except ImportError:
            pass
 
 
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = load_vectorstore()
        if st.session_state.chains is None:
            st.session_state.chains = load_chains(st.session_state.vectorstore)
 
        from rag_pipeline import get_retriever, format_context
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
 
        retriever = get_retriever(st.session_state.vectorstore, k=3)
        docs      = retriever.invoke(question)
        context   = format_context(docs)
 
        risk_label = result.get("risk_label", "Unknown")
        alert_text = result.get("alert", "Tidak tersedia")
 
        system = f"""Kamu adalah asisten kendaraan yang membantu pemilik kendaraan.
Pemilik kendaraan sudah menerima peringatan ini:
 
Status risiko: {risk_label}
Pesan peringatan:
{alert_text}
 
Konteks SOP:
{context}
 
Aturan penting:
- Jawab HANYA dalam Bahasa Indonesia yang mudah dipahami.
- JANGAN menyebut nama sensor teknis atau nilai sensor mentah.
- JANGAN memberikan informasi di luar konteks peringatan dan SOP di atas.
- Tetap tenang, jelas, dan membantu.
- Jika pertanyaan di luar konteks kendaraan, tolak dengan sopan.
"""
 
        llm    = st.session_state.chains["llm"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",  "{question}"),
        ])
        chain  = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
 
    except Exception as e:
        return f"Maaf, tidak dapat menjawab saat ini: {e}"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-logo">🚗 PredictiveCare</div>
        <div class="topbar-sub">Pemantauan Kesehatan Kendaraan</div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# PLATE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown("### Pilih Kendaraan Anda")
 
try:
    plates = api_plates(track=2)
except Exception:
    plates = []
 
if plates:
    selected_plate = st.selectbox("Nomor Plat", options=plates, label_visibility="collapsed")
else:
    selected_plate = st.text_input("Nomor Plat", value="B 1234 ABC", label_visibility="collapsed")
 
check_clicked = st.button("Cek Kondisi Kendaraan 🔍", use_container_width=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# LOAD & RUN (via FastAPI)
# ─────────────────────────────────────────────────────────────────────────────
 
if check_clicked and selected_plate:
    st.session_state.chat_history = []   # clear chat on new check
 
    with st.spinner("Memuat data kendaraan ..."):
        try:
            data = api_alert(selected_plate)
            st.session_state.result = data
            if MONITORING:
                try: set_active_sessions("owner", 1)
                except Exception: pass
        except Exception as e:
            st.error(f"Tidak dapat memuat data: {e}")
 
# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.result:
    data       = st.session_state.result
    risk_class = data.get("risk_class", 0)
    risk_label = RISK_META[risk_class][0]
    emoji      = RISK_META[risk_class][2]
    tagline    = RISK_META[risk_class][3]
    risk_css   = RISK_META[risk_class][1]
    alert      = data.get("alert")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Vehicle card ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="vehicle-card">
        <div class="vehicle-owner">{data.get('owner_name', '—')}</div>
        <div class="vehicle-plate">{data.get('plate_number', '—')}</div>
        <div class="vehicle-name">{data.get('car_model', '—')} · {data.get('car_year', '—')}</div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Risk status ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="risk-card risk-{risk_css}">
        <div class="risk-title">{emoji} {risk_label}</div>
        <div style="font-size:0.88rem;">{tagline}</div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Push alert bubble ─────────────────────────────────────────────────
    if risk_class == 0:
        st.markdown("""
<div class="alert-bubble" style="border-color:#bbf7d0;">
✅ <b>Kendaraan Anda dalam kondisi baik.</b>
 
Semua sensor berada dalam rentang normal hari ini. Tidak ada tindakan yang diperlukan.
 
Tetap pantau kondisi kendaraan Anda secara rutin.</div>
        """, unsafe_allow_html=True)
    elif alert:
        st.markdown(f'<div class="alert-bubble">{alert}</div>', unsafe_allow_html=True)
    else:
        fallbacks = {
            1: "🟡 Pantau kondisi kendaraan Anda selama 3–5 hari ke depan.",
            2: "🟠 Jadwalkan servis di bengkel resmi dalam 7 hari.\nHubungi: +62-800-000-0000",
            3: "🔴 JANGAN MENGENDARAI KENDARAAN.\nHubungi layanan darurat: +62-800-000-0000",
        }
        st.markdown(f'<div class="alert-bubble">{fallbacks.get(risk_class,"")}</div>', unsafe_allow_html=True)
 
    # ── Sensor pills ──────────────────────────────────────────────────────
    # Sensor values now come directly from the API response top-level fields.
    # Build entire grid as one HTML string (required — see fix note above).
    st.markdown("### Status Sensor")
    pills_html = '<div class="sensor-grid">'
    pills_added = 0
    for db_key, (label, unit) in SENSOR_DISPLAY.items():
        # API returns sensor values at top level of the response dict
        val = data.get(db_key)
        if val is None:
            continue
        try:
            val_f  = float(val)
            status = sensor_status_t2(db_key, val_f)
            # Format nicely: 1 decimal for most, integer for speed
            val_display = str(int(val_f)) if db_key == "speed_kmh" else f"{val_f:.1f}"
        except (TypeError, ValueError):
            continue
        pills_html += (
            f'<div class="sensor-pill {status}">'
            f'<div class="sensor-pill-label">{label}</div>'
            f'<div class="sensor-pill-value">{val_display}'
            f'<span style="font-size:0.7rem;opacity:0.6;margin-left:3px">{unit}</span>'
            f'</div></div>'
        )
        pills_added += 1
    pills_html += '</div>'
 
    if pills_added > 0:
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.caption("Data sensor tidak tersedia.")
 
    # ─────────────────────────────────────────────────────────────────────
    # FOLLOW-UP CHAT (Bahasa Indonesia)
    # Shown after any diagnosis — owner can ask questions about their alert
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)
    st.markdown('<div class="chat-section-title">💬 Ada pertanyaan tentang kendaraan Anda?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#8a8578;font-size:0.82rem;margin-bottom:14px;">'
        'Tanyakan apa saja seputar kondisi kendaraan Anda. Asisten akan menjawab dalam Bahasa Indonesia.'
        '</div>',
        unsafe_allow_html=True,
    )
 
    # Suggested questions — in Bahasa Indonesia, context-aware
    if risk_class == 0:
        suggestions = [
            "Kapan sebaiknya saya servis berikutnya?",
            "Apa yang perlu saya periksa secara rutin?",
        ]
    elif risk_class == 1:
        suggestions = [
            "Apakah aman untuk perjalanan jauh?",
            "Berapa lama saya bisa menunggu sebelum servis?",
            "Apa yang harus saya pantau?",
        ]
    elif risk_class == 2:
        suggestions = [
            "Apakah kendaraan masih aman dikendarai?",
            "Berapa lama proses servisnya?",
            "Bengkel mana yang terdekat?",
        ]
    else:  # High risk
        suggestions = [
            "Apakah saya boleh mengemudi ke bengkel?",
            "Apa yang terjadi jika saya tetap mengendarai?",
            "Bagaimana cara menghubungi layanan darurat?",
        ]
 
    st.markdown("**Pertanyaan umum:**")
    for suggestion in suggestions:
        if st.button(suggestion, key=f"owner_suggest_{suggestion[:20]}", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "content": suggestion}
            )
            with st.spinner("Sedang menjawab ..."):
                answer = run_owner_chat(suggestion, data)
            st.session_state.chat_history.append(
                {"role": "ai", "content": answer}
            )
 
    # Free-text input
    user_input = st.chat_input(
        "Ketik pertanyaan Anda ...",
        key="owner_chat_input",
    )
    if user_input and user_input.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input.strip()}
        )
        with st.spinner("Sedang menjawab ..."):
            answer = run_owner_chat(user_input.strip(), data)
        st.session_state.chat_history.append(
            {"role": "ai", "content": answer}
        )
 
    # Render chat history — build as one HTML string so the flex/grid
    # container works correctly within Streamlit's rendering model
    if st.session_state.chat_history:
        chat_html = '<div class="owner-chat-wrap">'
        for msg in st.session_state.chat_history:
            content_safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="owner-msg-user">'
                    f'<div class="bubble">{content_safe}</div>'
                    f'</div>'
                )
            else:
                chat_html += (
                    f'<div class="owner-msg-ai">'
                    f'<div class="owner-avatar">🚗</div>'
                    f'<div class="bubble">{content_safe}</div>'
                    f'</div>'
                )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
 
        if st.button("Hapus percakapan", key="clear_owner_chat"):
            st.session_state.chat_history = []
            st.rerun()
 
    # ── Evaluation detail (small, for demo) ───────────────────────────────
    with st.expander("ℹ️ Detail evaluasi"):
        st.caption(f"True Risk Class : {data.get('true_risk_class')} — {data.get('true_risk_label')}")
        st.caption(f"Predicted Class : {risk_class} — {risk_label}")
        if data.get("response_time_ms", 0) > 0:
            st.caption(f"Response time   : {data['response_time_ms']} ms")
 
    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
 
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:32px 16px;">
        <div style="font-size:3.5rem;margin-bottom:12px;">🚗</div>
        <div style="font-size:1rem;font-weight:700;margin-bottom:6px;">Cek Kondisi Kendaraan Anda</div>
        <div style="color:#8a8578;font-size:0.88rem;">Pilih nomor plat dan tekan tombol di atas.</div>
    </div>
    """, unsafe_allow_html=True)
