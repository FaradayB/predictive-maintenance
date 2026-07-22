"""
=============================================================================
 llm_chain.py
 Vehicle Predictive Maintenance — LLM Chains (Google Gemini)
=============================================================================
 Two prompt chains matching the system flow diagram:

   Track 1 — Technician Fault Brief
     Input  : fault_class (int), fault_label (str), sensor_readings (dict)
     Context: RAG retrieval from sop_track1_technician_fault_diagnosis.md
     Model  : gemini-2.0-flash
     Output : structured Technician Fault Brief in English

   Track 2 — Owner Risk Alert (Bahasa Indonesia)
     Input  : risk_class (int), risk_label (str), sensor_readings (dict)
     Context: RAG retrieval from sop_track2_owner_risk_alert.md
     Model  : gemini-2.0-flash
     Output : plain-language Push Alert in Bahasa Indonesia

 Usage:
   from llm_chain import build_chains, run_track1, run_track2
   chains = build_chains(vectorstore)
   brief  = run_track1(chains, fault_class=6, fault_label="Oil Pressure Issue", sensors={...})
   alert  = run_track2(chains, risk_class=3, risk_label="High Risk", sensors={...})
=============================================================================
"""

import os
import time
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from rag_pipeline import get_retriever, format_context

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LLM_MODEL       = "gemini-2.0-flash"
TEMPERATURE     = 0.1   # low temperature — factual, grounded outputs
MAX_TOKENS      = 1024


# ─────────────────────────────────────────────────────────────────────────────
# LLM INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialise Gemini 2.0 Flash via langchain-google-genai.
    Requires GOOGLE_API_KEY in .env file.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Add it to your .env file:\n"
            "  GOOGLE_API_KEY=your_key_here\n"
            "Get a free key: https://aistudio.google.com/app/apikey"
        )
    log.info(f"Initialising LLM  (model={LLM_MODEL}, temp={TEMPERATURE})")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 1 — TECHNICIAN FAULT BRIEF
# ─────────────────────────────────────────────────────────────────────────────

TRACK1_SYSTEM_PROMPT = """You are an expert automotive diagnostic assistant for authorised workshops.

Your role is to generate a structured Technician Fault Brief based on:
1. The ML anomaly detection result (fault class and label)
2. The vehicle's actual sensor readings
3. The Standard Operating Procedure (SOP) retrieved from the knowledge base

CRITICAL RULES:
- Answer ONLY using information from the provided SOP context. Do not use general automotive knowledge not present in the context.
- Every inspection step must come directly from the SOP, not from model memory.
- If a sensor reading is outside the normal range defined in the SOP, flag it explicitly.
- Do not speculate about causes not mentioned in the SOP.
- Output must be professional, structured, and actionable for a workshop technician.

SOP CONTEXT (use this as your sole knowledge source):
{context}
"""

TRACK1_HUMAN_PROMPT = """Generate a Technician Fault Brief for the following vehicle anomaly.

FAULT DETECTION RESULT:
- Fault Class : {fault_class} — {fault_label}
- Priority    : {priority}

SENSOR READINGS (from 30-day telemetry):
{sensor_readings}

Output the brief using this exact structure:

TECHNICIAN FAULT BRIEF
======================
Fault Class   : [class id] — [fault label]
Priority      : [priority level]
Detection     : 30-day telemetry anomaly (ML classification)

SENSOR FINDINGS
---------------
[List each sensor that is outside normal SOP thresholds. Format: Sensor: actual value (SOP normal range: X–Y)]

PROBABLE CAUSE
--------------
[Top 1–3 probable causes from the SOP, ranked by likelihood]

INSPECTION CHECKLIST
--------------------
[Numbered steps directly from the SOP for this fault class]

RECOMMENDED ACTION
------------------
[Exact recommended action from SOP]

SOP REFERENCE
-------------
[State which SOP document and section this brief is based on]
"""

def build_track1_chain(llm: ChatGoogleGenerativeAI):
    """Build the Track 1 prompt chain for technician fault briefs."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRACK1_SYSTEM_PROMPT),
        ("human",  TRACK1_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 2 — OWNER RISK ALERT (BAHASA INDONESIA)
# ─────────────────────────────────────────────────────────────────────────────

TRACK2_SYSTEM_PROMPT = """Kamu adalah asisten notifikasi kendaraan yang bertugas mengirim peringatan kondisi kendaraan kepada pemilik kendaraan di Indonesia.

You are a vehicle notification assistant responsible for sending vehicle condition alerts to vehicle owners in Indonesia.

Your role is to generate a plain-language push alert in Bahasa Indonesia based on:
1. The risk classification result (risk class and label)
2. The vehicle's daily sensor readings
3. The Standard Operating Procedure (SOP) retrieved from the knowledge base

CRITICAL RULES:
- Output MUST be in Bahasa Indonesia.
- Answer ONLY using information from the provided SOP context. Do not use knowledge not present in the context.
- NEVER show raw sensor values or OBD codes to the owner.
- NEVER use technical sensor names (e.g. do not say "coolant temperature sensor" — say "mesin mulai panas").
- Use the plain-language translation table in the SOP context to convert technical terms.
- Match the urgency tone to the risk class: calm for Class 1, firm for Class 2, urgent for Class 3.
- Use the exact alert template structure from the SOP for this risk class.

SOP CONTEXT (use this as your sole knowledge source):
{context}
"""

TRACK2_HUMAN_PROMPT = """Generate a push alert in Bahasa Indonesia for the following vehicle risk detection.

RISK DETECTION RESULT:
- Risk Class : {risk_class} — {risk_label}

SENSOR READINGS (from today's 12-hour monitoring window):
{sensor_readings}

Generate the push alert following the SOP template for this risk class.
The alert must:
- Open with the correct risk emoji and level indicator from the SOP
- Explain what was detected in plain Bahasa Indonesia (no technical terms)
- State the urgency clearly
- Give a specific action the owner must take
- Include the emergency contact if risk class is 2 or 3

Output ONLY the push alert text that will appear on the owner's phone. Nothing else.
"""

def build_track2_chain(llm: ChatGoogleGenerativeAI):
    """Build the Track 2 prompt chain for owner push alerts in Bahasa Indonesia."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRACK2_SYSTEM_PROMPT),
        ("human",  TRACK2_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_chains(vectorstore: Chroma) -> Dict[str, Any]:
    """
    Build both LLM chains and return them together with the vectorstore.

    Args:
        vectorstore: Pre-built Chroma instance from rag_pipeline.py

    Returns:
        Dict with keys: llm, vectorstore, track1_chain, track2_chain
    """
    llm = get_llm()
    log.info("Building Track 1 chain (Technician Fault Brief) ...")
    t1_chain = build_track1_chain(llm)
    log.info("Building Track 2 chain (Owner Risk Alert — Bahasa Indonesia) ...")
    t2_chain = build_track2_chain(llm)
    log.info("Both chains ready.")
    return {
        "llm":           llm,
        "vectorstore":   vectorstore,
        "track1_chain":  t1_chain,
        "track2_chain":  t2_chain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FAULT CLASS METADATA
# ─────────────────────────────────────────────────────────────────────────────

FAULT_METADATA = {
    0: {"label": "Normal",                 "priority": "None — no action required"},
    1: {"label": "Battery Degradation",    "priority": "Medium — schedule within 14 days"},
    2: {"label": "Brake System Issue",     "priority": "High — inspect within 3 days"},
    3: {"label": "Cooling System Problem", "priority": "High — inspect within 3 days"},
    4: {"label": "Engine Misfire",         "priority": "High — inspect within 3 days"},
    5: {"label": "Alternator Failure",     "priority": "Medium — schedule within 7 days"},
    6: {"label": "Oil Pressure Issue",     "priority": "Critical — do not drive, inspect immediately"},
    7: {"label": "Transmission Problem",   "priority": "High — inspect within 3 days"},
}

RISK_METADATA = {
    0: {"label": "No Risk",     "id": "Tidak Ada Risiko"},
    1: {"label": "Low Risk",    "id": "Risiko Rendah"},
    2: {"label": "Medium Risk", "id": "Risiko Sedang"},
    3: {"label": "High Risk",   "id": "Risiko Tinggi"},
}


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 1 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_track1(
    chains: Dict[str, Any],
    fault_class: int,
    sensor_readings: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run the Track 1 chain: retrieve SOP context → generate Technician Fault Brief.

    Args:
        chains:          Output of build_chains()
        fault_class:     ML model output (0–7)
        sensor_readings: Dict of sensor name → value from 30-day telemetry

    Returns:
        Dict with fault_class, fault_label, brief (text), context, response_time_ms
    """
    meta         = FAULT_METADATA.get(fault_class, FAULT_METADATA[0])
    fault_label  = meta["label"]
    priority     = meta["priority"]

    # Format sensor readings for prompt
    sensor_str = "\n".join(
        f"  {k}: {v}" for k, v in sensor_readings.items()
    )

    # RAG retrieval — timed separately for monitoring
    rag_query = (
        f"{fault_label} inspection procedure sensor thresholds "
        f"fault class {fault_class}"
    )
    retriever  = get_retriever(chains["vectorstore"], k=4)
    t_rag      = time.time()
    retrieved  = retriever.invoke(rag_query)
    rag_ms     = int((time.time() - t_rag) * 1000)
    context    = format_context(retrieved)

    log.info(f"Track 1 — running chain for Class {fault_class}: {fault_label}")
    t0 = time.time()

    brief = chains["track1_chain"].invoke({
        "context":         context,
        "fault_class":     fault_class,
        "fault_label":     fault_label,
        "priority":        priority,
        "sensor_readings": sensor_str,
    })

    elapsed_ms = int((time.time() - t0) * 1000)

    # Estimate token usage from character counts
    # Gemini tokenises ~4 chars per token on average
    context_chars  = len(context)
    prompt_chars   = len(sensor_str) + len(fault_label) + context_chars
    response_chars = len(brief)
    input_tokens   = int(prompt_chars  / 4)
    output_tokens  = int(response_chars / 4)

    log.info(f"Track 1 — response generated  ({elapsed_ms}ms, ~{input_tokens}in/{output_tokens}out tokens)")

    return {
        "track":            1,
        "fault_class":      fault_class,
        "fault_label":      fault_label,
        "priority":         priority,
        "brief":            brief,
        "context_chunks":   len(retrieved),
        "response_time_ms": elapsed_ms,
        "rag_retrieval_ms": rag_ms,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 2 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_track2(
    chains: Dict[str, Any],
    risk_class: int,
    sensor_readings: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run the Track 2 chain: retrieve SOP context → generate Owner Push Alert in Bahasa Indonesia.

    Args:
        chains:          Output of build_chains()
        risk_class:      ML model output (0–3). Class 0 returns None (no alert).
        sensor_readings: Dict of sensor name → value from 12-hour daily window

    Returns:
        Dict with risk_class, risk_label, alert (text or None), response_time_ms
    """
    meta       = RISK_METADATA.get(risk_class, RISK_METADATA[0])
    risk_label = meta["label"]

    # Class 0 — no alert, no LLM call
    if risk_class == 0:
        log.info("Track 2 — Class 0 (No Risk). No alert generated.")
        return {
            "track":            2,
            "risk_class":       0,
            "risk_label":       "No Risk",
            "alert":            None,
            "context_chunks":   0,
            "response_time_ms": 0,
            "rag_retrieval_ms": 0,
            "input_tokens":     0,
            "output_tokens":    0,
        }

    sensor_str = "\n".join(
        f"  {k}: {v}" for k, v in sensor_readings.items()
    )

    # RAG retrieval — timed separately for monitoring
    rag_query = (
        f"risk class {risk_class} {risk_label} owner alert template "
        f"bahasa indonesia notification action"
    )
    retriever  = get_retriever(chains["vectorstore"], k=4)
    t_rag      = time.time()
    retrieved  = retriever.invoke(rag_query)
    rag_ms     = int((time.time() - t_rag) * 1000)
    context    = format_context(retrieved)

    log.info(f"Track 2 — running chain for Risk Class {risk_class}: {risk_label}")
    t0 = time.time()

    alert = chains["track2_chain"].invoke({
        "context":         context,
        "risk_class":      risk_class,
        "risk_label":      risk_label,
        "sensor_readings": sensor_str,
    })

    elapsed_ms = int((time.time() - t0) * 1000)

    # Estimate token usage (~4 chars per token)
    context_chars  = len(context)
    prompt_chars   = len(sensor_str) + len(risk_label) + context_chars
    response_chars = len(alert) if alert else 0
    input_tokens   = int(prompt_chars  / 4)
    output_tokens  = int(response_chars / 4)

    log.info(f"Track 2 — alert generated  ({elapsed_ms}ms, ~{input_tokens}in/{output_tokens}out tokens)")

    return {
        "track":            2,
        "risk_class":       risk_class,
        "risk_label":       risk_label,
        "alert":            alert,
        "context_chunks":   len(retrieved),
        "response_time_ms": elapsed_ms,
        "rag_retrieval_ms": rag_ms,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline import build_vectorstore

    print("\n" + "=" * 60)
    print("  LLM Chain — Self Test (Google Gemini)")
    print("=" * 60)

    vs     = build_vectorstore(force_rebuild=False)
    chains = build_chains(vs)

    # ── Track 1 test: Oil Pressure Issue (Class 6 — Critical) ──
    print("\n--- Track 1 Test: Class 6 — Oil Pressure Issue ---\n")
    t1_result = run_track1(
        chains,
        fault_class=6,
        sensor_readings={
            "O2 Sensor Voltage":    0.46,
            "MAF (g/s)":            5.9,
            "Throttle Position %":  15.3,
            "Crank RPM":            895,
            "Cam Advance (deg)":    10.4,
            "Knock Count (30d)":    0,
            "Coolant Temp (C)":     91.0,
            "Oil Pressure (PSI)":   17.5,
            "MAP (kPa)":            36.6,
            "EGR Duty %":           21.8,
            "Battery Voltage (V)":  14.02,
            "Fuel Temp (C)":        36.3,
        },
    )
    print(t1_result["brief"])
    print(f"\n[Response time: {t1_result['response_time_ms']}ms | "
          f"Context chunks: {t1_result['context_chunks']}]")

    # ── Track 2 test: High Risk (Class 3) ──
    print("\n--- Track 2 Test: Class 3 — High Risk ---\n")
    t2_result = run_track2(
        chains,
        risk_class=3,
        sensor_readings={
            "O2 Sensor Voltage":    0.68,
            "MAF (g/s)":            4.5,
            "Throttle Position %":  14.0,
            "Coolant Temp (C)":     112.0,
            "Oil Pressure (PSI)":   17.0,
            "Battery Voltage (V)":  11.2,
            "TPMS (PSI)":           24.0,
            "Ambient Temp (C)":     38.0,
            "Cabin Humidity %":     70.0,
            "Fuel Level %":         8.0,
            "Brake Pedal Events":   45,
            "Avg Speed (km/h)":     95.0,
        },
    )
    print(t2_result["alert"])
    print(f"\n[Response time: {t2_result['response_time_ms']}ms | "
          f"Context chunks: {t2_result['context_chunks']}]")

    print("\n" + "=" * 60)
    print("  Self-test complete.")
    print("=" * 60)
