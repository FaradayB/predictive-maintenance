"""
=============================================================================
 src/logger.py
 Vehicle Predictive Maintenance — Structured JSONL Request Logger
=============================================================================
 Writes one JSON line per request to logs/app.jsonl.
 Each line is a complete, self-contained record of a prediction request.

 Format (one line per request):
   {
     "timestamp": "2026-05-31T19:30:01.123Z",
     "request_id": "uuid4",
     "track": 1,
     "plate_number": "B 1234 ABC",
     "test_id": "TST1-6-001",
     "predicted_class": 6,
     "predicted_label": "Oil Pressure Issue",
     "true_class": 6,
     "true_label": "Oil Pressure Issue",
     "correct": true,
     "llm_called": true,
     "response_time_ms": 3200,
     "context_chunks": 4,
     "input_tokens": 842,
     "output_tokens": 412,
     "cost_usd": 0.000634,
     "safety_passed": true,
     "output_valid": true,
     "status": "ok",
     "error": null
   }

 This file feeds:
   - src/eval_runner.py  (accuracy / F1 calculation)
   - src/dashboard.py    (cost and performance reports)
=============================================================================
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR  = Path("logs")
LOG_FILE  = LOGS_DIR / "app.jsonl"

# Google Gemini pricing (as of May 2026 — update if rates change)
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

# Keep these for backward compatibility
COST_INPUT_PER_TOKEN  = 0.10  / 1_000_000
COST_OUTPUT_PER_TOKEN = 0.40  / 1_000_000
COST_EMBED_PER_TOKEN  = 0.025 / 1_000_000


def _ensure_log_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# COST CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    embed_tokens: int = 0,
    model: str = "gemini-2.0-flash",
) -> float:
    """
    Calculate the estimated USD cost of a single request.

    Args:
        input_tokens:  Tokens in the LLM prompt (context + question)
        output_tokens: Tokens in the LLM response
        embed_tokens:  Tokens embedded for RAG retrieval (optional)
        model:         LLM model name for accurate pricing

    Returns:
        Estimated cost in USD
    """
    pricing    = PRICING.get(model, PRICING["default"])
    llm_cost   = (
        (input_tokens  / 1_000_000) * pricing["input_per_1m"] +
        (output_tokens / 1_000_000) * pricing["output_per_1m"]
    )
    embed_cost = embed_tokens * COST_EMBED_PER_TOKEN
    return round(llm_cost + embed_cost, 8)


# ─────────────────────────────────────────────────────────────────────────────
# LOG ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def log_request(
    track: int,
    plate_number: str,
    predicted_class: int,
    predicted_label: str,
    response_time_ms: int,
    context_chunks: int,
    llm_called: bool,
    test_id: Optional[str]       = None,
    true_class: Optional[int]    = None,
    true_label: Optional[str]    = None,
    input_tokens: int            = 0,
    output_tokens: int           = 0,
    embed_tokens: int            = 0,
    model: str                   = "gemini-2.0-flash",
    rag_retrieval_ms: int        = 0,
    safety_passed: bool          = True,
    output_valid: bool           = True,
    status: str                  = "ok",
    error: Optional[str]         = None,
) -> str:
    """
    Write a structured log entry to logs/app.jsonl.

    Returns:
        The request_id (UUID) of the logged entry.
    """
    _ensure_log_dir()

    request_id = str(uuid.uuid4())
    cost       = calculate_cost(input_tokens, output_tokens, embed_tokens, model)
    correct    = (predicted_class == true_class) if true_class is not None else None

    entry: Dict[str, Any] = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "request_id":       request_id,
        "track":            track,
        "plate_number":     plate_number,
        "test_id":          test_id,
        "predicted_class":  predicted_class,
        "predicted_label":  predicted_label,
        "true_class":       true_class,
        "true_label":       true_label,
        "correct":          correct,
        "llm_called":       llm_called,
        "model":            model,
        "response_time_ms": response_time_ms,
        "rag_retrieval_ms": rag_retrieval_ms,
        "context_chunks":   context_chunks,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "embed_tokens":     embed_tokens,
        "cost_usd":         cost,
        "input_price_per_1m":  PRICING.get(model, PRICING["default"])["input_per_1m"],
        "output_price_per_1m": PRICING.get(model, PRICING["default"])["output_per_1m"],
        "safety_passed":    safety_passed,
        "output_valid":     output_valid,
        "status":           status,
        "error":            error,
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log.debug(f"Logged request {request_id} | T{track} | {plate_number} | {predicted_label}")
    except Exception as e:
        log.error(f"Failed to write log entry: {e}")

    return request_id


# ─────────────────────────────────────────────────────────────────────────────
# LOG READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_logs(
    track: Optional[int] = None,
    limit: Optional[int] = None,
) -> list:
    """
    Read all log entries from app.jsonl.

    Args:
        track: Filter by track (1 or 2). None = all tracks.
        limit: Return only the last N entries.

    Returns:
        List of dicts, most recent first.
    """
    if not LOG_FILE.exists():
        return []

    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if track is None or entry.get("track") == track:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    # Most recent first
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    return entries[:limit] if limit else entries


def read_logs_as_df():
    """
    Read all log entries as a pandas DataFrame.
    Used by eval_runner.py and dashboard.py.
    """
    import pandas as pd
    entries = read_logs()
    if not entries:
        return pd.DataFrame()
    return pd.DataFrame(entries)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Logger — Self Test ===\n")

    # Write some mock entries
    ids = []
    mock_requests = [
        (1, "B 1234 ABC", 6, "Oil Pressure Issue",  3200, 4, True,  "TST1-6-001", 6, "Oil Pressure Issue",  842, 412),
        (1, "T 4567 BCD", 0, "Normal",               0,    0, False, "TST1-0-005", 0, "Normal",              0,   0),
        (2, "B 1234 ABC", 3, "High Risk",            2900, 4, True,  "TST2-3-013", 3, "High Risk",           620, 310),
        (2, "B 7890 NOP", 0, "No Risk",              0,    0, False, "TST2-0-001", 0, "No Risk",             0,   0),
    ]

    for track, plate, pred_cls, pred_lbl, resp_ms, chunks, llm, tid, true_cls, true_lbl, inp, out in mock_requests:
        rid = log_request(
            track=track, plate_number=plate,
            predicted_class=pred_cls, predicted_label=pred_lbl,
            response_time_ms=resp_ms, context_chunks=chunks,
            llm_called=llm, test_id=tid,
            true_class=true_cls, true_label=true_lbl,
            input_tokens=inp, output_tokens=out,
        )
        ids.append(rid)
        print(f"  Logged: {rid[:8]}... | T{track} | {pred_lbl}")

    # Read back
    print(f"\n  Total entries in log: {len(read_logs())}")
    print(f"  Track 1 entries:      {len(read_logs(track=1))}")
    print(f"  Track 2 entries:      {len(read_logs(track=2))}")

    # Cost check
    cost = calculate_cost(input_tokens=842, output_tokens=412)
    print(f"\n  Cost for 842 input + 412 output tokens: ${cost:.8f}")

    df = read_logs_as_df()
    if not df.empty:
        print(f"\n  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Total cost logged: ${df['cost_usd'].sum():.6f}")

    print("\n=== Self-test complete ===")
