"""Centralised configuration.

Every model name, path, and credential is resolved here from the environment
(with sensible defaults) so the rest of the codebase never calls os.getenv
directly and the three modules can no longer disagree on model names. See
.env.example for the full list.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Google / Gemini ---------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
# Generation model (used by the LLM chains).
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
# Embedding model (used by the RAG pipeline). Must have a default so an unset
# env var never turns into model=None (which previously crashed the pipeline).
GOOGLE_EMBEDDING: str = os.getenv("GOOGLE_EMBEDDING", "models/text-embedding-004")

# --- Database ----------------------------------------------------------------
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: str = os.getenv("DB_PORT", "5432")
DB_NAME: str = os.getenv("DB_NAME", "vehicle_maintenance")
DB_USER: str = os.getenv("DB_USER", "vpm_user")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# --- Paths -------------------------------------------------------------------
# Resolve relative to the repo root (two levels up from this file:
# src/predictivecare/config.py -> repo root).
_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", str(_ROOT / "docs")))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(_ROOT / "models")))
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(_ROOT / "chroma_db")))
DATASET_PATH: str = os.getenv("DATASET_PATH", str(_ROOT / "data" / "Vehicle_Sensor_TestSet.xlsx"))

# --- Service wiring ----------------------------------------------------------
API_URL: str = os.getenv("API_URL", "http://localhost:8010")

TRACK1_MODEL_PATH: Path = MODELS_DIR / "track1_fault_classifier.pkl"
TRACK2_MODEL_PATH: Path = MODELS_DIR / "track2_risk_classifier.pkl"


def require_google_api_key() -> str:
    """Return the Google API key or raise a clear error if it is missing."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a free key: https://aistudio.google.com/app/apikey"
        )
    return GOOGLE_API_KEY
