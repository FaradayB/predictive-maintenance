# PredictiveCare — AI-Powered Predictive Maintenance System
**Hacktiv8 LLM Bootcamp — Final Project (PTP Program)**

> *"Most vehicles don't break down from bad parts — they break down because no one saw it coming."*

PredictiveCare is an end-to-end AI system that detects vehicle fault patterns **before warning lights appear**, delivering a structured diagnostic brief to workshop technicians and a plain-language alert to vehicle owners, grounded in Standard Operating Procedure documents via a RAG pipeline.

---

## System Architecture

```
Vehicle Sensors & OBD
        │
        ▼
  IoT Layer  (every 2 hours, 07:00 – 19:00 WIB)
        │
        ▼
  PostgreSQL Database
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
Track 1 — Workshop / Mechanic        Track 2 — Vehicle Owner
30-day telemetry → AVG → ML          12-hr readings → AVG → ML
8 fault classes                      4 risk classes
        │                                      │
   [Normal] → No action              [No Risk] → No notification
        │                                      │
   [Anomaly] ↓                          [Risk] ↓
        │                                      │
LLM + RAG (Gemini 2.0 Flash)         LLM Summarizer (Gemini 2.0 Flash)
SOP: sop_track1_*.md                 SOP: sop_track2_*.md
ChromaDB vector store                ChromaDB vector store
        │                                      │
        ▼                                      ▼
Technician Fault Brief               Push Alert (owner-facing)   
(English, structured, cited)         (Plain language, no sensor values)
+ Follow-up Chat (English)           + Follow-up Chat (English)
        │                                      │
apps/technician.py :8501              apps/owner.py :8502

FastAPI Backend :8010 ← both apps call this
Prometheus :9090 ← scrapes :8000/metrics
Grafana :3000    ← reads from Prometheus
Nginx :80        ← routes domains to correct app
```

---

## Project Structure

```
predictivecare/
├── src/predictivecare/            # Installable package (pip install -e .)
│   ├── config.py                  # Centralised keys, model names, paths
│   ├── features.py                # Canonical feature order (shared: train + inference)
│   ├── api.py                     # FastAPI backend: prediction endpoints
│   ├── rag.py                     # RAG: load, chunk, embed, store, retrieve
│   ├── llm.py                     # LLM prompt chains (Track 1 + Track 2)
│   ├── database.py                # PostgreSQL connection, queries, averaging
│   ├── safety.py                  # Input validation + output guardrails
│   ├── logger.py                  # Structured JSONL request logging
│   ├── monitoring.py              # Prometheus metrics
│   └── dashboard.py               # Cost + performance report generator
│
├── apps/
│   ├── technician.py              # Streamlit tablet UI (Track 1, :8501)
│   └── owner.py                   # Streamlit mobile UI (Track 2, :8502)
│
├── ml/
│   ├── train.py                   # Train + select both classifiers -> models/
│   └── evaluate_llm.py            # LLM-as-judge over generated briefs/alerts
│
├── docs/                          # SOP knowledge base (RAG corpus, Track 1 + 2)
├── data/                          # Training dataset + test set (.xlsx)
├── models/                        # Trained .pkl (gitignored; run ml/train.py)
├── deploy/                        # nginx.conf, prometheus.yml, grafana/
├── tests/                         # Unit tests (safety, features, config)
├── scripts/stress_test.py
│
├── Dockerfile, docker-compose.yml # 7 services: apps + db + monitoring + proxy
├── pyproject.toml, requirements.txt
└── .env.example                   # Copy to .env and add your keys
```

---

## Quick Start — Docker (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/predictivecare.git
cd predictivecare
```

### 2. Configure environment variables

```bash
cp .env.example .env
nano .env
```

Fill in:

```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_EMBEDDING=models/text-embedding-004
GOOGLE_MODEL=gemini-2.0-flash
DB_PASSWORD=StrongPassword123
GRAFANA_PASSWORD=YourGrafanaPassword123
```

Get a free Google API key at: https://aistudio.google.com/app/apikey

### 3. Edit deploy/nginx.conf with your domain

```bash
nano deploy/nginx.conf
```

Replace all 4 occurrences of `YOURDOMAIN.COM` with your actual domain.

### 4. Build and start all services

```bash
docker compose up -d --build
```

Starts 7 containers: PostgreSQL, FastAPI, Technician app, Owner app, Prometheus, Grafana, Nginx.

### 5. Seed database and build vector store

```bash
docker compose exec api python -m predictivecare.database
docker compose exec api python -m predictivecare.rag
```

### 6. Verify everything is running

```bash
docker compose ps
curl http://localhost:8010/health
```

### Run locally without Docker

```bash
pip install -e .                   # install the predictivecare package
python ml/train.py                 # train the models into models/ (gitignored)
python -m predictivecare.rag       # build the RAG vector store (needs GOOGLE_API_KEY)
uvicorn predictivecare.api:app --port 8010
# then, in separate terminals:
streamlit run apps/technician.py --server.port 8501
streamlit run apps/owner.py --server.port 8502
```

### 7. Generate evaluation data

```bash
docker compose exec api python3 -m src.eval_runner --track both
```

---

## Access URLs

| Service | URL | Device |
|---|---|---|
| Technician App | `http://technician.YOURDOMAIN.COM` | Tablet (landscape) |
| Owner App | `http://owner.YOURDOMAIN.COM` | Phone (portrait) |
| Grafana Dashboard | `http://grafana.YOURDOMAIN.COM` | Laptop |
| FastAPI Docs | `http://api.YOURDOMAIN.COM/docs` | Browser |
| Prometheus UI | `http://YOUR_IP:9090` | Browser |

---

## Grafana Setup

1. Open Grafana → **Connections → Data sources → Add → Prometheus**
2. URL: `http://prometheus:9090`
3. Click **Save & Test**
4. Create dashboard with these panels:

| Panel | PromQL Query | Visualization |
|---|---|---|
| Total Queries by Track | `sum by (track) (vpm_queries_total)` | Stat |
| LLM Response Time p95 | `histogram_quantile(0.95, rate(vpm_response_time_seconds_bucket[5m]))` | Gauge |
| Fault Class Distribution | `vpm_fault_class_total` | Pie chart |
| Risk Class Distribution | `vpm_risk_class_total` | Pie chart |
| Query Rate per Minute | `rate(vpm_queries_total[1m])` | Time series |
| Active Sessions | `vpm_active_sessions` | Stat |
| Response Time Distribution | `rate(vpm_response_time_seconds_bucket[5m])` | Time series |
| Healthy Vehicle Skips | `vpm_normal_skip_total` | Stat |

---

## Docker Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs api
docker compose logs technician

# Restart one service
docker compose restart api

# Run a command inside a container
docker compose exec api python -m predictivecare.database

# Rebuild after code changes
docker compose up -d --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System component status |
| GET | `/api/v1/plates?track=1` | List all plate numbers |
| POST | `/api/v1/track1/diagnose` | Track 1 — fault classification + LLM brief |
| POST | `/api/v1/track2/alert` | Track 2: risk detection + English owner alert |

Full interactive docs: `http://api.YOURDOMAIN.COM/docs`

---

## ML Models

`ml/train.py` trains seven candidate classifiers per track inside a
StandardScaler pipeline and keeps the best per track by 5-fold cross-validated
weighted F1. The saved artifact is the full pipeline, so inference does not scale
features separately. Trained on simulated sensor data.

| Track | Selected model | Classes | Weighted F1 (test) |
|---|---|---|---|
| Track 1: Fault Classifier | best of 7 by CV F1 | 8 fault types | ~0.97 |
| Track 2: Risk Classifier | best of 7 by CV F1 | 4 risk levels | ~0.99 |

Selection is data-driven (Extra Trees currently wins both tracks). Run
`python ml/train.py` to reproduce the models and print the full per-candidate table.

**Input:** 30-day averaged telemetry (Track 1) or 7-reading daily average (Track 2)

To retrain from scratch:
```bash
python ml/train.py
```

---

## RAG Pipeline

| Stage | Implementation | Detail |
|---|---|---|
| 1. Document Loading | `TextLoader` | 2 SOP `.md` files from `docs/` |
| 2. Chunking | `RecursiveCharacterTextSplitter` | 500 chars / 50 overlap, markdown-aware |
| 3. Embedding | Google `text-embedding-004` | `retrieval_document` for index, `retrieval_query` for search |
| 4. Vector Store | ChromaDB | Persisted to `chroma_db/`, ~80 vectors |
| 5. Retrieval | MMR | k=4, lambda=0.7 — balances relevance + diversity |

---

## Dataset

| Dataset | Rows | Structure | Used For |
|---|---|---|---|
| Track 1 — Technician | 600 | 20 owners × 30 daily readings | 30-day fault trend detection |
| Track 2 — Owner | 140 | 20 owners × 7 hourly readings | Daily risk classification |

Fault and risk classes are **coherently aligned** — an owner with High Risk in Track 2 has a corresponding critical fault in Track 1.

---

## Classification Reference

### Track 1 — Fault Classes

| Class | Fault | Priority |
|---|---|---|
| 0 | Normal | None — no action required |
| 1 | Battery Degradation | Medium — schedule within 14 days |
| 2 | Brake System Issue | High — inspect within 3 days |
| 3 | Cooling System Problem | High — inspect within 3 days |
| 4 | Engine Misfire | High — inspect within 3 days |
| 5 | Alternator Failure | Medium — schedule within 7 days |
| 6 | Oil Pressure Issue | **Critical — do not drive** |
| 7 | Transmission Problem | High — inspect within 3 days |

### Track 2: Risk Classes

| Class | Level | Owner Action |
|---|---|---|
| 0 | No Risk | No notification sent |
| 1 | Low Risk | Monitor 3-5 days |
| 2 | Medium Risk | Schedule service within 7 days |
| 3 | High Risk | Stop driving, call +62-800-000-0000 |

---

## Key Design Decisions

**Anti-hallucination:** LLM system prompt restricts output to retrieved SOP context only. `temperature=0.1` minimises creative deviation. Track 1 briefs always include an SOP REFERENCE section.

**Averaging before classification:** Both tracks average their readings (30-day for Track 1, 7-hourly for Track 2) before passing to the ML classifier — trend detection, not single-snapshot decisions.

**No-alert efficiency:** Class 0 / Risk 0 results skip the LLM entirely — zero API cost and zero latency for healthy vehicles.

**Task-type embeddings:** `text-embedding-004` uses `retrieval_document` for indexing and `retrieval_query` at retrieval time — Google's recommended split for better retrieval accuracy.

**Follow-up chat:** Both apps include a post-prediction chat grounded in the same SOP context. Technician chat answers in English; owner chat also answers in English and never reveals raw sensor values.

**Safety layer:** `src/safety.py` validates inputs (plate format, sensor ranges) before they reach the ML model, and validates LLM outputs (required sections, language detection) before they reach users.

---

## Target Users

**Workshop Technician** — receives a pre-inspection fault brief before touching the vehicle. Knows exactly what to inspect, why, and in what order — based on 30 days of OBD telemetry trend analysis.

**Vehicle Owner**: receives a plain-language push notification in English. Never sees raw sensor values or technical codes. Knows what to do and how urgently.

---

*PredictiveCare — Hacktiv8 LLM Bootcamp Final Project | June 2026*
