# Ceramic Surface Defect Inspection System

An AI-powered visual quality inspection system for ceramic parts. A photo of a
ceramic tile goes in; a tiered accept / review / reject decision comes out —
with the defect location highlighted, every inspection stored in SQL Server,
and an explanation grounded in the quality-control SOP.

Built as a fresh-graduate portfolio project targeting industrial visual
inspection, the problem documented in recent ceramic-substrate manufacturing
research (see [Research context](#research-context)).

## How it works

```text
                    ┌─────────────────────────────────────────────┐
                    │                                             │
 ceramic part photo │   ┌──────────┐    ┌─────────────────────┐  │
 ─────────────────► │   │ ResNet18 │───►│ patch memory bank   │  │
                    │   │ backbone │    │ (normal tiles only) │  │
                    │   └──────────┘    └─────────┬───────────┘  │
                    │                           distance map
                    │                           per patch
                    │                                 │
                    │                    ┌────────────▼───────────┐
                    │                    │ max patch distance =   │
                    │                    │ anomaly score          │
                    │                    └────────────┬───────────┘
                    │                                 │
                    │        ┌────────────────────────┼──────────────┐
                    │        ▼                        ▼              ▼
                    │   score ≤ 0.226          0.226–0.258     score > 0.258
                    │      ACCEPT                 REVIEW          REJECT
                    │   (auto-pass)         (human inspects)   (auto-fail)
                    │        │                        │              │
                    │        └───────────┬────────────┘              │
                    │                    ▼                           │
                    │            ┌───────────────┐                   │
                    │            │ SQL Server    │◄── human verdicts │
                    │            │ inspection log│    (feedback loop │
                    │            └───────┬───────┘     for retrain) │
                    │                    ▼                           │
                    │            ┌───────────────┐                   │
                    │            │ RAG + LLM     │──► explanation &  │
                    │            │ (quality SOP) │    next action    │
                    │            └───────────────┘                   │
                    └─────────────────────────────────────────────┘
```

## The model, honestly

**Approach:** memory-bank anomaly detection. The model sees only *normal*
ceramic tiles during setup (230 training images). Each new part is split into
patches; every patch is compared against a bank of remembered normal patches.
Patches unlike anything normal mark the anomaly; the worst patch distance is
the anomaly score.

This mirrors real factories, where good parts are plentiful and defect
examples are rare.

**Results on the held-out test set** (117 images, thresholds calibrated on
validation normals only — the test set was never used for any decision):

| Measure | Result |
|---|---:|
| Accuracy | 91.5% |
| Defect recall | 89.3% (75/84 defects caught) |
| Defect precision | 98.7% |
| Good parts auto-accepted | 32/33 |

**Tier outcome on the test set:**

- 65 defective parts auto-rejected with no human involvement
- 32 good parts auto-accepted with no human involvement
- 20 cases routed to human review — the model admits uncertainty instead of guessing

**Known limitations** (stated plainly, because hiding them is worse):

- Image-level detection is below published PatchCore results (~0.97 AUROC on
  this category); this prototype uses ResNet18 features at modest resolution
  on CPU hardware
- The heatmap is a coarse 16×16 grid meant to guide an inspector's eye, not
  pixel-precise segmentation
- The prototype is trained on the public MVTec AD `tile` category as a proxy
  for production ceramic substrates; a production system would retrain on the
  factory's own parts and cameras

## Components

| Layer | Tech | File |
|---|---|---|
| Anomaly model | PyTorch, torchvision ResNet18 | `scripts/build_inspector.py` |
| API service | FastAPI | `app/main.py` |
| Inference | PyTorch, memory-bank nearest neighbor | `app/inspector.py` |
| Persistence | SQL Server 2022 (Docker) | `app/database.py` |
| Explanation | RAG: Hugging Face sentence embeddings + LLM API (Groq/OpenAI, template fallback) | `app/rag.py`, `app/explainer.py` |
| Dashboard | vanilla HTML/JS | `app/static/index.html` |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model version |
| `POST /api/v1/inspections` | upload part image → tier + score + anomaly heatmap |
| `GET /api/v1/inspections/{id}` | fetch one inspection record |
| `GET /api/v1/inspections` | inspection history |
| `POST /api/v1/feedback/{id}` | record human verdict (normal / defective) |
| `GET /api/v1/metrics` | aggregate stats: tier counts, review rate, feedback breakdown |
| `POST /api/v1/quality-guidance/{id}` | RAG + LLM explanation of a decision from the SOP |

## Run it

Prerequisites: Python 3.10+, Docker, the MVTec AD `tile` category extracted
to `data/tile/` (the dataset is CC BY-NC-SA 4.0 — download it from MVTec and
keep it out of any redistribution).

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. SQL Server (Docker)
docker run -d --name tile-inspection-sql \
  -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=<your-password>" \
  -p 1433:1433 mcr.microsoft.com/mssql/server:2022-latest
python scripts/create_database.py   # creates the TileInspection DB

# 3. build the model (trains on data/tile/train/good)
python scripts/build_inspector.py

# 4. serve
uvicorn app.main:app --port 8901
```

Then open `http://127.0.0.1:8901/` for the dashboard or `/docs` for the API.

Environment variables:

- `INSPECTION_DB` — SQL Server connection string (override)
- `GROQ_API_KEY` or `OPENAI_API_KEY` — enables real LLM explanations;
  without a key the system falls back to a deterministic template so the
  demo always works

## Why the three-tier design

An inspector that only outputs "defective / not defective" is unsafe to
deploy: a model WILL be wrong sometimes. The confidence bands route uncertain
cases to humans while still automating the clear-cut majority. Human verdicts
are stored next to the model's decision — that table is exactly the
retraining dataset a real deployment would use to improve the model, closing
the loop.

## Research context

Ceramic substrate inspection is an active industrial problem:

- "Defect detection of ceramic substrates based on improved YOLOv9" (Springer,
  2025) — notes low-contrast, subtle defects and noisy industrial datasets
- "A Real-Time Automated Defect Detection System for Ceramic Pieces" (MDPI
  Sensors, 2024) — deep learning systems replacing manual examination
- Machine-vision vendors (e.g. Intelgic) explicitly describe ceramics
  inspection as manual, time-consuming, and inconsistent today

This project is a fresh-graduate-scale prototype of that exact problem:
anomaly detection with calibrated thresholds, a human-review safety tier, an
inspection audit trail, and SOP-grounded explanations.

## Dataset

MVTec Anomaly Detection dataset, `tile` category (230 train / 117 test
images). Licensed CC BY-NC-SA 4.0, © MVTec Software GmbH — not included in
this repository.
