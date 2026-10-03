"""Tile inspection API."""

import io
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, field_validator

from . import database
from .explainer import get_explainer
from .inspector import MODEL_VERSION, get_inspector
from .rag import get_rag

app = FastAPI(title="Tile Inspection API", version="0.3.0")

VALID_LABELS = ("normal", "defective")


class FeedbackRequest(BaseModel):
    label: str

    @field_validator("label")
    @classmethod
    def check_label(cls, value: str) -> str:
        if value not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}")
        return value


@app.on_event("startup")
def startup():
    database.init_schema()


@app.get("/health")
def health():
    get_inspector()
    database.get_connection().close()
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/api/v1/inspections")
async def create_inspection(image: UploadFile = File(...)):
    payload = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(payload))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")
    result = get_inspector().inspect(pil_image)
    inspection_id = uuid.uuid4().hex
    database.insert_inspection(inspection_id, image.filename or "unknown", result["model_version"], result["score"], result["tier"])
    return {"inspection_id": inspection_id, "filename": image.filename, **result}


@app.get("/api/v1/inspections/{inspection_id}")
def get_inspection(inspection_id: str):
    row = database.fetch_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    return database.row_to_dict(row)


@app.get("/api/v1/inspections")
def list_inspections(limit: int = Query(50, ge=1, le=200), tier: str | None = Query(None)):
    rows = database.fetch_inspections(limit=limit, tier=tier)
    return {"count": len(rows), "inspections": [database.row_to_dict(row) for row in rows]}


@app.post("/api/v1/feedback/{inspection_id}")
def submit_feedback(inspection_id: str, request: FeedbackRequest):
    row = database.fetch_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    updated = database.record_feedback(inspection_id, request.label)
    if updated == 0:
        raise HTTPException(status_code=404, detail="inspection not found")
    return {"inspection_id": inspection_id, "human_label": request.label, "recorded": True}


@app.get("/api/v1/metrics")
def metrics():
    return database.fetch_metrics()


@app.post("/api/v1/quality-guidance/{inspection_id}")
def quality_guidance(inspection_id: str):
    row = database.fetch_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    record = database.row_to_dict(row)
    query = f"tile inspection tier {record['tier']} defect handling procedure"
    if record["reviewer_label"]:
        query += f" human verdict {record['reviewer_label']}"
    sop_sections = get_rag().retrieve(query, top_k=2)
    guidance = get_explainer().explain(
        tier=record["tier"],
        score=record["score"],
        thresholds={"review": round(get_inspector().review_threshold, 4), "reject": round(get_inspector().reject_threshold, 4)},
        human_verdict=record["reviewer_label"],
        sop_sections=sop_sections,
    )
    return {"inspection_id": inspection_id, **guidance}


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
