"""Tile inspection API."""

import io
import json
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
    database.insert_inspection(inspection_id, image.filename or "unknown", result["model_version"], result["score"], result["tier"], json.dumps(result["heat"]))
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


def describe_heat_region(heat: list[list[float]] | None) -> str:
    """Convert the heatmap grid into a human-readable tile region."""
    if not heat or not heat[0]:
        return "center region (no heatmap available)"
    rows = len(heat)
    cols = len(heat[0])
    max_value = max(max(row) for row in heat)
    hot_cells = [(r, c) for r, row in enumerate(heat) for c, value in enumerate(row) if value >= 0.85 * max_value]
    if not hot_cells:
        return "center region"
    avg_row = sum(r for r, _ in hot_cells) / len(hot_cells)
    avg_col = sum(c for _, c in hot_cells) / len(hot_cells)
    vertical = "upper" if avg_row < rows / 3 else "lower" if avg_row >= 2 * rows / 3 else "middle"
    horizontal = "left" if avg_col < cols / 3 else "right" if avg_col >= 2 * cols / 3 else "center"
    if horizontal == "center" and vertical == "middle":
        return "center region"
    return f"{vertical} {horizontal} region"


@app.post("/api/v1/quality-guidance/{inspection_id}")
@app.get("/api/v1/quality-guidance/{inspection_id}")
def quality_guidance(inspection_id: str):
    row = database.fetch_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    record = database.row_to_dict(row)
    inspector = get_inspector()
    query = "defect types crack glue strip gray stroke oil rough surface handling procedure " + f"tier {record['tier']}"
    if record["reviewer_label"]:
        query += f" human verdict {record['reviewer_label']}"
    sop_sections = get_rag().retrieve(query, top_k=3)
    heat_region = describe_heat_region(record.get("heat"))
    guidance = get_explainer().explain(
        tier=record["tier"],
        score=record["score"],
        thresholds={"review": round(inspector.review_threshold, 4), "reject": round(inspector.reject_threshold, 4)},
        human_verdict=record["reviewer_label"],
        sop_sections=sop_sections,
        heat_region=heat_region,
    )
    return {"inspection_id": inspection_id, "heat_region": heat_region, **guidance}


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
