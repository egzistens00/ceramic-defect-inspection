"""SQL Server persistence for inspection records."""

import json
import os
import uuid
from datetime import datetime, timezone

import pyodbc

CONNECTION_STRING = os.environ.get(
    "INSPECTION_DB",
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,1433;DATABASE=TileInspection;UID=sa;PWD=ScrewInsp2026!",
)

SCHEMA = """
IF OBJECT_ID('dbo.inspections', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.inspections (
        inspection_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        filename NVARCHAR(260) NOT NULL,
        model_version NVARCHAR(64) NOT NULL,
        score FLOAT NOT NULL,
        tier NVARCHAR(16) NOT NULL,
        heat NVARCHAR(MAX) NULL,
        reviewed BIT NOT NULL DEFAULT 0,
        reviewer_label NVARCHAR(16) NULL,
        created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
    );
END;
IF COL_LENGTH('dbo.inspections', 'heat') IS NULL
BEGIN
    ALTER TABLE dbo.inspections ADD heat NVARCHAR(MAX) NULL;
END;
IF COL_LENGTH('dbo.inspections', 'defect_type') IS NULL
BEGIN
    ALTER TABLE dbo.inspections ADD defect_type NVARCHAR(32) NULL;
END;
IF COL_LENGTH('dbo.inspections', 'defect_confidence') IS NULL
BEGIN
    ALTER TABLE dbo.inspections ADD defect_confidence FLOAT NULL;
END;
"""


def get_connection() -> pyodbc.Connection:
    return pyodbc.connect(CONNECTION_STRING, timeout=15)


def init_schema() -> None:
    with get_connection() as connection:
        connection.execute(SCHEMA)
        connection.commit()


def insert_inspection(inspection_id: str, filename: str, model_version: str, score: float, tier: str, heat: str | None = None, defect_type: str | None = None, defect_confidence: float | None = None) -> None:
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO dbo.inspections (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uuid.UUID(inspection_id), filename, model_version, score, tier, heat, defect_type, defect_confidence),
        )
        connection.commit()


def fetch_inspection(inspection_id: str):
    with get_connection() as connection:
        cursor = connection.execute("SELECT inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, reviewed, reviewer_label, created_at FROM dbo.inspections WHERE inspection_id = ?", (uuid.UUID(inspection_id),))
        return cursor.fetchone()


def fetch_inspections(limit: int = 50, tier: str | None = None):
    query = "SELECT inspection_id, filename, model_version, score, tier, reviewed, reviewer_label, created_at FROM dbo.inspections"
    params: list = []
    if tier:
        query += " WHERE tier = ?"
        params.append(tier)
    query += " ORDER BY created_at DESC OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY"
    params.append(limit)
    with get_connection() as connection:
        return connection.execute(query, params).fetchall()


def record_feedback(inspection_id: str, reviewer_label: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE dbo.inspections SET reviewed = 1, reviewer_label = ? WHERE inspection_id = ?",
            (reviewer_label, uuid.UUID(inspection_id)),
        )
        connection.commit()
        return cursor.rowcount


def fetch_metrics() -> dict:
    with get_connection() as connection:
        totals = connection.execute(
            "SELECT COUNT(*), SUM(CASE WHEN tier = 'accept' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'review' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'reject' THEN 1 ELSE 0 END), SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) FROM dbo.inspections"
        ).fetchone()
        feedback = connection.execute(
            "SELECT tier, reviewer_label, COUNT(*) FROM dbo.inspections WHERE reviewed = 1 GROUP BY tier, reviewer_label"
        ).fetchall()
    total, accepted, review, rejected, reviewed = totals
    return {
        "total": int(total or 0),
        "tiers": {"accept": int(accepted or 0), "review": int(review or 0), "reject": int(rejected or 0)},
        "reviewed": int(reviewed or 0),
        "feedback_breakdown": [{"tier": row[0], "human_label": row[1], "count": row[2]} for row in feedback],
    }


def row_to_dict(row) -> dict:
    return {
        "inspection_id": str(row.inspection_id),
        "filename": row.filename,
        "model_version": row.model_version,
        "score": row.score,
        "tier": row.tier,
        "heat": json.loads(row.heat) if getattr(row, "heat", None) else None,
        "defect_type": getattr(row, "defect_type", None),
        "defect_confidence": getattr(row, "defect_confidence", None),
        "reviewed": bool(row.reviewed),
        "reviewer_label": row.reviewer_label,
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else str(row.created_at),
    }
