"""Database layer with automatic backend selection.

Uses SQL Server when INSPECTION_DB is configured (local development).
Falls back to a local SQLite file when it is not (free-tier deployment).
The interface is identical either way.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[1] / "inspections.db"
_CONNECTION_STRING = os.environ.get("INSPECTION_DB")

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score REAL NOT NULL,
    tier TEXT NOT NULL,
    heat TEXT,
    defect_type TEXT,
    defect_confidence REAL,
    reviewed INTEGER NOT NULL DEFAULT 0,
    reviewer_label TEXT,
    created_at TEXT NOT NULL
);
"""

_COLUMNS = "inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, reviewed, reviewer_label, created_at"


def _adapt(value):
    if value is None:
        return None
    if hasattr(value, "hex"):
        return value.hex()
    return value


class _Row:
    """Attribute-style row access that works for both pyodbc and sqlite3 rows."""

    def __init__(self, data: dict):
        self.__dict__.update(data)


def _sqlite_row_to_dict(row) -> dict:
    keys = ["inspection_id", "filename", "model_version", "score", "tier", "heat", "defect_type", "defect_confidence", "reviewed", "reviewer_label", "created_at"]
    return _Row(dict(zip(keys, row)))


def init_schema() -> None:
    if not _CONNECTION_STRING:
        with sqlite3.connect(_DB_PATH) as connection:
            connection.executescript(SQLITE_SCHEMA)
        return
    import pyodbc

    with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
        connection.execute(SCHEMA)
        connection.commit()


def insert_inspection(inspection_id: str, filename: str, model_version: str, score: float, tier: str, heat: str | None = None, defect_type: str | None = None, defect_confidence: float | None = None) -> None:
    if not _CONNECTION_STRING:
        with sqlite3.connect(_DB_PATH) as connection:
            connection.execute(
                "INSERT INTO inspections (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, reviewed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, datetime.now(timezone.utc).isoformat()),
            )
        return
    import pyodbc

    with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
        connection.execute(
            "INSERT INTO dbo.inspections (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uuid.UUID(inspection_id), filename, model_version, score, tier, heat, defect_type, defect_confidence),
        )
        connection.commit()


def fetch_inspection(inspection_id: str):
    if not _CONNECTION_STRING:
        with sqlite3.connect(_DB_PATH) as connection:
            row = connection.execute(f"SELECT {_COLUMNS} FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
            return _sqlite_row_to_dict(row) if row else None
    import pyodbc

    with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
        cursor = connection.execute(f"SELECT {_COLUMNS} FROM dbo.inspections WHERE inspection_id = ?", (uuid.UUID(inspection_id),))
        return cursor.fetchone()


def fetch_inspections(limit: int = 50, tier: str | None = None):
    if not _CONNECTION_STRING:
        query = f"SELECT {_COLUMNS} FROM inspections"
        params: list = []
        if tier:
            query += " WHERE tier = ?"
            params.append(tier)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(_DB_PATH) as connection:
            return [_sqlite_row_to_dict(row) for row in connection.execute(query, params).fetchall()]
    import pyodbc

    query = f"SELECT {_COLUMNS} FROM dbo.inspections"
    params = []
    if tier:
        query += " WHERE tier = ?"
        params.append(tier)
    query += " ORDER BY created_at DESC OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY"
    params.append(limit)
    with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
        return connection.execute(query, params).fetchall()


def record_feedback(inspection_id: str, reviewer_label: str) -> int:
    if not _CONNECTION_STRING:
        with sqlite3.connect(_DB_PATH) as connection:
            cursor = connection.execute("UPDATE inspections SET reviewed = 1, reviewer_label = ? WHERE inspection_id = ?", (reviewer_label, inspection_id))
            return cursor.rowcount
    import pyodbc

    with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
        cursor = connection.execute("UPDATE dbo.inspections SET reviewed = 1, reviewer_label = ? WHERE inspection_id = ?", (reviewer_label, uuid.UUID(inspection_id)))
        connection.commit()
        return cursor.rowcount


def fetch_metrics() -> dict:
    if not _CONNECTION_STRING:
        with sqlite3.connect(_DB_PATH) as connection:
            total, accepted, review, rejected, reviewed = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN tier = 'accept' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'review' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'reject' THEN 1 ELSE 0 END), SUM(reviewed) FROM inspections"
            ).fetchone()
            feedback = connection.execute("SELECT tier, reviewer_label, COUNT(*) FROM inspections WHERE reviewed = 1 GROUP BY tier, reviewer_label").fetchall()
    else:
        import pyodbc

        with pyodbc.connect(_CONNECTION_STRING, timeout=15) as connection:
            total, accepted, review, rejected, reviewed = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN tier = 'accept' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'review' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'reject' THEN 1 ELSE 0 END), SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) FROM dbo.inspections"
            ).fetchone()
            feedback = connection.execute("SELECT tier, reviewer_label, COUNT(*) FROM dbo.inspections WHERE reviewed = 1 GROUP BY tier, reviewer_label").fetchall()
    return {
        "total": int(total or 0),
        "tiers": {"accept": int(accepted or 0), "review": int(review or 0), "reject": int(rejected or 0)},
        "reviewed": int(reviewed or 0),
        "feedback_breakdown": [{"tier": row[0], "human_label": row[1], "count": row[2]} for row in feedback],
    }


def row_to_dict(row) -> dict:
    return {
        "inspection_id": row.inspection_id if isinstance(row.inspection_id, str) else str(row.inspection_id),
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
