"""Database layer with automatic backend selection.

Behavior:
- If INSPECTION_DB is set AND SQL Server is reachable: SQL Server mode
  (local development, Docker container).
- Otherwise: falls back to a local SQLite file (inspections.db) with a
  console warning, so the application always starts.
"""

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[1] / "inspections.db"
_CONNECTION_STRING = os.environ.get("INSPECTION_DB")

SQLSERVER_SCHEMA = """
IF OBJECT_ID('dbo.inspections', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.inspections (
        inspection_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        filename NVARCHAR(260) NOT NULL,
        model_version NVARCHAR(64) NOT NULL,
        score FLOAT NOT NULL,
        tier NVARCHAR(16) NOT NULL,
        heat NVARCHAR(MAX) NULL,
        defect_type NVARCHAR(32) NULL,
        defect_confidence FLOAT NULL,
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

_backend: str | None = None


class _Row:
    """Attribute-style row access that works for both pyodbc and sqlite3 rows."""

    def __init__(self, data: dict):
        self.__dict__.update(data)


def _sqlite_row(row) -> _Row:
    return _Row(dict(zip(_COLUMNS.split(", "), row)))


def _sqlserver() -> bool:
    return _backend == "sqlserver"


def init_schema() -> None:
    global _backend
    if _CONNECTION_STRING:
        try:
            import pyodbc

            with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
                connection.execute(SQLSERVER_SCHEMA)
                connection.commit()
            _backend = "sqlserver"
            print("[database] backend: SQL Server")
            return
        except Exception as error:
            print(f"[database] SQL Server unreachable ({type(error).__name__}); falling back to SQLite")
    _backend = "sqlite"
    with closing(sqlite3.connect(_DB_PATH)) as connection:
        connection.executescript(SQLITE_SCHEMA)
        connection.commit()
    print("[database] backend: SQLite (inspections.db)")


def insert_inspection(inspection_id: str, filename: str, model_version: str, score: float, tier: str, heat: str | None = None, defect_type: str | None = None, defect_confidence: float | None = None) -> None:
    if _sqlserver():
        import pyodbc

        with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
            connection.execute(
                "INSERT INTO dbo.inspections (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (uuid.UUID(inspection_id), filename, model_version, score, tier, heat, defect_type, defect_confidence),
            )
            connection.commit()
    else:
        with closing(sqlite3.connect(_DB_PATH)) as connection:
            connection.execute(
                "INSERT INTO inspections (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, reviewed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (inspection_id, filename, model_version, score, tier, heat, defect_type, defect_confidence, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()


def fetch_inspection(inspection_id: str):
    if _sqlserver():
        import pyodbc

        with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
            cursor = connection.execute(f"SELECT {_COLUMNS} FROM dbo.inspections WHERE inspection_id = ?", (uuid.UUID(inspection_id),))
            return cursor.fetchone()
    with closing(sqlite3.connect(_DB_PATH)) as connection:
        row = connection.execute(f"SELECT {_COLUMNS} FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
        return _sqlite_row(row) if row else None


def fetch_inspections(limit: int = 50, tier: str | None = None):
    if _sqlserver():
        import pyodbc

        query = f"SELECT {_COLUMNS} FROM dbo.inspections"
        params: list = []
        if tier:
            query += " WHERE tier = ?"
            params.append(tier)
        query += " ORDER BY created_at DESC OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY"
        params.append(limit)
        with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
            return connection.execute(query, params).fetchall()
    query = f"SELECT {_COLUMNS} FROM inspections"
    params = []
    if tier:
        query += " WHERE tier = ?"
        params.append(tier)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with closing(sqlite3.connect(_DB_PATH)) as connection:
        return [_sqlite_row(row) for row in connection.execute(query, params).fetchall()]


def record_feedback(inspection_id: str, reviewer_label: str) -> int:
    if _sqlserver():
        import pyodbc

        with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
            cursor = connection.execute("UPDATE dbo.inspections SET reviewed = 1, reviewer_label = ? WHERE inspection_id = ?", (reviewer_label, uuid.UUID(inspection_id)))
            connection.commit()
            return cursor.rowcount
    with closing(sqlite3.connect(_DB_PATH)) as connection:
        cursor = connection.execute("UPDATE inspections SET reviewed = 1, reviewer_label = ? WHERE inspection_id = ?", (reviewer_label, inspection_id))
        connection.commit()
        return cursor.rowcount


def fetch_metrics() -> dict:
    if _backend is None:
        init_schema()
    if _sqlserver():
        import pyodbc

        with closing(pyodbc.connect(_CONNECTION_STRING, timeout=15)) as connection:
            total, accepted, review, rejected, reviewed = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN tier = 'accept' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'review' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'reject' THEN 1 ELSE 0 END), SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) FROM dbo.inspections"
            ).fetchone()
            feedback = connection.execute("SELECT tier, reviewer_label, COUNT(*) FROM dbo.inspections WHERE reviewed = 1 GROUP BY tier, reviewer_label").fetchall()
    else:
        with closing(sqlite3.connect(_DB_PATH)) as connection:
            total, accepted, review, rejected, reviewed = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN tier = 'accept' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'review' THEN 1 ELSE 0 END), SUM(CASE WHEN tier = 'reject' THEN 1 ELSE 0 END), SUM(reviewed) FROM inspections"
            ).fetchone()
            feedback = connection.execute("SELECT tier, reviewer_label, COUNT(*) FROM inspections WHERE reviewed = 1 GROUP BY tier, reviewer_label").fetchall()
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
