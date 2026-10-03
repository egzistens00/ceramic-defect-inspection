"""One-time helper: create the TileInspection database if missing."""

import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,1433;DATABASE=master;UID=sa;PWD=ScrewInsp2026!",
    timeout=10,
    autocommit=True,
)
conn.cursor().execute("IF DB_ID('TileInspection') IS NULL CREATE DATABASE TileInspection")
print("database ready")
conn.close()
