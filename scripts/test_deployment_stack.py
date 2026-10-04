"""Full-stack smoke test with SQLite backend (simulates free-tier deployment)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

health = client.get("/health")
print("health:", health.status_code, health.json()["status"])

with open("data/tile/test/oil/002.png", "rb") as handle:
    response = client.post("/api/v1/inspections", files={"image": ("oil.png", handle, "image/png")})
data = response.json()
print("inspection:", response.status_code, data["tier"], data["score"], "defect:", data["defect_type"])

guidance = client.post(f"/api/v1/quality-guidance/{data['inspection_id']}").json()
print("guidance source:", guidance["source"])
print("guidance:", guidance["explanation"][:180])

listing = client.get("/api/v1/inspections?limit=5").json()
print("history count:", listing["count"])
metrics = client.get("/api/v1/metrics").json()
print("metrics total:", metrics["total"])

feedback = client.post(f"/api/v1/feedback/{data['inspection_id']}", json={"label": "defective"})
print("feedback:", feedback.status_code, feedback.json())
