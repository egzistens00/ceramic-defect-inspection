"""Inspect the raw Groq response structure."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

key = os.environ.get("GROQ_API_KEY")
response = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Explain in 2 sentences why a cracked ceramic tile should be rejected."}],
        "max_tokens": 300,
    },
    timeout=60,
)
print("status:", response.status_code)
import json

body = response.json()
message = body["choices"][0]["message"]
print("message keys:", list(message.keys()))
for key_name in ("content", "reasoning"):
    if message.get(key_name):
        print(f"--- {key_name} ---")
        print(message[key_name][:600])
