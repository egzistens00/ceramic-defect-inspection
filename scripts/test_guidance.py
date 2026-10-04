"""End-to-end LLM guidance test through the live API."""

import json
import sys
import urllib.request

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/tile/test/oil/002.png"

boundary = "----pantryboundary"
with open(image_path, "rb") as handle:
    payload = handle.read()
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="tile.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode() + payload + f"\r\n--{boundary}--\r\n".encode()

request = urllib.request.Request(
    "http://127.0.0.1:8901/api/v1/inspections",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    inspection = json.load(response)
print(f"INSPECTED: {image_path}")
print(f"tier={inspection['tier']}  score={inspection['score']}")

guidance_request = urllib.request.Request(
    f"http://127.0.0.1:8901/api/v1/quality-guidance/{inspection['inspection_id']}",
    data=b"",
    method="POST",
)
with urllib.request.urlopen(guidance_request, timeout=180) as response:
    guidance = json.load(response)
print(f"SOURCE: {guidance['source']}")
print()
print(guidance["explanation"])
if "note" in guidance:
    print("\nNOTE:", guidance["note"])
