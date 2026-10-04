"""Direct test of the explainer module with the Groq key."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.explainer import get_explainer

print("provider detected:", os.environ.get("GROQ_API_KEY") is not None)
explainer = get_explainer()
print("explainer provider:", explainer.provider)
result = explainer.explain(
    "reject",
    0.4664,
    {"review": 0.2263, "reject": 0.2584},
    None,
    [{"text": "Defect: Crack: Cracks are linear fractures that may appear on the ceramic surface. Any visible crack is an automatic reject.", "score": 0.5}],
)
print("source:", result["source"])
print(result["explanation"])
if "note" in result:
    print("NOTE:", result["note"])
