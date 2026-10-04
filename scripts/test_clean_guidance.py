"""Local test with the markdown-stripped prompt."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.explainer import get_explainer
from app.rag import get_rag

sections = get_rag().retrieve("defect rough surface handling procedure tier review", top_k=3)
result = get_explainer().explain(
    tier="review",
    score=0.2436,
    thresholds={"review": 0.2263, "reject": 0.2584},
    human_verdict=None,
    sop_sections=sections,
    heat_region="center region",
    defect_type="rough",
)
print("SOURCE:", result["source"])
print()
print(result["explanation"])
assert "**" not in result["explanation"], "markdown still present!"
print()
print("no markdown markers - clean")
