"""Reproduce the guidance endpoint failure directly to see the real error."""

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.explainer import get_explainer
from app.rag import get_rag

try:
    rag = get_rag()
    sections = rag.retrieve("tile inspection tier reject defect handling procedure oil", top_k=2)
    print("RAG OK:", [s["text"][:40] for s in sections])
    explainer = get_explainer()
    print("explainer provider:", explainer.provider)
    result = explainer.explain(
        tier="reject",
        score=0.3279,
        thresholds={"review": 0.2263, "reject": 0.2584},
        human_verdict=None,
        sop_sections=sections,
    )
    print("SOURCE:", result["source"])
    print(result["explanation"])
except Exception:
    traceback.print_exc()
