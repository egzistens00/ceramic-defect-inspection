"""LLM-powered inspection explanation with template fallback."""

import os
from functools import lru_cache

TIER_CONTEXT = {
    "accept": "The tile passed the automatic inspection.",
    "review": "The AI is uncertain and the tile requires human review.",
    "reject": "The tile failed the automatic inspection.",
}

PROMPT_TEMPLATE = """You are a quality-control assistant for a ceramic tile manufacturing line.
Explain an AI inspection result to a factory worker in simple English.

Inspection result:
- Decision tier: {tier}
- Anomaly score: {score} (review threshold {review}, reject threshold {reject})
- Anomaly location on the tile: {heat_region}
- Human verdict: {human_verdict}

Relevant sections from the quality control SOP:
{sop_sections}

Write 3 short paragraphs max:
1. What the inspection found, in plain words. IMPORTANT: the AI only detects
THAT something is abnormal and WHERE it is — it does not identify the defect
type. Tell the worker to examine the highlighted region ({heat_region}) and
compare it against the defect types described in the SOP sections above.
2. What the worker should do next, based strictly on the SOP sections above.
3. One sentence on why the score means this tier.
Do not invent procedures not in the SOP. Do not claim a specific defect type
was identified. Be concise."""


class Explainer:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.provider = "groq" if os.environ.get("GROQ_API_KEY") else ("openai" if os.environ.get("OPENAI_API_KEY") else None)

    def explain(self, tier: str, score: float, thresholds: dict, human_verdict: str | None, sop_sections: list[dict], heat_region: str = "not computed") -> dict:
        if self.provider:
            try:
                return self._llm_explain(tier, score, thresholds, human_verdict, sop_sections, heat_region)
            except Exception as error:
                fallback = self._template_explain(tier, score, thresholds, human_verdict, sop_sections, heat_region)
                fallback["note"] = f"LLM unavailable ({type(error).__name__}); template explanation shown"
                return fallback
        return self._template_explain(tier, score, thresholds, human_verdict, sop_sections, heat_region)

    def _llm_explain(self, tier: str, score: float, thresholds: dict, human_verdict: str | None, sop_sections: list[dict], heat_region: str) -> dict:
        sections_text = "\n\n".join(f"[SOP] {section['text'][:800]}" for section in sop_sections)
        prompt = PROMPT_TEMPLATE.format(
            tier=tier,
            score=score,
            review=thresholds.get("review"),
            reject=thresholds.get("reject"),
            human_verdict=human_verdict or "not yet reviewed",
            sop_sections=sections_text,
            heat_region=heat_region,
        )
        if self.provider == "groq":
            import httpx

            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200, "reasoning_effort": "low"},
                timeout=30,
            )
            response.raise_for_status()
            text = (response.json()["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty LLM response")
        else:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=300)
            text = completion.choices[0].message.content.strip()
        return {"explanation": text, "source": f"llm:{self.provider}", "sop_sections": sop_sections}

    def _template_explain(self, tier: str, score: float, thresholds: dict, human_verdict: str | None, sop_sections: list[dict], heat_region: str = "not computed") -> dict:
        context = TIER_CONTEXT.get(tier, "Inspection completed.")
        action = {
            "accept": "The tile can continue to packaging.",
            "review": "Place the tile on the inspection mat, compare it with the boundary samples using the heatmap as a guide, and record your verdict in the dashboard.",
            "reject": "Remove the tile from the line for quarantined inspection and log the machine and batch number.",
        }.get(tier, "Follow the SOP.")
        verdict = f" A human reviewer labeled this tile as {human_verdict}." if human_verdict else ""
        return {
            "explanation": f"{context} The anomaly score is {score}, with the review threshold at {thresholds.get('review')} and the reject threshold at {thresholds.get('reject')}. The anomaly is concentrated in the {heat_region}.{verdict} {action}",
            "source": "template",
            "sop_sections": sop_sections,
        }


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    return Explainer()
