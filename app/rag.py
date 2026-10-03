"""RAG engine: chunk the SOP, embed with Hugging Face, retrieve relevant sections."""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

SOP_PATH = Path(__file__).resolve().parents[1] / "data" / "quality_sop.md"


def load_chunks() -> list[str]:
    text = SOP_PATH.read_text(encoding="utf-8")
    sections = re.split(r"\n## ", text)
    chunks = []
    for section in sections:
        clean = section.strip()
        if len(clean) > 40:
            title_end = clean.split("\n", 1)
            title = title_end[0].strip().lstrip("# ")
            body = title_end[1].strip() if len(title_end) > 1 else ""
            chunks.append(f"{title}: {body}")
    return chunks


class RagEngine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = load_chunks()
        self.embeddings = self.model.encode(self.chunks, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.asarray(self.embeddings @ query_embedding)
        order = np.argsort(-scores)[:top_k]
        return [{"text": self.chunks[i], "score": float(scores[i])} for i in order]


@lru_cache(maxsize=1)
def get_rag() -> RagEngine:
    return RagEngine()
