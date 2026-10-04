"""Inference wrapper for the locked memory-bank inspector."""

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "inspector_model.pt"
MODEL_VERSION = "memory_bank_inspector_v1"


class Inspector:
    def __init__(self, artifact_path: Path = ARTIFACT):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(artifact_path, map_location=self.device)
        self.bank = data["bank"].to(self.device)
        self.defect_banks = {name: tensor.to(self.device) for name, tensor in data.get("defect_banks", {}).items()}
        self.review_threshold = float(data["review_threshold"])
        self.reject_threshold = float(data["reject_threshold"])
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights).to(self.device).eval()
        self.backbone.fc = torch.nn.Identity()
        self.captured: dict[str, torch.Tensor] = {}
        self.backbone.layer3.register_forward_hook(lambda module, inputs, output: self.captured.__setitem__("features", output))
        self.preprocess = weights.transforms()
        self.last_heat: list[list[float]] | None = None

    def inspect(self, image: Image.Image) -> dict:
        with torch.no_grad():
            self.backbone(self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device))
        feature_map = self.captured["features"].squeeze(0)
        height, width = feature_map.shape[1], feature_map.shape[2]
        flat = torch.nn.functional.normalize(feature_map.flatten(1).T, dim=1)
        similarity = flat @ self.bank.T
        distances = (1 - similarity.max(dim=1).values).reshape(height, width).cpu().numpy()
        score = float(distances.max())
        tier = "reject" if score > self.reject_threshold else "review" if score > self.review_threshold else "accept"
        defect_type, defect_confidence = self._identify_defect(flat)
        if tier == "accept":
            defect_type, defect_confidence = None, None
        self.last_heat = [[round(float(value), 4) for value in row] for row in distances]
        return {
            "model_version": MODEL_VERSION,
            "score": round(score, 4),
            "tier": tier,
            "defect_type": defect_type,
            "defect_confidence": defect_confidence,
            "thresholds": {"review": round(self.review_threshold, 4), "reject": round(self.reject_threshold, 4)},
            "heat": self.last_heat,
        }

    def _identify_defect(self, flat: torch.Tensor) -> tuple[str | None, float | None]:
        """Classify anomalous patches against per-defect-type banks (nearest-neighbor vote)."""
        if not self.defect_banks:
            return None, None
        distances = 1 - (flat @ self.bank.T).max(dim=1).values
        top = torch.topk(distances, 10).indices
        scores = {}
        for name, bank in self.defect_banks.items():
            similarity = flat[top] @ bank.T
            scores[name] = float((1 - similarity.max(dim=1).values).mean())
        best = min(scores, key=scores.get)
        sorted_scores = sorted(scores.values())
        margin = (sorted_scores[1] - sorted_scores[0]) / (sorted_scores[-1] + 1e-9)
        return best, round(margin, 4)


@lru_cache(maxsize=1)
def get_inspector() -> Inspector:
    return Inspector()
