"""Build the final inspector: memory-bank detection + localization (locked).

Architecture decision (see model_card.json history):
- ResNet18 layer3 patch memory bank for BOTH detection (max patch distance)
  and localization (distance heatmap).
- Threshold calibrated on 40 held-out validation normals; test evaluated once.
- A 3-tier confidence policy turns imperfect scores into a safe decision:
  accept (auto-pass), reject (auto-fail), review (human inspection).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

SEED = 42
VALIDATION_COUNT = 40


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights).to(device).eval()
    backbone.fc = torch.nn.Identity()
    captured: dict[str, torch.Tensor] = {}
    backbone.layer3.register_forward_hook(lambda module, inputs, output: captured.__setitem__("features", output))
    preprocess = weights.transforms()

    def patch_features(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.no_grad():
            backbone(preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
        feature_map = captured["features"].squeeze(0)
        channels, height, width = feature_map.shape
        flat = feature_map.flatten(1).T
        return torch.nn.functional.normalize(flat, dim=1), (height, width)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    shuffled = normal_files[:]
    random.Random(SEED).shuffle(shuffled)
    validation_files, reference_files = shuffled[:VALIDATION_COUNT], shuffled[VALIDATION_COUNT:]

    bank = torch.cat([patch_features(path)[0] for path in reference_files])

    def distance_map(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        flat, shape = patch_features(path)
        similarity = flat @ bank.T
        return (1 - similarity.max(dim=1).values).reshape(shape).cpu().numpy(), shape

    validation_scores = [float(distance_map(path)[0].max()) for path in validation_files]
    review_threshold = float(np.percentile(validation_scores, 95))
    reject_threshold = float(np.percentile(validation_scores, 99))

    records = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        for image_path in sorted(defect_dir.glob("*.png")):
            heat, shape = distance_map(image_path)
            records.append({"defect": defect_dir.name, "score": float(heat.max()), "shape": shape})

    labels = [0 if record["defect"] == "good" else 1 for record in records]
    scores = [record["score"] for record in records]
    predicted = [int(score > review_threshold) for score in scores]
    tiers = ["reject" if score > reject_threshold else "review" if score > review_threshold else "accept" for score in scores]
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "review_threshold": review_threshold,
        "reject_threshold": reject_threshold,
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "report": classification_report(labels, predicted, target_names=["normal", "defective"], output_dict=True),
        "tier_counts": {tier: tiers.count(tier) for tier in ("accept", "review", "reject")},
        "tier_outcome": {
            "auto_accepted_good": sum(1 for label, tier in zip(labels, tiers) if label == 0 and tier == "accept"),
            "auto_rejected_defective": sum(1 for label, tier in zip(labels, tiers) if label == 1 and tier == "reject"),
            "sent_to_human": sum(1 for tier in tiers if tier != "accept"),
        },
    }

    torch.save(
        {
            "bank": bank.cpu(),
            "review_threshold": review_threshold,
            "reject_threshold": reject_threshold,
            "backbone": "resnet18-layer3-imagenet",
            "components": {"detection": "memory_bank_max_patch_distance", "localization": "memory_bank_distance_heatmap"},
        },
        project / "artifacts" / "inspector_model.pt",
    )
    model_card = {
        "selected_model": "memory_bank_inspector",
        "method": "ResNet18 layer3 feature map; each patch scored by cosine distance to its nearest neighbor in a bank of reference-normal patches; single mechanism for detection (max distance) and localization (distance heatmap)",
        "confidence_policy": {
            "accept": f"score <= {review_threshold:.4f} (auto-pass)",
            "review": f"{review_threshold:.4f} < score <= {reject_threshold:.4f} (human inspection)",
            "reject": f"score > {reject_threshold:.4f} (auto-fail)",
        },
        "threshold_policy": "review = p95, reject = p99 of max-patch scores on 40 held-out validation normals; test evaluated once",
        "split": {"seed": SEED, "reference_normals": len(reference_files), "validation_normals": len(validation_files)},
        "test_metrics": metrics,
        "known_limitations": [
            "image-level AUC 0.838 is below published PatchCore results (~0.97 on tile) which use WideResNet-50 features at 224px and GPU-grade tuning",
            "localization heatmap is coarse (16x16 grid) and intended to guide operators, not replace pixel-precise segmentation",
            "train/test exposure shift exists in this category; per-image standardization removed the shift but also the signal, indicating the autoencoder's apparent strength was partly exposure-driven",
        ],
        "rejected_baselines": [
            {"model": "convolutional_autoencoder_raw", "image_level_roc_auc": 0.996, "reason": "ranking relied partly on train/test exposure shift; validation-calibrated threshold could not transfer (defective recall 1/119)"},
            {"model": "convolutional_autoencoder_standardized", "image_level_roc_auc": 0.477, "reason": "removing brightness signal destroyed ranking"},
            {"model": "mean_reference_grid_patches", "image_level_roc_auc": 0.501, "reason": "rotation variance defeats a single averaged reference"},
            {"model": "memory_bank_pooled_grid_patches", "image_level_roc_auc": 0.744, "reason": "4x4 grid with pooled final-layer features too coarse"},
        ],
    }
    (project / "artifacts" / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
