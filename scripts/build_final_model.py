"""Build the final detector: PatchCore-style memory bank on intermediate features."""

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
    model = resnet18(weights=weights).to(device).eval()
    captured: dict[str, torch.Tensor] = {}
    model.layer3.register_forward_hook(lambda module, inputs, output: captured.__setitem__("features", output))
    preprocess = weights.transforms()

    def patch_bank(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.no_grad():
            model(preprocess(image.convert("RGB")).unsqueeze(0).to(device))
        feature_map = captured["features"].squeeze(0)
        channels, height, width = feature_map.shape
        flat = feature_map.flatten(1).T
        return torch.nn.functional.normalize(flat, dim=1).cpu(), (height, width)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    shuffled = normal_files[:]
    random.Random(SEED).shuffle(shuffled)
    validation_files, reference_files = shuffled[:VALIDATION_COUNT], shuffled[VALIDATION_COUNT:]

    bank = torch.cat([patch_bank(Image.open(path))[0] for path in reference_files])

    def distance_map(image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        flat, shape = patch_bank(image)
        similarity = flat @ bank.T
        return (1 - similarity.max(dim=1).values).reshape(shape).numpy(), shape

    validation_scores = [float(distance_map(Image.open(path))[0].max()) for path in validation_files]
    threshold = float(np.percentile(validation_scores, 95))

    records = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        for image_path in sorted(defect_dir.glob("*.png")):
            distances, shape = distance_map(Image.open(image_path))
            score = float(distances.max())
            predicted_cells = distances >= 0.5 * distances.max()
            iou = None
            if defect_dir.name != "good":
                mask = np.asarray(Image.open(project / "data/tile/ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png").convert("L").resize(shape, Image.NEAREST)) > 0
                intersection = np.logical_and(predicted_cells, mask).sum()
                union = np.logical_or(predicted_cells, mask).sum()
                iou = float(intersection / union) if union else 1.0
            records.append({"defect": defect_dir.name, "score": score, "iou": iou})

    labels = [0 if record["defect"] == "good" else 1 for record in records]
    scores = [record["score"] for record in records]
    predicted = [int(score > threshold) for score in scores]
    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "report": classification_report(labels, predicted, target_names=["normal", "defective"], output_dict=True),
    }
    ious = [record["iou"] for record in records if record["iou"] is not None]
    localization = {"images": len(ious), "mean_iou": float(np.mean(ious)), "policy": "cells with distance >= 50% of image max vs NEAREST-resized mask"}

    torch.save({"bank": bank, "threshold": threshold, "backbone": "resnet18-layer3-imagenet", "image_score": "max_patch_nearest_neighbor_distance"}, project / "artifacts" / "final_detector.pt")
    model_card = {
        "selected_model": "patchcore_style_memory_bank",
        "method": "ResNet18 layer3 feature map (16x16 positions, 256-dim, L2-normalized); each patch scored by cosine distance to its nearest neighbor in a bank of all reference-normal patches",
        "image_score": "maximum patch distance",
        "threshold_policy": "95th percentile of max-patch scores on 40 held-out validation normal images; test set never used for calibration",
        "reference_images": len(reference_files),
        "bank_size": int(bank.shape[0]),
        "test_metrics": metrics,
        "localization": localization,
        "rejected_baselines": [
            {"model": "convolutional_autoencoder", "image_level_roc_auc": 0.986, "localization_mean_iou": 0.055, "reason": "strong image-level detection but poor localization; original threshold calibrated on test normals (leakage)"},
            {"model": "mean_reference_grid_patches", "image_level_roc_auc": 0.501, "reason": "rotation variance defeats a single averaged reference"},
            {"model": "memory_bank_pooled_grid_patches", "image_level_roc_auc": 0.744, "reason": "4x4 grid with pooled final-layer features too coarse"},
        ],
    }
    (project / "artifacts" / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "localization": localization}, indent=2))


if __name__ == "__main__":
    main()
