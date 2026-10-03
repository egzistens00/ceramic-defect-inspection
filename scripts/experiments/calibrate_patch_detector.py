"""Calibrate the patch detector honestly and produce the final model artifact."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

GRID = 4
SEED = 42
VALIDATION_FRACTION = 0.125


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = weights.transforms()

    def embed(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        crops = []
        for row in range(GRID):
            for col in range(GRID):
                box = (col * width // GRID, row * height // GRID, (col + 1) * width // GRID, (row + 1) * height // GRID)
                crops.append(transform(image.crop(box)))
        with torch.no_grad():
            features = model(torch.stack(crops).to(device)).cpu()
        return torch.nn.functional.normalize(features, dim=1)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    shuffled = normal_files[:]
    random.Random(SEED).shuffle(shuffled)
    validation_count = round(len(shuffled) * VALIDATION_FRACTION)
    validation_files, reference_files = shuffled[:validation_count], shuffled[validation_count:]

    reference = torch.stack([embed(Image.open(path).convert("RGB")) for path in reference_files]).mean(dim=0)

    def patch_distances(image: Image.Image) -> np.ndarray:
        return 1 - (embed(image) * reference).sum(dim=1).numpy()

    validation_scores = [float(np.sort(patch_distances(Image.open(path).convert("RGB")))[-4:].mean()) for path in validation_files]
    threshold = float(np.percentile(validation_scores, 95))

    records = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        for image_path in sorted(defect_dir.glob("*.png")):
            image = Image.open(image_path).convert("RGB")
            scores = patch_distances(image)
            score = float(np.sort(scores)[-4:].mean())
            heat = scores.reshape(GRID, GRID)
            predicted_cells = heat >= np.percentile(heat, 75)
            iou = None
            if defect_dir.name != "good":
                mask_path = project / "data/tile/ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                mask = np.asarray(Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)) > 0
                intersection = np.logical_and(predicted_cells, mask).sum()
                union = np.logical_or(predicted_cells, mask).sum()
                iou = float(intersection / union) if union else 1.0
            records.append({"defect": defect_dir.name, "score": score, "iou": iou})

    labels = [0 if r["defect"] == "good" else 1 for r in records]
    scores = [r["score"] for r in records]
    predicted = [int(s > threshold) for s in scores]
    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "report": classification_report(labels, predicted, target_names=["normal", "defective"], output_dict=True),
    }
    ious = [r["iou"] for r in records if r["iou"] is not None]
    localization = {"images": len(ious), "mean_iou": float(np.mean(ious))}

    torch.save({"reference": reference, "threshold": threshold, "grid": GRID, "backbone": "resnet18-imagenet", "image_score": "mean_top4_patch_distance"}, project / "artifacts" / "patch_detector.pt")
    model_card = {
        "selected_model": "patch_feature_detector",
        "method": "ResNet18 (ImageNet) features on a 4x4 image grid; cosine distance to a reference built from training normals only",
        "image_score": "mean of the 4 highest patch distances",
        "threshold": threshold,
        "threshold_policy": "95th percentile of scores on 40 held-out validation normal images",
        "reference_images": len(reference_files),
        "validation_images": len(validation_files),
        "test_metrics": metrics,
        "localization": localization,
        "rejected_baseline": {
            "model": "convolutional_autoencoder",
            "image_level_roc_auc": 0.986,
            "localization_mean_iou": 0.0548,
            "reason": "poor defect localization despite good image-level detection",
            "note": "autoencoder threshold was calibrated on test normals (leakage); superseded by this calibrated evaluation",
        },
    }
    (project / "artifacts" / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "localization": localization}, indent=2))


if __name__ == "__main__":
    main()
