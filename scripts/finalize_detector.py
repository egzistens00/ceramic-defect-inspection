"""Build the final memory-bank patch detector with honest calibration."""

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
VALIDATION_COUNT = 40
TOP_K_PATCHES = 4


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = weights.transforms()

    def patch_features(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        crops = [transform(image.crop((col * width // GRID, row * height // GRID, (col + 1) * width // GRID, (row + 1) * height // GRID))) for row in range(GRID) for col in range(GRID)]
        with torch.no_grad():
            features = model(torch.stack(crops).to(device)).cpu()
        return torch.nn.functional.normalize(features, dim=1)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    shuffled = normal_files[:]
    random.Random(SEED).shuffle(shuffled)
    validation_files, reference_files = shuffled[:VALIDATION_COUNT], shuffled[VALIDATION_COUNT:]

    bank = torch.cat([patch_features(Image.open(path).convert("RGB")) for path in reference_files])

    def patch_distances(image: Image.Image) -> np.ndarray:
        features = patch_features(image)
        similarity = features @ bank.T
        return (1 - similarity.max(dim=1).values).numpy()

    validation_scores = [float(np.sort(patch_distances(Image.open(path).convert("RGB")))[-TOP_K_PATCHES:].mean()) for path in validation_files]
    threshold = float(np.percentile(validation_scores, 95))

    records = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        for image_path in sorted(defect_dir.glob("*.png")):
            image = Image.open(image_path).convert("RGB")
            distances = patch_distances(image)
            score = float(np.sort(distances)[-TOP_K_PATCHES:].mean())
            heat = distances.reshape(GRID, GRID)
            predicted_cells = heat >= np.percentile(heat, 75)
            iou = None
            if defect_dir.name != "good":
                mask = np.asarray(Image.open(project / "data/tile/ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png").convert("L").resize((GRID, GRID), Image.NEAREST)) > 0
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
    localization = {"images": len(ious), "mean_iou": float(np.mean(ious))}

    torch.save({"bank": bank, "threshold": threshold, "grid": GRID, "backbone": "resnet18-imagenet", "image_score": "mean_top4_nearest_neighbor_patch_distance"}, project / "artifacts" / "patch_detector.pt")
    model_card = {
        "selected_model": "memory_bank_patch_detector",
        "method": "ResNet18 (ImageNet) features on a 4x4 grid; each patch scored by distance to its nearest neighbor in a bank of all reference normal patches",
        "image_score": "mean of the 4 highest patch distances",
        "threshold_policy": "95th percentile of scores on 40 held-out validation normal images (no test data used)",
        "reference_images": len(reference_files),
        "validation_images": len(validation_files),
        "test_metrics": metrics,
        "localization": localization,
        "rejected_baselines": [
            {
                "model": "convolutional_autoencoder",
                "image_level_roc_auc": 0.986,
                "localization_mean_iou": 0.055,
                "reason": "poor defect localization; threshold was also calibrated on test normals (leakage)",
            },
            {
                "model": "mean_reference_patch_features",
                "image_level_roc_auc": 0.501,
                "localization_mean_iou": 0.017,
                "reason": "screcks are rotated; comparing fixed grid patches to a single averaged reference cannot discriminate normal from defective",
            },
        ],
    }
    (project / "artifacts" / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "localization": localization}, indent=2))


if __name__ == "__main__":
    main()
