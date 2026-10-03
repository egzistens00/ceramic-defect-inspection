"""Evaluate grid-patch feature distances for MVTec tile localization."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = weights.transforms()
    grid = 4

    def embeddings(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        crops = []
        for row in range(grid):
            for col in range(grid):
                box = (col * width // grid, row * height // grid, (col + 1) * width // grid, (row + 1) * height // grid)
                crops.append(transform(image.crop(box)))
        with torch.no_grad():
            result = model(torch.stack(crops).to(device)).cpu()
        return torch.nn.functional.normalize(result, dim=1)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    normal_features = torch.stack([embeddings(Image.open(path).convert("RGB")) for path in normal_files]).mean(dim=0)
    rows = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        if defect_dir.name == "good":
            continue
        for image_path in sorted(defect_dir.glob("*.png")):
            image = Image.open(image_path).convert("RGB")
            distances = 1 - (embeddings(image) * normal_features).sum(dim=1).numpy()
            heat = distances.reshape(grid, grid)
            mask = np.asarray(Image.open(project / "data/tile/ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png").convert("L").resize((grid, grid))) > 0
            predicted = heat >= np.percentile(heat, 75)
            intersection = np.logical_and(predicted, mask).sum()
            union = np.logical_or(predicted, mask).sum()
            rows.append({"defect": defect_dir.name, "iou": float(intersection / union if union else 1.0), "precision": float(intersection / predicted.sum() if predicted.sum() else 0), "recall": float(intersection / mask.sum() if mask.sum() else 0)})
    summary = {"images": len(rows), "mean_iou": float(np.mean([row["iou"] for row in rows])), "mean_precision": float(np.mean([row["precision"] for row in rows])), "mean_recall": float(np.mean([row["recall"] for row in rows]))}
    output = project / "artifacts" / "patch_localization.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
