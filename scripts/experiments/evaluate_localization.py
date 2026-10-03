"""Evaluate anomaly heatmap overlap with MVTec ground-truth masks."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_autoencoder import Autoencoder


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(project / "artifacts" / "tile_autoencoder.pt", map_location=device)
    size = checkpoint["image_size"]
    model = Autoencoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    rows = []
    with torch.no_grad():
        for defect_dir in sorted((project / "data/tile/test").iterdir()):
            if defect_dir.name == "good":
                continue
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = project / "data/tile/ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                reconstructed = model(image)
                error = (reconstructed - image).pow(2).mean(dim=1).squeeze(0).cpu().numpy()
                mask = np.asarray(Image.open(mask_path).convert("L").resize((size, size))) > 0
                threshold = np.percentile(error, 97)
                predicted = error >= threshold
                intersection = np.logical_and(predicted, mask).sum()
                union = np.logical_or(predicted, mask).sum()
                rows.append({"defect": defect_dir.name, "image": image_path.name, "iou": float(intersection / union if union else 1.0), "precision": float(intersection / predicted.sum() if predicted.sum() else 0.0), "recall": float(intersection / mask.sum() if mask.sum() else 0.0)})
    summary = {"images": len(rows), "mean_iou": float(np.mean([row["iou"] for row in rows])), "mean_precision": float(np.mean([row["precision"] for row in rows])), "mean_recall": float(np.mean([row["recall"] for row in rows])), "by_defect": {}}
    for defect in sorted({row["defect"] for row in rows}):
        subset = [row for row in rows if row["defect"] == defect]
        summary["by_defect"][defect] = {"images": len(subset), "mean_iou": float(np.mean([row["iou"] for row in subset])), "mean_precision": float(np.mean([row["precision"] for row in subset])), "mean_recall": float(np.mean([row["recall"] for row in subset]))}
    output = project / "artifacts" / "localization.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
