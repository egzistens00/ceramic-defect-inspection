"""Evaluate reconstruction error and save anomaly heatmaps for MVTec tiles."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_autoencoder import Autoencoder


class ImageRecords(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index]).convert("RGB")), self.labels[index], str(self.files[index])


def collect(root: Path):
    files, labels = [], []
    for folder in sorted((root / "test").iterdir()):
        if folder.is_dir():
            for path in sorted(folder.glob("*.png")):
                files.append(path)
                labels.append(0 if folder.name == "good" else 1)
    return files, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/tile_autoencoder.pt")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(project / args.model, map_location=device)
    size = checkpoint["image_size"]
    model = Autoencoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    files, labels = collect(project / "data" / "tile")
    loader = DataLoader(ImageRecords(files, labels, transform), batch_size=16, shuffle=False)
    scores, actual, records = [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_files in loader:
            reconstruction = model(images.to(device)).cpu()
            errors = ((reconstruction - images) ** 2).mean(dim=(1, 2, 3)).numpy()
            scores.extend(errors.tolist())
            actual.extend(batch_labels.tolist())
            records.extend(batch_files)
    normal_scores = [score for score, label in zip(scores, actual) if label == 0]
    threshold = float(np.percentile(normal_scores, 95))
    predicted = [int(score > threshold) for score in scores]
    metrics = {"threshold": threshold, "roc_auc": roc_auc_score(actual, scores), "confusion_matrix": confusion_matrix(actual, predicted).tolist(), "report": classification_report(actual, predicted, target_names=["normal", "defective"], output_dict=True)}
    output = project / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
