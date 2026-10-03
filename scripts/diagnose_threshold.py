"""Diagnose score distributions: validation vs test-good vs test-defective."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_autoencoder import Autoencoder

SEED = 42
VALIDATION_COUNT = 40
EPOCHS = 10
BATCH_SIZE = 16
IMAGE_SIZE = 128


class FileList(Dataset):
    def __init__(self, files, transform):
        self.files, self.transform = files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index]).convert("RGB"))


def percentiles(values):
    array = np.array(values)
    return {f"p{p}": float(np.percentile(array, p)) for p in (5, 25, 50, 75, 95, 100)}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    shuffled = normal_files[:]
    random.Random(SEED).shuffle(shuffled)
    validation_files, reference_files = shuffled[:VALIDATION_COUNT], shuffled[VALIDATION_COUNT:]

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    loader = DataLoader(FileList(reference_files, transform), batch_size=BATCH_SIZE, shuffle=True)
    autoencoder = Autoencoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        autoencoder.train()
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = loss_fn(autoencoder(images), images)
            loss.backward()
            optimizer.step()

    autoencoder.eval()

    def score(path: Path) -> float:
        with torch.no_grad():
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            return float(((autoencoder(image) - image) ** 2).mean().item())

    reference_scores = [score(path) for path in reference_files]
    validation_scores = [score(path) for path in validation_files]
    test_good = [score(path) for path in sorted((project / "data/tile/test/good").glob("*.png"))]
    test_defective = []
    for defect_dir in sorted((project / "data/tile/test").iterdir()):
        if defect_dir.name != "good":
            test_defective.extend(score(path) for path in sorted(defect_dir.glob("*.png")))

    print("reference (train) normals:", percentiles(reference_scores))
    print("validation normals:       ", percentiles(validation_scores))
    print("test good:                ", percentiles(test_good))
    print("test defective:           ", percentiles(test_defective))


if __name__ == "__main__":
    main()
