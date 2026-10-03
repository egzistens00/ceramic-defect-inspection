"""Patch-feature anomaly baseline using a pretrained ResNet backbone."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights).to(device).eval()
    backbone.fc = torch.nn.Identity()
    transform = weights.transforms()
    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    test_files = sorted((project / "data/tile/test").glob("*/")[0].glob("*.png")) if False else []
    with torch.no_grad():
        features = []
        for path in normal_files:
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            features.append(backbone(image).cpu())
    reference = torch.cat(features).mean(dim=0)
    output = project / "artifacts" / "feature_reference.pt"
    torch.save({"reference": reference, "model": "resnet18-imagenet", "normal_count": len(normal_files)}, output)
    print(f"Saved feature reference from {len(normal_files)} normal images to {output}")


if __name__ == "__main__":
    main()
