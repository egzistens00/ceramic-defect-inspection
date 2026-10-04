"""Build per-defect-type patch banks for defect identification.

Detection remains unsupervised (normal-only memory bank). Identification is a
supervised add-on: MVTec provides defect labels only in the test folder, so we
use a stratified split of the 84 defective images — banks built from one part,
accuracy measured on the held-out part. Detection metrics are unaffected.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

SEED = 42
TRAIN_FRACTION = 0.65
TOP_K = 10

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "tile"
ARTIFACTS = PROJECT / "artifacts"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
backbone = resnet18(weights=weights).to(device).eval()
backbone.fc = torch.nn.Identity()
captured: dict[str, torch.Tensor] = {}
backbone.layer3.register_forward_hook(lambda module, inputs, output: captured.__setitem__("features", output))
preprocess = weights.transforms()

model_data = torch.load(ARTIFACTS / "inspector_model.pt", map_location=device)
normal_bank = model_data["bank"].to(device)


def features(image: Image.Image):
    with torch.no_grad():
        backbone(preprocess(image).unsqueeze(0).to(device))
    feature_map = captured["features"].squeeze(0)
    height, width = feature_map.shape[1], feature_map.shape[2]
    flat = torch.nn.functional.normalize(feature_map.flatten(1).T, dim=1)
    return flat, (height, width)


def defect_dirs():
    return [path for path in sorted((DATA / "test").iterdir()) if path.is_dir() and path.name != "good"]


def main() -> None:
    randomizer = random.Random(SEED)
    split: dict[str, dict] = {}
    for folder in defect_dirs():
        files = sorted(folder.glob("*.png"))
        randomizer.shuffle(files)
        bank_count = round(len(files) * TRAIN_FRACTION)
        split[folder.name] = {"bank": files[:bank_count], "eval": files[bank_count:]}

    defect_banks: dict[str, torch.Tensor] = {}
    for defect_type, parts in split.items():
        chunks = []
        for path in parts["bank"]:
            flat, (height, width) = features(Image.open(path).convert("RGB"))
            mask_path = DATA / "ground_truth" / defect_type / f"{path.stem}_mask.png"
            mask = Image.open(mask_path).convert("L").resize((width, height), Image.BILINEAR)
            selection = torch.from_numpy((np.asarray(mask) / 255.0 > 0.3).flatten())
            if selection.sum() == 0:
                distances = 1 - (flat @ normal_bank.T).max(dim=1).values
                selection = torch.zeros_like(selection)
                selection[torch.topk(distances, 5).indices] = True
            chunks.append(flat[selection])
        defect_banks[defect_type] = torch.cat(chunks).cpu()
        print(f"bank {defect_type}: {defect_banks[defect_type].shape[0]} patches from {len(parts['bank'])} images")

    correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = {}
    for defect_type, parts in split.items():
        for path in parts["eval"]:
            flat, _ = features(Image.open(path).convert("RGB"))
            distances = 1 - (flat @ normal_bank.T).max(dim=1).values
            top = torch.topk(distances, TOP_K).indices
            scores = {}
            for name, bank in defect_banks.items():
                similarity = flat[top] @ bank.to(device).T
                scores[name] = float((1 - similarity.max(dim=1).values).mean())
            prediction = min(scores, key=scores.get)
            confusion.setdefault(defect_type, {}).setdefault(prediction, 0)
            confusion[defect_type][prediction] += 1
            correct += prediction == defect_type
            total += 1

    accuracy = correct / total if total else 0.0
    print(f"held-out identification accuracy: {accuracy:.3f} ({correct}/{total})")
    print(json.dumps(confusion, indent=2))

    model_data["defect_banks"] = defect_banks
    model_data["defect_types"] = list(defect_banks)
    torch.save(model_data, ARTIFACTS / "inspector_model.pt")

    card = {
        "component": "defect_type_identification",
        "method": "per-defect-type patch memory banks; anomalous patches (top-10 by normal-bank distance) classified by nearest-neighbor distance to each defect bank",
        "supervision": "supervised add-on; MVTec labels exist only in the test folder, so a stratified split (seed 42, 65/35) of the 84 defective images was used: banks from one part, accuracy on the held-out part",
        "held_out_accuracy": accuracy,
        "held_out_images": total,
        "confusion": confusion,
        "impact_on_detection_metrics": "none - the unsupervised detector (91.5% accuracy, 98.7% precision) is unchanged",
    }
    (ARTIFACTS / "defect_model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"saved defect banks to inspector_model.pt and card to defect_model_card.json")


if __name__ == "__main__":
    main()
