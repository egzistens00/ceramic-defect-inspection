"""Generate visual heatmaps from the patch-based anomaly detector."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights, resnet18

GRID = 4
EXAMPLES = [
    ("scratch_head", "000.png"),
    ("scratch_neck", "000.png"),
    ("thread_top", "000.png"),
]


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = weights.transforms()

    def patch_embeddings(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        crops = []
        for row in range(GRID):
            for col in range(GRID):
                box = (col * width // GRID, row * height // GRID, (col + 1) * width // GRID, (row + 1) * height // GRID)
                crops.append(transform(image.crop(box)))
        with torch.no_grad():
            result = model(torch.stack(crops).to(device)).cpu()
        return torch.nn.functional.normalize(result, dim=1)

    normal_files = sorted((project / "data/tile/train/good").glob("*.png"))
    normal_features = torch.stack([patch_embeddings(Image.open(path).convert("RGB")) for path in normal_files]).mean(dim=0)

    panels = []
    for defect, filename in EXAMPLES:
        image_path = project / "data/tile/test" / defect / filename
        mask_path = project / "data/tile/ground_truth" / defect / filename.replace(".png", "_mask.png")
        image = Image.open(image_path).convert("RGB").resize((256, 256))

        distances = 1 - (patch_embeddings(image) * normal_features).sum(dim=1).numpy()
        heat = distances.reshape(GRID, GRID)
        heat_map = Image.fromarray(np.uint8(255 * (heat - heat.min()) / (heat.max() - heat.min() + 1e-9))).resize((256, 256), Image.BILINEAR)
        heat_rgb = Image.merge("RGB", (heat_map, Image.new("L", (256, 256)), Image.new("L", (256, 256))))
        overlay = Image.blend(image, heat_rgb, 0.45)

        mask = Image.open(mask_path).convert("L").resize((256, 256))
        mask_rgb = Image.new("RGB", (256, 256), "white")
        mask_rgb.paste((0, 180, 0), mask=mask)

        row = Image.new("RGB", (768, 290), "white")
        row.paste(image, (0, 30))
        row.paste(overlay, (256, 30))
        row.paste(mask_rgb, (512, 30))
        draw = ImageDraw.Draw(row)
        for x, title in zip((0, 256, 512), (f"{defect} original", "patch anomaly heatmap", "ground truth mask")):
            draw.text((x + 5, 5), title, fill="black")
        panels.append(row)

    output = project / "artifacts" / "patch_heatmaps.jpg"
    canvas = Image.new("RGB", (768, 290 * len(panels)), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (0, index * 290))
    canvas.save(output, quality=92)
    print(f"Wrote {output} ({len(panels)} examples)")


if __name__ == "__main__":
    main()
