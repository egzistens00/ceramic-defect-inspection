"""Generate reconstruction-error heatmaps for representative tile images."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from train_autoencoder import Autoencoder


def colorize(values: np.ndarray) -> Image.Image:
    values = np.clip(values * 5.0, 0, 1)
    red = (values * 255).astype(np.uint8)
    return Image.fromarray(np.stack([red, np.zeros_like(red), np.zeros_like(red)], axis=-1))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(project / "artifacts" / "tile_autoencoder.pt", map_location=device)
    size = checkpoint["image_size"]
    model = Autoencoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    selections = [
        ("normal", project / "data/tile/test/good/000.png", None),
        ("scratch_head", project / "data/tile/test/scratch_head/000.png", project / "data/tile/ground_truth/scratch_head/000_mask.png"),
        ("thread_top", project / "data/tile/test/thread_top/000.png", project / "data/tile/ground_truth/thread_top/000_mask.png"),
    ]
    panels = []
    with torch.no_grad():
        for label, image_path, mask_path in selections:
            original = Image.open(image_path).convert("RGB")
            tensor = transform(original).unsqueeze(0).to(device)
            reconstruction = model(tensor).squeeze(0).cpu()
            error = (reconstruction - tensor.cpu().squeeze(0)).pow(2).mean(dim=0).numpy()
            original = original.resize((256, 256))
            rebuilt = Image.fromarray((reconstruction.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((256, 256))
            heatmap = Image.blend(rebuilt, colorize(error).resize((256, 256)), 0.55)
            mask = Image.open(mask_path).convert("L").resize((256, 256)) if mask_path else Image.new("L", (256, 256))
            mask_rgb = Image.new("RGB", (256, 256), "white")
            mask_rgb.paste((0, 180, 0), mask=mask)
            row = Image.new("RGB", (1024, 290), "white")
            row.paste(original, (0, 30)); row.paste(rebuilt, (256, 30)); row.paste(heatmap, (512, 30)); row.paste(mask_rgb, (768, 30))
            draw = ImageDraw.Draw(row)
            for x, title in zip((0, 256, 512, 768), (f"{label} original", "reconstruction", "error heatmap", "ground truth mask")):
                draw.text((x + 5, 5), title, fill="black")
            panels.append(row)
    output = project / "artifacts" / "heatmaps.jpg"
    output.parent.mkdir(exist_ok=True)
    canvas = Image.new("RGB", (1024, 290 * len(panels)), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (0, index * 290))
    canvas.save(output, quality=92)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
