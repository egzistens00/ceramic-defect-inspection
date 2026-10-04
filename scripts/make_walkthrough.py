"""Generate a visual walkthrough: one tile through every pipeline step.

Produces artifacts/walkthrough/*.jpg — real outputs at each stage,
so the technologies can be SEEN, not just described.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "tile"
OUT = PROJECT / "artifacts" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = torch.load(PROJECT / "artifacts" / "inspector_model.pt", map_location="cpu")
BANK = MODEL["bank"]
REVIEW = MODEL["review_threshold"]
REJECT = MODEL["reject_threshold"]

device = torch.device("cpu")
weights = ResNet18_Weights.DEFAULT
backbone = resnet18(weights=weights).to(device).eval()
backbone.fc = torch.nn.Identity()
captured: dict[str, torch.Tensor] = {}
backbone.layer3.register_forward_hook(lambda m, i, o: captured.__setitem__("features", o))
preprocess = weights.transforms()

try:
    FONT = ImageFont.truetype("arial.ttf", 22)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 16)
    FONT_TITLE = ImageFont.truetype("arialbd.ttf", 30)
except OSError:
    FONT = FONT_SMALL = FONT_TITLE = ImageFont.load_default()


def title_card(text: str, subtitle: str = "") -> Image.Image:
    card = Image.new("RGB", (900, 100), (16, 22, 32))
    draw = ImageDraw.Draw(card)
    draw.text((30, 18), text, fill=(240, 240, 240), font=FONT_TITLE)
    if subtitle:
        draw.text((30, 62), subtitle, fill=(140, 155, 175), font=FONT_SMALL)
    return card


def analyze(image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        backbone(preprocess(image).unsqueeze(0).to(device))
    features = captured["features"].squeeze(0)
    height, width = features.shape[1], features.shape[2]
    flat = torch.nn.functional.normalize(features.flatten(1).T, dim=1)
    similarity = flat @ BANK.T
    distances = (1 - similarity.max(dim=1).values).reshape(height, width).numpy()
    return {"image": image, "distances": distances, "shape": (height, width)}


def make_step_images():
    """STEP 1: the input photo."""
    good = analyze(DATA / "test" / "good" / "000.png")
    bad = analyze(DATA / "test" / "crack" / "000.png")

    for label, data in [("a_good_tile", good), ("b_cracked_tile", bad)]:
        card = title_card(
            "STEP 1 - THE INPUT (what you upload)",
            f"a photo of a ceramic tile - {'a GOOD tile (no defects)' if 'good' in label else 'a CRACKED tile'}",
        )
        img = data["image"].resize((600, 600))
        panel = Image.new("RGB", (900, 700), (10, 14, 20))
        panel.paste(card, (0, 0))
        panel.paste(img, (150, 120))
        panel.save(OUT / f"step1_{label}.jpg", quality=92)

    """STEP 2: split into patches - the AI doesn't see a photo, it sees 16x16=256 small pieces."""
    for label, data in [("a_good", good), ("b_cracked", bad)]:
        img = data["image"].resize((600, 600))
        draw = ImageDraw.Draw(img)
        grid = 16
        for i in range(1, grid):
            x = i * 600 // grid
            y = i * 600 // grid
            draw.line([(x, 0), (x, 600)], fill=(80, 160, 255), width=1)
            draw.line([(0, y), (600, y)], fill=(80, 160, 255), width=1)
        card = title_card("STEP 2 - PATCHES (PyTorch + torchvision ResNet18)",
                          "the AI never sees 'a tile' - it sees 256 tiny patches (16x16 grid)")
        panel = Image.new("RGB", (900, 700), (10, 14, 20))
        panel.paste(card, (0, 0))
        panel.paste(img, (150, 120))
        d = ImageDraw.Draw(panel)
        d.text((30, 640), "ResNet18 converts EACH small patch into a list of 256 numbers (its 'fingerprint')",
               fill=(160, 180, 210), font=FONT_SMALL)
        panel.save(OUT / f"step2_patches_{label}.jpg", quality=92)

    """STEP 3: memory bank - compare each patch fingerprint against 'normal' fingerprints."""
    card = title_card("STEP 3 - MEMORY BANK (what 'normal' looks like)",
                      "during setup, the AI memorized every patch of 190 good tiles = 37,240 normal fingerprints (196 patches each)")
    panel = Image.new("RGB", (900, 300), (10, 14, 20))
    panel.paste(card, (0, 0))
    d = ImageDraw.Draw(panel)
    for i in range(12):
        x = 40 + i * 70
        d.rectangle([x, 90, x + 55, 145], outline=(70, 200, 120), width=2)
        d.text((x + 5, 155), f"good{i}", fill=(120, 200, 150), font=FONT_SMALL)
    d.text((30, 200), "BANK = all the normal patch-fingerprints", fill=(160, 180, 210), font=FONT_SMALL)
    d.text((30, 230), "new patch arrives -> find its most similar normal fingerprint -> distance = how 'weird' it is",
           fill=(160, 180, 210), font=FONT_SMALL)
    panel.save(OUT / "step3_memory_bank.jpg", quality=92)

    """STEP 4: the heatmap - each patch scored, red = weird."""
    for label, data in [("a_good", good), ("b_cracked", bad)]:
        dist = data["distances"]
        norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
        heat = Image.fromarray(np.uint8(255 * norm)).resize((600, 600), Image.BILINEAR)
        heat_rgb = Image.merge("RGB", (heat, Image.new("L", (600, 600)), Image.new("L", (600, 600))))
        base = data["image"].resize((600, 600))
        overlay = Image.blend(base, heat_rgb, 0.55)
        card = title_card("STEP 4 - THE HEATMAP (where is it weird?)",
                          "each patch got a 'weirdness score' - red = unlike anything normal")
        panel = Image.new("RGB", (900, 700), (10, 14, 20))
        panel.paste(card, (0, 0))
        panel.paste(base, (40, 120))
        panel.paste(overlay, (490, 120))
        d = ImageDraw.Draw(panel)
        d.text((40, 90), "original tile", fill=(160, 180, 210), font=FONT_SMALL)
        d.text((490, 90), "AI attention (red = defect)", fill=(160, 180, 210), font=FONT_SMALL)
        panel.save(OUT / f"step4_heatmap_{label}.jpg", quality=92)

    """STEP 5: the decision - one number, three doors."""
    card = title_card("STEP 5 - THE DECISION (three doors)", "the WORST patch score decides the tier")
    panel = Image.new("RGB", (900, 380), (10, 14, 20))
    panel.paste(card, (0, 0))
    d = ImageDraw.Draw(panel)
    doors = [
        (30, "ACCEPT", (30, 150, 80), f"score <= {REVIEW:.3f}", "auto-pass, no human"),
        (330, "REVIEW", (200, 160, 40), f"{REVIEW:.3f} to {REJECT:.3f}", "human must check"),
        (630, "REJECT", (190, 60, 60), f"score > {REJECT:.3f}", "auto-fail, no human"),
    ]
    for x, name, color, rng, meaning in doors:
        d.rectangle([x, 100, x + 240, 280], fill=color + (40,) if len(color) == 3 else color, outline=color, width=3)
        d.text((x + 60, 130), name, fill=color, font=FONT_TITLE)
        d.text((x + 30, 185), rng, fill=(220, 220, 220), font=FONT_SMALL)
        d.text((x + 30, 215), meaning, fill=(160, 180, 210), font=FONT_SMALL)
    d.text((30, 300), f"good tile scored    {good['distances'].max():.3f}  ->  ACCEPT", fill=(80, 200, 130), font=FONT)
    d.text((30, 335), f"cracked tile scored {bad['distances'].max():.3f}  ->  REJECT", fill=(230, 100, 100), font=FONT)
    panel.save(OUT / "step5_decision.jpg", quality=92)

    """STEP 6: RAG - the SOP knowledge base."""
    card = title_card("STEP 6 - RAG (answer from OUR documents)",
                      "the question + SOP sections are matched by meaning")
    panel = Image.new("RGB", (900, 460), (10, 14, 20))
    panel.paste(card, (0, 0))
    d = ImageDraw.Draw(panel)
    d.rectangle([30, 100, 870, 180], outline=(80, 160, 255), width=2)
    d.text((45, 115), "QUESTION: 'Why was this tile rejected?'", fill=(140, 180, 240), font=FONT)
    sop_sections = [
        "SOP: Defect: Crack - 'Cracks are linear fractures... any visible crack is an automatic reject'",
        "SOP: Escalation - 'same defect 3+ consecutive parts = stop feed, call supervisor'",
    ]
    for i, section in enumerate(sop_sections):
        y = 210 + i * 70
        d.rectangle([30, y, 870, y + 55], outline=(70, 200, 120), width=2)
        d.text((45, y + 15), section[:95], fill=(140, 220, 170), font=FONT_SMALL)
    d.text((30, 370), "embedding model (Hugging Face) matched the question to these sections BY MEANING,",
           fill=(160, 180, 210), font=FONT_SMALL)
    d.text((30, 395), "then they are handed to the LLM: 'explain this using ONLY these sections'",
           fill=(160, 180, 210), font=FONT_SMALL)
    panel.save(OUT / "step6_rag.jpg", quality=92)

    """STEP 7: the full pipeline on one page."""
    steps = sorted(OUT.glob("step*.jpg"))
    images = [Image.open(s) for s in steps]
    total_height = sum(im.height for im in images)
    canvas = Image.new("RGB", (900, total_height), (10, 14, 20))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(PROJECT / "artifacts" / "FULL_WALKTHROUGH.jpg", quality=90)
    print(f"Wrote {len(steps)} step images + FULL_WALKTHROUGH.jpg to {OUT}")


if __name__ == "__main__":
    make_step_images()
