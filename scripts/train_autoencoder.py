"""Train a small convolutional autoencoder on normal MVTec tile images."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolder(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.files = sorted(root.glob("*.png"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        return self.transform(Image.open(self.files[index]).convert("RGB")), str(self.files[index])


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    transform = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    dataset = ImageFolder(root / "data" / "tile" / "train" / "good", transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, _ in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), images)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        epoch_loss = total / len(dataset)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")
    output = root / "artifacts"
    output.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": args.size}, output / "tile_autoencoder.pt")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved model to {output / 'tile_autoencoder.pt'} on {device}")


if __name__ == "__main__":
    main()
