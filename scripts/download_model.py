"""Fetch the model artifact for deployment.

On Render (free tier) the repo cannot ship the 36 MB model, so this script
downloads it from the GitHub release attached to this repository. Locally it
is a no-op when the artifact already exists.

Environment variables:
  MODEL_RELEASE_URL - direct download URL of inspector_model.pt
"""

import os
import sys
import urllib.request
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "inspector_model.pt"


def main() -> None:
    if ARTIFACT.exists():
        print(f"model already present: {ARTIFACT}")
        return
    url = os.environ.get("MODEL_RELEASE_URL")
    if not url:
        print("MODEL_RELEASE_URL not set and no local model found")
        sys.exit(1)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading model from {url}")
    urllib.request.urlretrieve(url, ARTIFACT)
    print(f"saved: {ARTIFACT} ({ARTIFACT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
