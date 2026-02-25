#!/usr/bin/env python3
"""Download model weights (YOLO11s + Real-ESRGAN).

Downloads PyTorch weights from GitHub releases.
To export to Core ML, run:
    python tools/export_pt2coreml.py yolo
    python tools/export_pt2coreml.py realesrgan --variant x2plus

Usage:
    python tools/fetch_models.py
    python tools/fetch_models.py --output-dir models/
"""

import argparse
import urllib.request
from pathlib import Path

MODELS = [
    {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
        "filename": "yolo11s.pt",
        "description": "YOLO11s detection",
    },
    {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "description": "Real-ESRGAN x2 super-resolution (23 RRDB blocks)",
    },
]


def download(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=report)
    print()

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--output-dir", default="models/", help="Directory to save models (default: models/)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        dest = output_dir / model["filename"]
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"Found {dest} ({size_mb:.1f} MB) — {model['description']}, skipping")
        else:
            print(f"\n{model['description']}:")
            download(model["url"], dest)

    print()
    print("Next steps (export to Core ML):")
    if not (output_dir / "yolo11s.mlpackage").exists():
        print("  python tools/export_pt2coreml.py yolo")
    if not (output_dir / "realesrgan_x2plus.mlpackage").exists():
        print("  python tools/export_pt2coreml.py realesrgan")


if __name__ == "__main__":
    main()
