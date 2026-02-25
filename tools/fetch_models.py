#!/usr/bin/env python3
"""Download YOLO11s PyTorch weights.

Downloads yolo11s.pt from Ultralytics GitHub releases.
To export to Core ML, run: python tools/export_pt2coreml.py

Usage:
    python tools/fetch_models.py
    python tools/fetch_models.py --output-dir models/
"""

import argparse
import urllib.request
from pathlib import Path

PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"
PT_FILENAME = "yolo11s.pt"


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
    parser = argparse.ArgumentParser(description="Download YOLO11s weights")
    parser.add_argument(
        "--output-dir", default="models/", help="Directory to save models (default: models/)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / PT_FILENAME

    if pt_path.exists():
        size_mb = pt_path.stat().st_size / (1024 * 1024)
        print(f"Found {pt_path} ({size_mb:.1f} MB), skipping download")
    else:
        download(PT_URL, pt_path)

    mlpackage_path = output_dir / "yolo11s.mlpackage"
    if not mlpackage_path.exists():
        print()
        print("Next step: export to Core ML")
        print("  python tools/export_pt2coreml.py")


if __name__ == "__main__":
    main()
