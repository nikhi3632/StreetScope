#!/usr/bin/env python3
"""Export YOLO11 PyTorch weights to Core ML (.mlpackage) format.

Requires: pip install ultralytics coremltools

Usage:
    python tools/export_pt2coreml.py
    python tools/export_pt2coreml.py --input models/yolo11s.pt --output models/yolo11s.mlpackage
"""

import argparse
import shutil
import sys
from pathlib import Path


def export(input_path: str, output_path: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    pt = Path(input_path)
    if not pt.exists():
        print(f"Error: {pt} not found")
        sys.exit(1)

    print(f"Loading {pt}")
    model = YOLO(str(pt))

    print("Exporting to Core ML...")
    model.export(format="coreml", imgsz=640, nms=False)

    # ultralytics writes next to the .pt file by default
    default_mlpackage = pt.with_suffix(".mlpackage")
    out = Path(output_path)

    if default_mlpackage != out:
        if out.exists():
            shutil.rmtree(out)
        default_mlpackage.rename(out)

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"Done: {out} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11s to Core ML")
    parser.add_argument("--input", default="models/yolo11s.pt", help="Path to .pt weights")
    parser.add_argument(
        "--output", default="models/yolo11s.mlpackage", help="Output Core ML path"
    )
    args = parser.parse_args()
    export(args.input, args.output)


if __name__ == "__main__":
    main()
