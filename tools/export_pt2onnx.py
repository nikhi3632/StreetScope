#!/usr/bin/env python3
"""Export YOLOv8n .pt weights to ONNX format.

Requires: pip install ultralytics

Usage:
    python tools/export_onnx.py
    python tools/export_onnx.py --input models/yolov8n.pt --output models/yolov8n.onnx
"""

import argparse
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

    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=640, simplify=True)

    # ultralytics writes next to the .pt file by default
    default_onnx = pt.with_suffix(".onnx")
    out = Path(output_path)

    if default_onnx != out:
        default_onnx.rename(out)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done: {out} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX")
    parser.add_argument("--input", default="models/yolov8n.pt", help="Path to .pt weights")
    parser.add_argument("--output", default="models/yolov8n.onnx", help="Output ONNX path")
    args = parser.parse_args()
    export(args.input, args.output)


if __name__ == "__main__":
    main()
