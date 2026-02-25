#!/usr/bin/env python3
"""Export YOLO11 PyTorch weights to Core ML (.mlpackage) format.

Ultralytics handles the hard part (YOLO11 C2PSA attention block conversion).
A monkey-patch fixes a coremltools _cast bug when using torch >= 2.10.

The resulting model:
  - Input: Image (RGB, 640x640)
  - Internal: mul(÷255) normalization → convolutions
  - Output: (1, 84, 8400) raw detections

Requires: pip install ultralytics coremltools

Usage:
    python tools/export_pt2coreml.py
    python tools/export_pt2coreml.py --input models/yolo11s.pt --output models/yolo11s.mlpackage
"""

import argparse
import shutil
import sys
from pathlib import Path


def patch_coremltools_cast():
    """Fix coremltools _cast for torch 2.10+ (numpy array instead of scalar).

    Torch 2.10 traces C2PSA attention blocks with multi-dimensional arrays where
    coremltools expects 0-d scalars. The fix: call .item() before casting.
    """
    import coremltools.converters.mil.frontend.torch.ops as torch_ops
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs

    def cast_fixed(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            if not isinstance(x.val, dtype):
                val = x.val.item() if hasattr(x.val, "item") else x.val
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    torch_ops._cast = cast_fixed


def export(input_path: str, output_path: str) -> None:
    try:
        import coremltools as ct
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    patch_coremltools_cast()

    pt = Path(input_path)
    if not pt.exists():
        print(f"Error: {pt} not found")
        sys.exit(1)

    print(f"Loading {pt}")
    model = YOLO(str(pt))
    print("Exporting to Core ML (Ultralytics)...")
    model.export(format="coreml", imgsz=640, nms=False)

    default_mlpackage = pt.with_suffix(".mlpackage")
    out = Path(output_path)
    if default_mlpackage != out:
        if out.exists():
            shutil.rmtree(out)
        default_mlpackage.rename(out)

    # Verify
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"Saved: {out} ({size_mb:.1f} MB)")

    mlmodel = ct.models.MLModel(str(out))
    spec = mlmodel.get_spec()
    inp = spec.description.input[0]
    it = inp.type.imageType
    cs_name = {10: "GRAY", 20: "RGB", 30: "BGR"}.get(it.colorSpace, str(it.colorSpace))
    print(f"Input: {inp.name}, colorSpace={cs_name}, {it.width}x{it.height}")

    if spec.HasField("mlProgram"):
        for _, fn in spec.mlProgram.functions.items():
            for _, block in fn.block_specializations.items():
                first_types = [op.type for op in list(block.operations)[:5]]
                print(f"First 5 ops: {first_types}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11 to Core ML")
    parser.add_argument("--input", default="models/yolo11s.pt", help="Path to .pt weights")
    parser.add_argument("--output", default="models/yolo11s.mlpackage", help="Output Core ML path")
    args = parser.parse_args()
    export(args.input, args.output)


if __name__ == "__main__":
    main()
