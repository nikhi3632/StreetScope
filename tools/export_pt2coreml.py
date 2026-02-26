#!/usr/bin/env python3
"""Export PyTorch models to Core ML (.mlpackage) format.

A monkey-patch fixes a coremltools _cast bug when using torch >= 2.10.

Usage:
    python tools/export_pt2coreml.py [--input PATH] [--output PATH]
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# coremltools patch (torch 2.10+ compatibility)
# ---------------------------------------------------------------------------


def patch_coremltools_cast():
    """Fix coremltools _cast for torch 2.10+ (numpy array instead of scalar).

    Torch 2.10 traces with multi-dimensional arrays where coremltools expects
    0-d scalars. The fix: call .item() before casting.
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


def verify_coreml(out_path: Path) -> None:
    """Print model info after export."""
    import coremltools as ct

    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    mlmodel = ct.models.MLModel(str(out_path))
    spec = mlmodel.get_spec()
    inp = spec.description.input[0]

    if inp.type.HasField("imageType"):
        it = inp.type.imageType
        cs_name = {10: "GRAY", 20: "RGB", 30: "BGR"}.get(it.colorSpace, str(it.colorSpace))
        print(f"Input: {inp.name}, colorSpace={cs_name}, {it.width}x{it.height}")
    else:
        mt = inp.type.multiArrayType
        shape = [mt.shape[i] for i in range(len(mt.shape))]
        print(f"Input: {inp.name}, shape={shape}")

    out_desc = spec.description.output[0]
    if out_desc.type.HasField("multiArrayType"):
        mt = out_desc.type.multiArrayType
        shape = [mt.shape[i] for i in range(len(mt.shape))]
        print(f"Output: {out_desc.name}, shape={shape}")


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------


def export_yolo(input_path: str, output_path: str) -> None:
    """Export YOLO11 via Ultralytics (handles C2PSA attention blocks)."""
    try:
        import coremltools as ct  # noqa: F401
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    patch_coremltools_cast()

    pt = Path(input_path)
    if not pt.exists():
        print(f"Error: {pt} not found. Run: python tools/fetch_models.py")
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

    verify_coreml(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to Core ML")
    parser.add_argument("--input", default="models/yolo11s.pt", help="Path to .pt weights")
    parser.add_argument("--output", default="models/yolo11s.mlpackage", help="Output Core ML path")

    args = parser.parse_args()
    export_yolo(args.input, args.output)


if __name__ == "__main__":
    main()
