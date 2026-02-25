#!/usr/bin/env python3
"""Export PyTorch models to Core ML (.mlpackage) format.

Supports:
  yolo      — YOLO11s object detection (via Ultralytics export)
  realesrgan — Real-ESRGAN super-resolution (via torch.jit.trace)

A monkey-patch fixes a coremltools _cast bug when using torch >= 2.10.

Usage:
    python tools/export_pt2coreml.py yolo [--input PATH] [--output PATH]
    python tools/export_pt2coreml.py realesrgan [--input PATH] [--output PATH]
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

    # Image input (YOLO) vs tensor input (Real-ESRGAN)
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
# Real-ESRGAN export
# ---------------------------------------------------------------------------

REALESRGAN_CONFIG = {
    "num_block": 23,
    "scale": 2,
    "num_feat": 64,
    "num_grow_ch": 32,
    "input_h": 288,
    "input_w": 512,
}


def import_rrdbnet():
    """Import RRDBNet, bypassing basicsr's broken __init__.

    basicsr's package init eagerly loads data modules that import a removed
    torchvision.transforms.functional_tensor (dropped in torchvision 0.20+).
    We only need the architecture class, so we stub the basicsr package to
    prevent __init__.py from running its eager imports.
    """
    import importlib.util
    import types

    if "basicsr" not in sys.modules:
        spec = importlib.util.find_spec("basicsr")
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError("basicsr not installed. Run: pip install basicsr")
        stub = types.ModuleType("basicsr")
        stub.__path__ = list(spec.submodule_search_locations)
        sys.modules["basicsr"] = stub

    from basicsr.archs.rrdbnet_arch import RRDBNet

    return RRDBNet


def export_realesrgan(input_path: str, output_path: str) -> None:
    """Export Real-ESRGAN x2plus via torch.jit.trace + coremltools."""
    try:
        import coremltools as ct
        import numpy as np
        import torch

        RRDBNet = import_rrdbnet()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    patch_coremltools_cast()

    cfg = REALESRGAN_CONFIG
    pt_path = Path(input_path)
    out_path = Path(output_path)

    if not pt_path.exists():
        print(f"Error: {pt_path} not found. Run: python tools/fetch_models.py")
        sys.exit(1)

    # Build model
    print(f"Loading RRDBNet ({cfg['num_block']} RRDB blocks, {cfg['scale']}x upscale)")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=cfg["num_feat"],
        num_block=cfg["num_block"],
        num_grow_ch=cfg["num_grow_ch"],
        scale=cfg["scale"],
    )

    # Load weights (Real-ESRGAN wraps in 'params_ema' key)
    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Trace
    h, w = cfg["input_h"], cfg["input_w"]
    scale = cfg["scale"]
    example_input = torch.rand(1, 3, h, w)
    print(f"Tracing with input shape (1, 3, {h}, {w})...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Verify PyTorch output shape
    with torch.no_grad():
        test_out = traced(example_input)
    expected = (1, 3, h * scale, w * scale)
    assert test_out.shape == expected, f"Expected {expected}, got {test_out.shape}"
    print(f"PyTorch output: {test_out.shape}")

    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, h, w))],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))

    verify_coreml(out_path)

    # Quick accuracy check
    print("Verifying...")
    loaded = ct.models.MLModel(str(out_path))
    test_input = np.random.rand(1, 3, h, w).astype(np.float32)
    coreml_out = loaded.predict({"input": test_input})
    out_name = loaded.get_spec().description.output[0].name
    coreml_arr = coreml_out[out_name]

    with torch.no_grad():
        pt_out = traced(torch.from_numpy(test_input)).numpy()
    diff = float(np.abs(pt_out - coreml_arr).mean())
    print(f"Mean abs diff (PyTorch vs Core ML): {diff:.6f}")
    print(f"Output range: [{coreml_arr.min():.3f}, {coreml_arr.max():.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to Core ML")
    subparsers = parser.add_subparsers(dest="model", required=True, help="Model to export")

    # YOLO
    yolo_parser = subparsers.add_parser("yolo", help="Export YOLO11s detection model")
    yolo_parser.add_argument("--input", default="models/yolo11s.pt", help="Path to .pt weights")
    yolo_parser.add_argument(
        "--output", default="models/yolo11s.mlpackage", help="Output Core ML path"
    )

    # Real-ESRGAN
    esrgan_parser = subparsers.add_parser(
        "realesrgan", help="Export Real-ESRGAN x2plus super-resolution"
    )
    esrgan_parser.add_argument(
        "--input", default="models/RealESRGAN_x2plus.pth", help="Path to .pth weights"
    )
    esrgan_parser.add_argument(
        "--output", default="models/realesrgan_x2plus.mlpackage", help="Output Core ML path"
    )

    args = parser.parse_args()

    if args.model == "yolo":
        export_yolo(args.input, args.output)
    elif args.model == "realesrgan":
        export_realesrgan(args.input, args.output)


if __name__ == "__main__":
    main()
