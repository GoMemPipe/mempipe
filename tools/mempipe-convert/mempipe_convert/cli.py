"""CLI interface for mempipe-convert."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mempipe-convert",
        description="Convert ML models to MemPipe .mpmodel format",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── onnx ──────────────────────────────────────────────────────────────
    p_onnx = sub.add_parser("onnx", help="Convert ONNX model to .mpmodel")
    p_onnx.add_argument("input", help="Input .onnx file path")
    p_onnx.add_argument("-o", "--output", required=True, help="Output .mpmodel file path")
    p_onnx.add_argument("--name", default=None, help="Model name (default: filename stem)")
    p_onnx.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default=None,
        help="Apply quantization",
    )
    p_onnx.add_argument("--platform", default="", help="Platform hints string")
    p_onnx.add_argument(
        "--transformer",
        action="store_true",
        default=False,
        help="Use transformer-specific conversion with pattern fusion (GELU, LayerNorm)",
    )
    p_onnx.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Fixed sequence length for dynamic dims (default: 128)",
    )

    # ── inspect ───────────────────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Inspect a .mpmodel file")
    p_inspect.add_argument("model", help="Path to .mpmodel file")

    # ── quantize ──────────────────────────────────────────────────────────
    p_quant = sub.add_parser("quantize", help="Post-training quantization")
    p_quant.add_argument("input", help="Input .mpmodel file path")
    p_quant.add_argument("-o", "--output", required=True, help="Output .mpmodel file path")
    p_quant.add_argument(
        "--method",
        choices=["dynamic", "static", "fp16"],
        default="dynamic",
        help="Quantization method (default: dynamic)",
    )

    # ── validate ──────────────────────────────────────────────────────────
    p_validate = sub.add_parser("validate", help="Validate .mpmodel against ONNX reference")
    p_validate.add_argument("model", help="Path to .mpmodel file")
    p_validate.add_argument("--reference", required=True, help="Path to reference .onnx file")
    p_validate.add_argument("--input", default=None, help="Path to test input .npy file")
    p_validate.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance (default: 1e-4)",
    )

    args = parser.parse_args(argv)

    if args.command == "onnx":
        _cmd_onnx(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "quantize":
        _cmd_quantize(args)
    elif args.command == "validate":
        _cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_onnx(args):
    if args.transformer:
        from mempipe_convert.transformer import from_onnx_transformer

        model = from_onnx_transformer(
            args.input,
            args.output,
            quantize=args.quantize,
            name=args.name,
            platform_hints=args.platform,
            fuse_patterns=True,
            seq_len=args.seq_len,
        )
    else:
        from mempipe_convert.converter import from_onnx

        model = from_onnx(
            args.input,
            args.output,
            quantize=args.quantize,
            name=args.name,
            platform_hints=args.platform,
        )
    print(f"Converted: {args.input} → {args.output}")
    print(f"  Ops: {len(model.graph)}, Tensors: {len(model.tensor_names)}")
    print(f"  Weights: {len(model.weights_blob):,} bytes")


def _cmd_inspect(args):
    from mempipe_convert.inspect import inspect_model

    info = inspect_model(args.model)
    print(info)


def _cmd_quantize(args):
    import numpy as np

    from mempipe_convert.quantize import quantize

    quantize(args.input, args.output, method=args.method)

    in_size = Path(args.input).stat().st_size
    out_size = Path(args.output).stat().st_size
    ratio = out_size / in_size if in_size > 0 else 0
    print(f"Quantized: {args.input} → {args.output}")
    print(f"  Method: {args.method}")
    print(f"  Size: {in_size:,} → {out_size:,} bytes ({ratio:.1%})")


def _cmd_validate(args):
    import numpy as np

    from mempipe_convert.validate import validate

    test_input = None
    if args.input:
        test_input = np.load(args.input)

    result = validate(args.model, args.reference, test_input, atol=args.atol)
    print(result)
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
