#!/usr/bin/env python3
"""Download GPT-2 from Hugging Face, export to ONNX, then convert to .mpmodel.

This script handles the full pipeline:
  1. Download GPT-2 weights from Hugging Face
  2. Export to ONNX with a fixed sequence length (no dynamic axes)
  3. Convert ONNX → .mpmodel using the transformer converter

Requirements:
    pip install transformers torch onnx

Usage:
    # Full pipeline (download + export + convert):
    python zoo/convert_gpt2.py --output gpt2.mpmodel --seq-len 128

    # Export ONNX only (then convert separately):
    python zoo/convert_gpt2.py --onnx-only --output gpt2.onnx --seq-len 128
    mempipe-convert onnx --transformer --seq-len 128 gpt2.onnx -o gpt2.mpmodel
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Ensure parent package is importable when run from zoo/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Download & convert GPT-2 to .mpmodel")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model name (default: gpt2)",
    )
    parser.add_argument(
        "--output",
        default="gpt2.mpmodel",
        help="Output path (default: gpt2.mpmodel)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Fixed sequence length (default: 128)",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default=None,
        help="Apply quantization",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export ONNX (skip .mpmodel conversion)",
    )
    args = parser.parse_args()

    print(f"Converting {args.model} → {args.output}")
    print(f"  Sequence length: {args.seq_len}")
    print()

    # ── Step 1: Export to ONNX ────────────────────────────────────────────
    print("Step 1: Exporting to ONNX...")
    onnx_path = _export_to_onnx(args.model, args.opset, args.seq_len)
    print(f"  ONNX saved to: {onnx_path}")
    print(f"  Size: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB")

    if args.onnx_only:
        # Convert external data to single-file ONNX for portability
        import onnx
        import shutil

        m = onnx.load(onnx_path)  # loads external data relative to onnx_path dir
        onnx.save(m, args.output)  # saves everything inline
        sz = Path(args.output).stat().st_size / 1024 / 1024
        print(f"\nONNX exported to: {args.output}  ({sz:.0f} MB, single-file)")
        print("Convert with:")
        print(
            f"  mempipe-convert onnx --transformer --seq-len {args.seq_len} "
            f"{args.output} -o {Path(args.output).stem}.mpmodel"
        )
        return

    # ── Step 2: Convert ONNX → .mpmodel ──────────────────────────────────
    print()
    print("Step 2: Converting ONNX → .mpmodel...")
    from mempipe_convert.transformer import from_onnx_transformer

    model = from_onnx_transformer(
        onnx_path,
        args.output,
        quantize=args.quantize,
        name=args.model,
        platform_hints="native",
        fuse_patterns=True,
        seq_len=args.seq_len,
    )

    print()
    print(f"Done! Written: {args.output}")
    print(f"  Ops:     {len(model.graph)}")
    print(f"  Tensors: {len(model.tensor_names)}")
    print(f"  Weights: {len(model.weights_blob) / 1024 / 1024:.1f} MB")


def _export_to_onnx(model_name: str, opset: int, seq_len: int) -> str:
    """Export a Hugging Face causal LM to ONNX with fixed sequence length."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch
    except ImportError:
        print("Error: transformers and torch are required.")
        print("  pip install transformers torch onnx")
        sys.exit(1)

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Disable KV cache to avoid DynamicCache in output (not needed for single-pass)
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval()

    # Create a fixed-length dummy input (pad to seq_len)
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
    # Fill with a real tokenized sequence for tracing
    dummy_text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(dummy_text, add_special_tokens=False)
    fill_len = min(len(tokens), seq_len)
    dummy_ids[0, :fill_len] = torch.tensor(tokens[:fill_len])

    # Export with FIXED axes (no dynamic_axes → shapes are static)
    onnx_dir = Path(tempfile.mkdtemp())
    onnx_path = str(onnx_dir / f"{model_name.replace('/', '_')}.onnx")

    print(f"  Exporting with opset {opset}, seq_len={seq_len}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids,),
            onnx_path,
            opset_version=opset,
            input_names=["input_ids"],
            output_names=["logits"],
            # No dynamic_axes → all shapes are static
        )

    return onnx_path


if __name__ == "__main__":
    main()
