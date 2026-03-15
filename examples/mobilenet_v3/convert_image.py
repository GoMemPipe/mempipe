#!/usr/bin/env python3
"""Convert any image to the raw float32 NCHW format required by MobileNetV3.

Usage:
    python3 convert_image.py                           # converts default sample image
    python3 convert_image.py photo.jpg                 # converts photo.jpg
    python3 convert_image.py photo.jpg -o input.rgb    # custom output name
    python3 convert_image.py photo.jpg --size 224      # custom size (default: 224)

Output: a raw binary file with shape [1, 3, H, W] as float32, pixel values in [0, 1].

NOTE: This MobileNet ONNX model already includes Sub/Div ops for ImageNet
      normalization as its first two graph nodes. Pass raw [0,1]-scaled pixels;
      do NOT apply external normalization (the --normalize flag is available
      for models that lack internal normalization).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# Default sample image shipped with the example
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "nyc-complete-street-brooklyn-cropped.webp")


def convert_image(
    input_path: str,
    output_path: str,
    size: int = 224,
    normalize: bool = True,
) -> None:
    """Load an image, resize to (size, size), normalize, and save as raw float32 NCHW."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required. Install it with: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size

    # Center-crop to square, then resize (matches standard ImageNet preprocessing)
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to float32 array [H, W, C]
    pixels = list(img.getdata())  # list of (R, G, B) tuples
    h, w = size, size

    # ImageNet normalization constants
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Pack as NCHW: [1, 3, H, W] = 1*3*224*224 = 150528 floats
    float_data = []
    for c in range(3):  # R, G, B channels
        for row in range(h):
            for col in range(w):
                pixel_idx = row * w + col
                val = pixels[pixel_idx][c] / 255.0
                if normalize:
                    val = (val - mean[c]) / std[c]
                float_data.append(val)

    # Write raw float32 binary
    raw = struct.pack(f"<{len(float_data)}f", *float_data)
    Path(output_path).write_bytes(raw)

    n_floats = len(float_data)
    n_bytes = len(raw)

    print(f"Converted: {input_path}")
    print(f"  Original:   {orig_w}×{orig_h}")
    print(f"  Resized:    {size}×{size} (center-cropped)")
    print(f"  Normalized: {'ImageNet (mean/std)' if normalize else 'raw [0,1]'}")
    print(f"  Shape:      [1, 3, {h}, {w}] ({n_floats} floats)")
    print(f"  Output:     {output_path} ({n_bytes:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image to MobileNetV3 raw float32 input format"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=f"Input image path (default: {os.path.basename(DEFAULT_IMAGE)})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .rgb file path (default: <input_stem>.rgb in same directory as this script)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Resize to size×size (default: 224)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply ImageNet normalization (only for models without internal norm ops)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem = Path(args.image).stem
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, f"{stem}.rgb")

    convert_image(
        args.image,
        args.output,
        size=args.size,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
