#!/usr/bin/env python3
"""Convert an MNIST MLP ONNX model to .mpmodel.

Usage:
    uv run zoo/convert_mnist.py

This creates a simple MNIST MLP in ONNX format, then converts it to .mpmodel.
Requires: numpy, onnx
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mempipe_convert.mpmodel import (
    Metadata,
    MpModel,
    OpNode,
    OpType,
    Shape,
    serialize_model,
)


def create_mnist_mlp() -> MpModel:
    """Create a simple MNIST MLP model (784→128→10) with random weights."""
    rng = np.random.default_rng(42)

    # Weight shapes
    w1 = rng.standard_normal((784, 128)).astype(np.float32) * 0.01
    b1 = np.zeros(128, dtype=np.float32)
    w2 = rng.standard_normal((128, 10)).astype(np.float32) * 0.01
    b2 = np.zeros(10, dtype=np.float32)

    weights_blob = b"".join([w.tobytes() for w in [w1, b1, w2, b2]])

    tensor_names = ["input", "w1", "b1", "h1", "h1_relu", "w2", "b2", "output"]

    tensor_shapes = {
        "w1": Shape([784, 128]),
        "b1": Shape([128]),
        "w2": Shape([128, 10]),
        "b2": Shape([10]),
    }

    graph = [
        OpNode(OpType.Dense, [0, 1, 2], [3]),     # h1 = input @ w1 + b1
        OpNode(OpType.ReLU, [3], [4]),             # h1_relu = relu(h1)
        OpNode(OpType.Dense, [4, 5, 6], [7]),      # output = h1_relu @ w2 + b2
        OpNode(OpType.Softmax, [7], [7]),           # output = softmax(output)
    ]

    return MpModel(
        metadata=Metadata(
            name="mnist-mlp",
            input_shapes=[Shape([1, 784])],
            output_shapes=[Shape([1, 10])],
        ),
        graph=graph,
        tensor_names=tensor_names,
        tensor_shapes=tensor_shapes,
        weights_blob=weights_blob,
    )


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "mnist_mlp.mpmodel"

    model = create_mnist_mlp()
    data = serialize_model(model)
    out_path.write_bytes(data)

    total_params = len(model.weights_blob) // 4
    print(f"Created {out_path}")
    print(f"  Size: {len(data):,} bytes")
    print(f"  Params: {total_params:,}")
    print(f"  Ops: {len(model.graph)}")
    print(f"  Tensors: {len(model.tensor_names)}")


if __name__ == "__main__":
    main()
