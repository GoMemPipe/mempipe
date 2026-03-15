#!/usr/bin/env python3
"""Convert a MobileNet-v2-like ONNX model to .mpmodel.

Usage:
    uv run zoo/convert_mobilenet.py

Creates a simplified MobileNet-v2 structure with random weights.
For a real model, use: mempipe-convert onnx mobilenet_v2.onnx -o mobilenet.mpmodel
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mempipe_convert.mpmodel import (
    Metadata,
    MpModel,
    OpNode,
    OpType,
    Shape,
    serialize_model,
)


def create_mobilenet_like() -> MpModel:
    """Create a simplified MobileNet-v2-like model with random weights.

    Architecture (simplified):
      input [1,3,32,32] → Conv2D(3→16,3×3) → BN → ReLU
                         → Conv2D(16→32,3×3) → BN → ReLU
                         → AvgPool2D(2×2)
                         → Flatten
                         → Dense(32*14*14 → 10)
                         → Softmax
    """
    rng = np.random.default_rng(123)

    # Layer 1: Conv 3→16
    conv1_w = rng.standard_normal((16, 3, 3, 3)).astype(np.float32) * 0.01
    bn1_gamma = np.ones(16, dtype=np.float32)
    bn1_beta = np.zeros(16, dtype=np.float32)
    bn1_mean = np.zeros(16, dtype=np.float32)
    bn1_var = np.ones(16, dtype=np.float32)

    # Layer 2: Conv 16→32
    conv2_w = rng.standard_normal((32, 16, 3, 3)).astype(np.float32) * 0.01
    bn2_gamma = np.ones(32, dtype=np.float32)
    bn2_beta = np.zeros(32, dtype=np.float32)
    bn2_mean = np.zeros(32, dtype=np.float32)
    bn2_var = np.ones(32, dtype=np.float32)

    # Dense: 32*14*14 → 10
    fc_w = rng.standard_normal((32 * 14 * 14, 10)).astype(np.float32) * 0.01
    fc_b = np.zeros(10, dtype=np.float32)

    weights = [
        conv1_w, bn1_gamma, bn1_beta, bn1_mean, bn1_var,
        conv2_w, bn2_gamma, bn2_beta, bn2_mean, bn2_var,
        fc_w, fc_b,
    ]
    weights_blob = b"".join(w.tobytes() for w in weights)

    tensor_names = [
        "input",
        # Conv1
        "conv1_w", "bn1_g", "bn1_b", "bn1_m", "bn1_v",
        "conv1_out", "bn1_out", "relu1_out",
        # Conv2
        "conv2_w", "bn2_g", "bn2_b", "bn2_m", "bn2_v",
        "conv2_out", "bn2_out", "relu2_out",
        # Pool + classify
        "pool_out", "flat_out",
        "fc_w", "fc_b", "output",
    ]

    tensor_shapes = {
        "conv1_w": Shape([16, 3, 3, 3]),
        "bn1_g": Shape([16]),
        "bn1_b": Shape([16]),
        "bn1_m": Shape([16]),
        "bn1_v": Shape([16]),
        "conv2_w": Shape([32, 16, 3, 3]),
        "bn2_g": Shape([32]),
        "bn2_b": Shape([32]),
        "bn2_m": Shape([32]),
        "bn2_v": Shape([32]),
        "fc_w": Shape([32 * 14 * 14, 10]),
        "fc_b": Shape([10]),
    }

    idx = {n: i for i, n in enumerate(tensor_names)}

    graph = [
        # Conv1 → BN → ReLU
        OpNode(OpType.Conv2D,
               [idx["input"], idx["conv1_w"]],
               [idx["conv1_out"]]),
        OpNode(OpType.BatchNorm,
               [idx["conv1_out"], idx["bn1_g"], idx["bn1_b"], idx["bn1_m"], idx["bn1_v"]],
               [idx["bn1_out"]]),
        OpNode(OpType.ReLU, [idx["bn1_out"]], [idx["relu1_out"]]),

        # Conv2 → BN → ReLU
        OpNode(OpType.Conv2D,
               [idx["relu1_out"], idx["conv2_w"]],
               [idx["conv2_out"]]),
        OpNode(OpType.BatchNorm,
               [idx["conv2_out"], idx["bn2_g"], idx["bn2_b"], idx["bn2_m"], idx["bn2_v"]],
               [idx["bn2_out"]]),
        OpNode(OpType.ReLU, [idx["bn2_out"]], [idx["relu2_out"]]),

        # Pool → Flatten → Dense → Softmax
        OpNode(OpType.AvgPool2D, [idx["relu2_out"]], [idx["pool_out"]]),
        OpNode(OpType.Flatten, [idx["pool_out"]], [idx["flat_out"]]),
        OpNode(OpType.Dense,
               [idx["flat_out"], idx["fc_w"], idx["fc_b"]],
               [idx["output"]]),
        OpNode(OpType.Softmax, [idx["output"]], [idx["output"]]),
    ]

    return MpModel(
        metadata=Metadata(
            name="mobilenet-v2-mini",
            input_shapes=[Shape([1, 3, 32, 32])],
            output_shapes=[Shape([1, 10])],
        ),
        graph=graph,
        tensor_names=tensor_names,
        tensor_shapes=tensor_shapes,
        weights_blob=weights_blob,
    )


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "mobilenet_mini.mpmodel"

    model = create_mobilenet_like()
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
