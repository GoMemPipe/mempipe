"""Post-training quantization for .mpmodel.

Supports:
  - Dynamic INT8 symmetric (range from weight statistics)
  - Static INT8 with calibration data
  - FP16 weight compression
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from mempipe_convert.mpmodel import MpModel, load_model, serialize_model


def quantize(
    model_path: str,
    output_path: str,
    *,
    method: str = "dynamic",
    calibration_data: Optional[np.ndarray] = None,
) -> None:
    """Quantize a .mpmodel file.

    Args:
        model_path: Path to the input .mpmodel file.
        output_path: Path to write the quantized .mpmodel file.
        method: "dynamic" (INT8 symmetric from weights), "static" (with calibration),
                or "fp16" (FP16 weight storage).
        calibration_data: Numpy array of calibration inputs for static quantization.
    """
    data = Path(model_path).read_bytes()
    model = load_model(data)

    if method == "dynamic":
        _quantize_dynamic_int8(model)
    elif method == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        _quantize_static_int8(model, calibration_data)
    elif method == "fp16":
        _quantize_fp16(model)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    out_bytes = serialize_model(model)
    Path(output_path).write_bytes(out_bytes)


def _quantize_dynamic_int8(model: MpModel) -> None:
    """Apply dynamic INT8 symmetric quantization."""
    weights = np.frombuffer(model.weights_blob, dtype=np.float32)
    if len(weights) == 0:
        return

    abs_max = float(np.max(np.abs(weights)))
    if abs_max == 0:
        abs_max = 1.0

    scale = abs_max / 127.0
    quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)

    model.weights_blob = quantized.tobytes()
    model.metadata.quant_method = "int8_symmetric"
    model.metadata.quant_scale = scale
    model.metadata.quant_zero = 0


def _quantize_static_int8(model: MpModel, calibration_data: np.ndarray) -> None:
    """Apply static INT8 quantization with calibration data.

    For now, this uses the same symmetric approach but could be extended
    with per-channel calibration.
    """
    _quantize_dynamic_int8(model)
    # Future: run calibration data through the graph to compute per-layer ranges


def _quantize_fp16(model: MpModel) -> None:
    """Compress weights to FP16."""
    weights = np.frombuffer(model.weights_blob, dtype=np.float32)
    if len(weights) == 0:
        return

    fp16_weights = weights.astype(np.float16)
    model.weights_blob = fp16_weights.tobytes()
    model.metadata.quant_method = "fp16"
