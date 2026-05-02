"""ONNX → .mpmodel converter."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mempipe_convert.mpmodel import (
    Metadata,
    MpModel,
    OpNode,
    OpType,
    Shape,
    serialize_model,
)

# ── ONNX op type → .mpmodel op type mapping ─────────────────────────────────

_ONNX_OP_MAP: Dict[str, OpType] = {
    "Gemm": OpType.Dense,
    "MatMul": OpType.MatMul,
    "Add": OpType.Add,
    "Relu": OpType.ReLU,
    "Sigmoid": OpType.Sigmoid,
    "Softmax": OpType.Softmax,
    "Conv": OpType.Conv2D,
    "MaxPool": OpType.MaxPool2D,
    "AveragePool": OpType.AvgPool2D,
    "BatchNormalization": OpType.BatchNorm,
    "Flatten": OpType.Flatten,
    "Reshape": OpType.Reshape,
    "Concat": OpType.Concat,
    # Transformer ops
    "Gelu": OpType.GELU,
    "LayerNormalization": OpType.LayerNorm,
    "Gather": OpType.Gather,
    "Mul": OpType.Mul,
    "Sub": OpType.Sub,
    "Transpose": OpType.Transpose,
    "Slice": OpType.Slice,
    "Tanh": OpType.Tanh,
    "Where": OpType.Where,
    "Split": OpType.Split,
    "Div": OpType.Div,
    "Pow": OpType.Pow,
    "IsNaN": OpType.IsNaN,
    "And": OpType.And,
}


def _encode_pool2d_attrs(node) -> bytes:
    """Encode MaxPool / AveragePool for MemPipe inference (matches Go parsePool2DAttrs).

    Layout (little-endian u16): kernelH, kernelW, strideH, strideW,
    padTop, padLeft, padBottom, padRight.

    ONNX defaults: strides 1 per spatial dim when absent; pads 0 when absent.
    """
    kh, kw = 1, 1
    sh, sw = 1, 1
    pt, pl, pb, pr = 0, 0, 0, 0
    for attr in node.attribute:
        if attr.name == "kernel_shape":
            ks = list(attr.ints)
            if len(ks) >= 2:
                kh, kw = int(ks[0]), int(ks[1])
            elif len(ks) == 1:
                kh = kw = int(ks[0])
        elif attr.name == "strides":
            st = list(attr.ints)
            if len(st) >= 2:
                sh, sw = int(st[0]), int(st[1])
            elif len(st) == 1:
                sh = sw = int(st[0])
        elif attr.name == "pads":
            p = list(attr.ints)
            if len(p) >= 4:
                pt, pl, pb, pr = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            elif len(p) == 2:
                pt, pl = int(p[0]), int(p[1])
                pb, pr = pt, pl
    return struct.pack("<HHHHHHHH", kh, kw, sh, sw, pt, pl, pb, pr)


def from_onnx(
    onnx_path: str,
    output_path: str,
    *,
    quantize: Optional[str] = None,
    name: Optional[str] = None,
    platform_hints: str = "",
) -> MpModel:
    """Convert an ONNX model to .mpmodel and write to disk.

    Args:
        onnx_path: Path to the .onnx file.
        output_path: Path to write the .mpmodel file.
        quantize: Optional quantization method: "int8" or "fp16".
        name: Model name (defaults to filename stem).
        platform_hints: Target platform hints string.

    Returns:
        The converted MpModel.
    """
    try:
        import onnx  # noqa: F811
    except ImportError as e:
        raise ImportError("onnx required: uv pip install onnx") from e

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_name = name or Path(onnx_path).stem
    graph = onnx_model.graph

    # ── Extract initializers (weights) ────────────────────────────────────
    initializers: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        arr = _onnx_tensor_to_numpy(init)
        initializers[init.name] = arr

    # ── Build tensor name table ───────────────────────────────────────────
    tensor_names: List[str] = []
    tensor_index: Dict[str, int] = {}
    tensor_shapes: Dict[str, Shape] = {}

    def _ensure_tensor(tname: str, shape: Optional[List[int]] = None) -> int:
        if tname in tensor_index:
            if shape and tname not in tensor_shapes:
                tensor_shapes[tname] = Shape(shape)
            return tensor_index[tname]
        idx = len(tensor_names)
        tensor_names.append(tname)
        tensor_index[tname] = idx
        if shape:
            tensor_shapes[tname] = Shape(shape)
        return idx

    # Register graph inputs
    for inp in graph.input:
        if inp.name not in initializers:
            shape = _get_onnx_shape(inp)
            _ensure_tensor(inp.name, shape)

    # Register initializers
    for iname, arr in initializers.items():
        _ensure_tensor(iname, list(arr.shape))

    # Register graph outputs
    output_names = []
    for out in graph.output:
        shape = _get_onnx_shape(out)
        _ensure_tensor(out.name, shape)
        output_names.append(out.name)

    # ── Convert ONNX nodes → OpNodes ──────────────────────────────────────
    op_nodes: List[OpNode] = []
    skipped = 0

    for node in graph.node:
        mp_op = _ONNX_OP_MAP.get(node.op_type)
        if mp_op is None:
            skipped += 1
            continue

        # Register intermediate tensors
        input_indices = []
        for inp_name in node.input:
            if not inp_name:
                continue
            idx = _ensure_tensor(inp_name)
            input_indices.append(idx)

        output_indices = []
        for out_name in node.output:
            if not out_name:
                continue
            idx = _ensure_tensor(out_name)
            output_indices.append(idx)

        attrs = b""
        if node.op_type in ("MaxPool", "AveragePool"):
            attrs = _encode_pool2d_attrs(node)

        op_nodes.append(OpNode(mp_op, input_indices, output_indices, attrs))

    if skipped:
        import warnings

        warnings.warn(f"Skipped {skipped} unsupported ONNX ops")

    # ── Pack weights ──────────────────────────────────────────────────────
    weights_blob, weight_offsets = _pack_weights(initializers, tensor_names)

    # ── Quantization ──────────────────────────────────────────────────────
    quant_method = ""
    quant_scale = 0.0
    quant_zero = 0
    if quantize == "int8":
        weights_blob, quant_scale, quant_zero = _quantize_weights_int8(weights_blob)
        quant_method = "int8_symmetric"

    # ── Extract input/output shapes ───────────────────────────────────────
    input_shapes = []
    for inp in graph.input:
        if inp.name not in initializers:
            shape = _get_onnx_shape(inp)
            if shape:
                input_shapes.append(Shape(shape))

    output_shapes = []
    for out in graph.output:
        shape = _get_onnx_shape(out)
        if shape:
            output_shapes.append(Shape(shape))

    # ── Build model ───────────────────────────────────────────────────────
    model = MpModel(
        metadata=Metadata(
            name=model_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            quant_method=quant_method,
            quant_scale=quant_scale,
            quant_zero=quant_zero,
            platform_hints=platform_hints,
        ),
        graph=op_nodes,
        tensor_names=tensor_names,
        tensor_shapes=tensor_shapes,
        weights_blob=weights_blob,
    )

    # ── Serialize and write ───────────────────────────────────────────────
    data = serialize_model(model)
    Path(output_path).write_bytes(data)

    return model


# ── Helpers ──────────────────────────────────────────────────────────────────


def _onnx_tensor_to_numpy(tensor) -> np.ndarray:
    """Convert an ONNX TensorProto to a numpy array."""
    import onnx

    return np.array(onnx.numpy_helper.to_array(tensor)).astype(np.float32)


def _get_onnx_shape(value_info) -> Optional[List[int]]:
    """Extract shape from an ONNX ValueInfoProto, returning None for dynamic dims."""
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.dim_param:
            dims.append(1)  # replace dynamic dim with 1 (batch)
        else:
            dims.append(d.dim_value)
    return dims if dims else None


def _pack_weights(
    initializers: Dict[str, np.ndarray],
    tensor_names: List[str],
) -> Tuple[bytes, Dict[str, int]]:
    """Pack initializer weights into a single aligned blob.

    The order follows tensor_names (so that weight tensor index i has its data
    at a known offset). Only tensors present in initializers are packed.
    Returns the blob and a mapping of name → byte offset.
    """
    parts: List[bytes] = []
    offsets: Dict[str, int] = {}
    pos = 0

    for name in tensor_names:
        if name not in initializers:
            continue
        arr = initializers[name]
        raw = arr.astype(np.float32).tobytes()
        offsets[name] = pos
        parts.append(raw)
        pos += len(raw)

    return b"".join(parts), offsets


def _quantize_weights_int8(
    weights_blob: bytes,
) -> Tuple[bytes, float, int]:
    """Apply symmetric INT8 quantization to float32 weights."""
    arr = np.frombuffer(weights_blob, dtype=np.float32)
    abs_max = float(np.max(np.abs(arr))) if len(arr) > 0 else 1.0
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized.tobytes(), scale, 0
