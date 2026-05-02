"""Transformer-specific ONNX → .mpmodel converter with pattern fusion.

Handles GPT-2 and similar Transformer models by:
1. Extending the ONNX op mapping with Transformer-specific operators.
2. Detecting and fusing decomposed GELU / LayerNorm sub-graphs.
3. Encoding operator-specific attributes (Transpose perm, Slice range, etc.).
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    onnx = None  # type: ignore[assignment]

from mempipe_convert.mpmodel import (
    Metadata,
    MpModel,
    OpNode,
    OpType,
    Shape,
    serialize_model,
)

# ── Extended ONNX op mapping (includes Transformer ops) ─────────────────────

_TRANSFORMER_ONNX_OP_MAP: Dict[str, OpType] = {
    # Original MLP / CNN ops
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
    # Transformer-specific ops
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
    # MobileNet / CNN ops
    "GlobalAveragePool": OpType.GlobalAvgPool2D,
    "HardSigmoid": OpType.HardSigmoid,
    "HardSwish": OpType.HardSwish,
}

# Ops we treat as shape-only / no-ops (skip in the graph)
_SKIP_OPS: Set[str] = {
    "Shape",
    "Unsqueeze",
    "Squeeze",
    "Cast",
    "Constant",
    "ConstantOfShape",
    "Identity",
    "Dropout",
}


# ── GELU fusion pattern detector ────────────────────────────────────────────


def _detect_gelu_pattern(
    nodes: List,
    idx: int,
    output_to_node: Dict[str, int],
) -> Optional[Tuple[int, str, str]]:
    """Detect the decomposed GELU pattern starting at node[idx].

    Common ONNX GELU decomposition:
        x → Div(sqrt(2)) → Erf → Add(1) → Mul → Mul(0.5) → y
    or:
        x → Mul(sqrt(2/pi)) → Pow(3) → Mul(0.044715) → Add → Tanh → Add(1) → Mul → Mul(0.5) → y

    Returns (num_nodes_consumed, input_name, output_name) or None.
    """
    if idx >= len(nodes):
        return None
    node = nodes[idx]

    # Pattern 1: Div → Erf → Add → Mul → Mul
    if node.op_type == "Div":
        remaining = len(nodes) - idx
        if remaining >= 5:
            n1, n2, n3, n4 = nodes[idx + 1], nodes[idx + 2], nodes[idx + 3], nodes[idx + 4]
            if (
                n1.op_type == "Erf"
                and n2.op_type == "Add"
                and n3.op_type == "Mul"
                and n4.op_type == "Mul"
            ):
                return (5, node.input[0], n4.output[0])

    # Pattern 2: tanh approximation variant — harder to detect reliably
    # We rely on the fused "Gelu" op from opset 20+ instead.
    return None


# ── LayerNorm fusion pattern detector ───────────────────────────────────────


def _detect_layernorm_pattern(
    nodes: List,
    idx: int,
    initializers: Dict[str, np.ndarray],
    output_to_node: Dict[str, int],
) -> Optional[Tuple[int, str, str, str, str]]:
    """Detect decomposed LayerNorm: ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add.

    Returns (num_nodes, input_name, output_name, gamma_name, beta_name) or None.
    """
    if idx + 8 >= len(nodes):
        return None

    n0 = nodes[idx]
    if n0.op_type != "ReduceMean":
        return None

    seq = [nodes[idx + i] for i in range(1, 9)]
    expected = ["Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"]
    if [n.op_type for n in seq] != expected:
        return None

    input_name = n0.input[0]
    output_name = seq[-1].output[0]

    # gamma is the second input to the Mul (seq[6])
    gamma_name = seq[6].input[1] if len(seq[6].input) > 1 else ""
    # beta is the second input to the final Add (seq[7])
    beta_name = seq[7].input[1] if len(seq[7].input) > 1 else ""

    return (9, input_name, output_name, gamma_name, beta_name)


# ── Attrs encoding helpers ──────────────────────────────────────────────────


def _encode_transpose_attrs(node) -> bytes:
    """Encode Transpose perm attribute: [ndims u16, perm0 u16, ...]."""
    perm = []
    for attr in node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
            break
    if not perm:
        return b""
    ndims = len(perm)
    fmt = "<" + "H" * (1 + ndims)
    return struct.pack(fmt, ndims, *perm)


def _init_int(arr: np.ndarray, idx: int = 0) -> int:
    """Read an integer value from an initializer that may be stored as int32-viewed-as-float32."""
    flat = arr.flatten()
    if flat.dtype == np.float32:
        return int(flat.view(np.int32)[idx])
    return int(flat[idx])


def _encode_slice_attrs(node, initializers: Dict[str, np.ndarray]) -> bytes:
    """Encode Slice attrs: [axis u16, start i32, end i32].

    ONNX Slice takes starts/ends/axes/steps as inputs (not attributes),
    so we resolve them from initializers.
    """
    if len(node.input) < 3:
        return b""

    starts_name = node.input[1]
    ends_name = node.input[2]
    axes_name = node.input[3] if len(node.input) > 3 else ""

    start = 0
    end = 0
    axis = 0

    if starts_name in initializers:
        start = _init_int(initializers[starts_name])
    if ends_name in initializers:
        end = _init_int(initializers[ends_name])
    if axes_name and axes_name in initializers:
        axis = _init_int(initializers[axes_name])

    return struct.pack("<Hii", axis, start, end)


def _encode_split_attrs(node) -> bytes:
    """Encode Split attrs: [axis u16, numSplits u16]."""
    axis = 0
    num_outputs = len(node.output)
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
    return struct.pack("<HH", axis, num_outputs)


def _encode_conv_attrs(node) -> bytes:
    """Encode Conv attrs: [group u16, strideH u16, strideW u16, padTop u16, padLeft u16, padBottom u16, padRight u16, dilH u16, dilW u16]."""
    group = 1
    strides = [1, 1]
    pads = [0, 0, 0, 0]  # top, left, bottom, right
    dilations = [1, 1]
    for attr in node.attribute:
        if attr.name == "group":
            group = int(attr.i)
        elif attr.name == "strides":
            strides = list(attr.ints) or [1, 1]
        elif attr.name == "pads":
            pads = list(attr.ints) or [0, 0, 0, 0]
        elif attr.name == "dilations":
            dilations = list(attr.ints) or [1, 1]
    # Ensure we have 2 strides and 4 pads
    if len(strides) == 1:
        strides = [strides[0], strides[0]]
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(dilations) == 1:
        dilations = [dilations[0], dilations[0]]
    return struct.pack(
        "<HHHHHHHHH",
        group,
        strides[0], strides[1],
        pads[0], pads[1], pads[2], pads[3],
        dilations[0], dilations[1],
    )


def _encode_softmax_attrs(node) -> bytes:
    """Encode Softmax axis as int16 LE (ONNX axis attribute; default -1 = last axis)."""
    axis = -1
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
            break
    return struct.pack("<h", axis)


def _encode_hardsigmoid_attrs(node) -> bytes:
    """Encode HardSigmoid attrs: [alpha f32, beta f32]."""
    alpha = 0.2  # ONNX default
    beta = 0.5   # ONNX default
    for attr in node.attribute:
        if attr.name == "alpha":
            alpha = float(attr.f)
        elif attr.name == "beta":
            beta = float(attr.f)
    return struct.pack("<ff", alpha, beta)


def _encode_pool2d_attrs(node) -> bytes:
    """Encode MaxPool / AveragePool for MemPipe (matches Go parsePool2DAttrs)."""
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


# ── MatMul 2D vs 3D dispatch ────────────────────────────────────────────────


def _encode_reshape_attrs(
    node,
    initializers: Dict[str, np.ndarray],
    inferred_shapes: Dict[str, List[int]],
    tensor_shapes: Dict[str, Shape],
) -> bytes:
    """Encode Reshape target shape: [ndims u16, dim0 i32, dim1 i32, ...].

    Resolve the target shape from (in priority order):
      1. The 2nd input (a shape initializer)
      2. The ONNX shape-inference output annotation
    Also resolves -1 and 0 placeholders.
    """
    target: Optional[List[int]] = None

    # Try: read shape from the second input (initializer constant)
    if len(node.input) > 1:
        shape_name = node.input[1]
        if shape_name in initializers:
            # Shape initializers may be stored as int32 viewed as float32;
            # recover the original int values via view.
            raw = initializers[shape_name].flatten()
            if raw.dtype == np.float32:
                target = raw.view(np.int32).tolist()
            else:
                target = raw.astype(int).tolist()

    # Fall back: use inferred output shape
    if target is None and node.output[0] in inferred_shapes:
        target = list(inferred_shapes[node.output[0]])

    if target is None:
        return b""

    # Resolve -1 and 0 dims
    inp_name = node.input[0]
    inp_shape = tensor_shapes.get(inp_name)
    if inp_shape:
        total = inp_shape.numel()
        known = 1
        neg_idx = -1
        for i, d in enumerate(target):
            if d == 0 and i < len(inp_shape.dims):
                target[i] = inp_shape.dims[i]
            if d == -1:
                neg_idx = i
        for i, d in enumerate(target):
            if d > 0:
                known *= d
        if neg_idx >= 0 and known > 0:
            target[neg_idx] = total // known

    ndims = len(target)
    return struct.pack("<H" + "i" * ndims, ndims, *target)


# ── Dynamic dim fixing + shape inference ────────────────────────────────────


def _fix_dynamic_dims(onnx_model, seq_len: int):
    """Replace dynamic dimensions with concrete values in-place."""
    for inp in onnx_model.graph.input:
        t = inp.type.tensor_type
        if t.HasField("shape"):
            for dim in t.shape.dim:
                if dim.dim_param:
                    if "batch" in dim.dim_param.lower():
                        dim.ClearField("dim_param")
                        dim.dim_value = 1
                    else:
                        dim.ClearField("dim_param")
                        dim.dim_value = seq_len

    for out in onnx_model.graph.output:
        t = out.type.tensor_type
        if t.HasField("shape"):
            for dim in t.shape.dim:
                if dim.dim_param:
                    if "batch" in dim.dim_param.lower():
                        dim.ClearField("dim_param")
                        dim.dim_value = 1
                    else:
                        dim.ClearField("dim_param")
                        dim.dim_value = seq_len


def _collect_inferred_shapes(onnx_model) -> Dict[str, List[int]]:
    """Collect concrete shapes from value_info annotations."""
    shapes: Dict[str, List[int]] = {}
    for vi in onnx_model.graph.value_info:
        t = vi.type.tensor_type
        if t.HasField("shape"):
            dims = []
            for d in t.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    break
            else:
                if dims:
                    shapes[vi.name] = dims
    # Also add graph output shapes
    for out in onnx_model.graph.output:
        t = out.type.tensor_type
        if t.HasField("shape"):
            dims = []
            for d in t.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    break
            else:
                if dims:
                    shapes[out.name] = dims
    return shapes


# ── MatMul 2D vs 3D dispatch ────────────────────────────────────────────────


def _classify_matmul(
    node,
    tensor_shapes: Dict[str, Shape],
) -> OpType:
    """Return MatMul for 2D inputs, BatchedMatMul for 3D+."""
    for inp in node.input:
        if inp in tensor_shapes:
            if len(tensor_shapes[inp].dims) >= 3:
                return OpType.BatchedMatMul
    return OpType.MatMul


# ── Main converter ──────────────────────────────────────────────────────────


def from_onnx_transformer(
    onnx_path: str,
    output_path: str,
    *,
    quantize: Optional[str] = None,
    name: Optional[str] = None,
    platform_hints: str = "",
    fuse_patterns: bool = True,
    seq_len: int = 128,
) -> MpModel:
    """Convert a Transformer ONNX model to .mpmodel with fusion passes.

    Args:
        onnx_path: Path to the .onnx file.
        output_path: Path to write the .mpmodel file.
        quantize: Optional quantization: "int8" or "fp16".
        name: Model name (defaults to filename stem).
        platform_hints: Target platform hints.
        fuse_patterns: If True, detect and fuse GELU/LayerNorm decompositions.
        seq_len: Fixed sequence length for dynamic dims (default: 128).

    Returns:
        The converted MpModel.
    """
    if onnx is None:
        raise ImportError("onnx required: pip install onnx")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Fix dynamic dimensions and run shape inference
    _fix_dynamic_dims(onnx_model, seq_len)
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        warnings.warn(f"ONNX shape inference failed: {e}")
    inferred_shapes = _collect_inferred_shapes(onnx_model)

    model_name = name or Path(onnx_path).stem
    graph = onnx_model.graph

    # ── Extract initializers ──────────────────────────────────────────────
    initializers: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        arr = np.array(numpy_helper.to_array(init))
        if arr.dtype in (np.int64, np.int32, np.int16, np.int8):
            # Integer tensors (indices, shape constants): preserve as int32 bits
            # viewed through float32 (so Go's Int32s() recovers the values).
            arr = arr.astype(np.int32).view(np.float32)
        elif arr.dtype == np.bool_:
            # Boolean tensors: True→int32(1), False→int32(0), bit-viewed as float32.
            arr = arr.astype(np.int32).view(np.float32)
        else:
            arr = arr.astype(np.float32)
        initializers[init.name] = arr

    # ── Pre-transpose Gemm weights with transB=1 ─────────────────────────
    for node in graph.node:
        if node.op_type == "Gemm":
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = int(attr.i)
            if transB and len(node.input) >= 2:
                w_name = node.input[1]
                if w_name in initializers:
                    w = initializers[w_name]
                    if w.ndim == 2:
                        initializers[w_name] = w.T.copy()

    # ── Tensor name table ─────────────────────────────────────────────────
    tensor_names: List[str] = []
    tensor_index: Dict[str, int] = {}
    tensor_shapes: Dict[str, Shape] = {}

    def _ensure_tensor(tname: str, shape: Optional[List[int]] = None) -> int:
        if tname in tensor_index:
            if shape is not None and tname not in tensor_shapes:
                tensor_shapes[tname] = Shape(shape if shape else [1])
            return tensor_index[tname]
        idx = len(tensor_names)
        tensor_names.append(tname)
        tensor_index[tname] = idx
        if shape is not None:
            tensor_shapes[tname] = Shape(shape if shape else [1])
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

    # ── Build output→node index for fusion ────────────────────────────────
    output_to_node: Dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            output_to_node[out] = i

    # ── Convert nodes with fusion passes ──────────────────────────────────
    op_nodes: List[OpNode] = []
    skipped = 0
    skip_until = -1  # index to skip to (for fused patterns)
    fused_gelu = 0
    fused_ln = 0

    i = 0
    nodes = list(graph.node)
    while i < len(nodes):
        node = nodes[i]

        # Skip nodes consumed by fusion
        if i < skip_until:
            i += 1
            continue

        # Skip shape-only / no-op nodes
        if node.op_type in _SKIP_OPS:
            # Still register their output tensors so downstream refs work
            for out_name in node.output:
                if out_name:
                    shape = inferred_shapes.get(out_name)
                    _ensure_tensor(out_name, shape)
            i += 1
            continue

        # ── Try GELU fusion ───────────────────────────────────────────────
        if fuse_patterns and node.op_type in ("Div", "Mul"):
            gelu = _detect_gelu_pattern(nodes, i, output_to_node)
            if gelu:
                count, inp_name, out_name = gelu
                in_idx = _ensure_tensor(inp_name)
                out_idx = _ensure_tensor(out_name)
                op_nodes.append(OpNode(OpType.GELU, [in_idx], [out_idx]))
                skip_until = i + count
                i = skip_until
                fused_gelu += 1
                continue

        # ── Try LayerNorm fusion ──────────────────────────────────────────
        if fuse_patterns and node.op_type == "ReduceMean":
            ln = _detect_layernorm_pattern(nodes, i, initializers, output_to_node)
            if ln:
                count, inp_name, out_name, gamma_name, beta_name = ln
                in_idx = _ensure_tensor(inp_name)
                gamma_idx = _ensure_tensor(gamma_name)
                beta_idx = _ensure_tensor(beta_name)
                out_idx = _ensure_tensor(out_name)
                op_nodes.append(
                    OpNode(OpType.LayerNorm, [in_idx, gamma_idx, beta_idx], [out_idx])
                )
                skip_until = i + count
                i = skip_until
                fused_ln += 1
                continue

        # ── Standard op mapping ───────────────────────────────────────────
        mp_op = _TRANSFORMER_ONNX_OP_MAP.get(node.op_type)

        # Special: classify MatMul as 2D or 3D
        if node.op_type == "MatMul":
            mp_op = _classify_matmul(node, tensor_shapes)

        if mp_op is None:
            skipped += 1
            # Register outputs so graph stays connected
            for out_name in node.output:
                if out_name:
                    shape = inferred_shapes.get(out_name)
                    _ensure_tensor(out_name, shape)
            i += 1
            continue

        # Register tensors
        input_indices = []
        for inp_name in node.input:
            if not inp_name:
                continue
            # Use inferred shape if we don't already have it
            shape = inferred_shapes.get(inp_name)
            idx = _ensure_tensor(inp_name, shape)
            input_indices.append(idx)

        output_indices = []
        for out_name in node.output:
            if not out_name:
                continue
            shape = inferred_shapes.get(out_name)
            idx = _ensure_tensor(out_name, shape)
            output_indices.append(idx)

        # Encode op-specific attributes
        attrs = b""
        if node.op_type == "Transpose":
            attrs = _encode_transpose_attrs(node)
        elif node.op_type == "Slice":
            attrs = _encode_slice_attrs(node, initializers)
        elif node.op_type == "Split":
            attrs = _encode_split_attrs(node)
        elif node.op_type == "Reshape":
            attrs = _encode_reshape_attrs(
                node, initializers, inferred_shapes, tensor_shapes
            )
        elif node.op_type == "Conv":
            attrs = _encode_conv_attrs(node)
        elif node.op_type == "Softmax":
            attrs = _encode_softmax_attrs(node)
        elif node.op_type == "HardSigmoid":
            attrs = _encode_hardsigmoid_attrs(node)
        elif node.op_type in ("MaxPool", "AveragePool"):
            attrs = _encode_pool2d_attrs(node)

        op_nodes.append(OpNode(mp_op, input_indices, output_indices, attrs))
        i += 1

    if skipped:
        warnings.warn(f"Skipped {skipped} unsupported ONNX ops during transformer conversion")
    if fused_gelu:
        print(f"  Fused {fused_gelu} GELU pattern(s)")
    if fused_ln:
        print(f"  Fused {fused_ln} LayerNorm pattern(s)")

    # ── Move output tensors to end of tensor_names ────────────────────────
    # The Go engine expects output tensors to be the LAST M entries.
    for oname in output_names:
        if oname not in tensor_index:
            continue
        old_idx = tensor_index[oname]
        if old_idx == len(tensor_names) - 1:
            continue  # already last
        # Remove from current position and append to end
        tensor_names.pop(old_idx)
        new_idx = len(tensor_names)
        tensor_names.append(oname)
        # Build old→new index mapping
        remap: Dict[int, int] = {}
        for ti, tn in enumerate(tensor_names):
            remap[tensor_index.get(tn, ti)] = ti
        # Rebuild tensor_index
        tensor_index.clear()
        for ti, tn in enumerate(tensor_names):
            tensor_index[tn] = ti
        # Remap all op_node input/output indices
        for opn in op_nodes:
            opn.input_indices = [remap.get(ii, ii) for ii in opn.input_indices]
            opn.output_indices = [remap.get(oi, oi) for oi in opn.output_indices]

    # ── Pack weights ──────────────────────────────────────────────────────
    weights_blob, _ = _pack_weights(initializers, tensor_names)

    # ── Quantization ──────────────────────────────────────────────────────
    quant_method = ""
    quant_scale = 0.0
    quant_zero = 0
    if quantize == "int8":
        weights_blob, quant_scale, quant_zero = _quantize_weights_int8(weights_blob)
        quant_method = "int8_symmetric"

    # ── Input / output shapes ─────────────────────────────────────────────
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

    data = serialize_model(model)
    Path(output_path).write_bytes(data)

    return model


# ── Helpers (mirrored from converter.py to keep this module self-contained) ──


def _get_onnx_shape(value_info) -> Optional[List[int]]:
    """Extract shape from ONNX ValueInfoProto."""
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.dim_param:
            dims.append(1)  # dynamic → batch=1
        else:
            dims.append(d.dim_value)
    return dims if dims else None


def _pack_weights(
    initializers: Dict[str, np.ndarray],
    tensor_names: List[str],
) -> Tuple[bytes, Dict[str, int]]:
    """Pack initializer weights into a single contiguous blob."""
    parts: List[bytes] = []
    offsets: Dict[str, int] = {}
    pos = 0
    for name in tensor_names:
        if name not in initializers:
            continue
        raw = initializers[name].astype(np.float32).tobytes()
        offsets[name] = pos
        parts.append(raw)
        pos += len(raw)
    return b"".join(parts), offsets


def _quantize_weights_int8(weights_blob: bytes) -> Tuple[bytes, float, int]:
    """Symmetric INT8 quantization."""
    arr = np.frombuffer(weights_blob, dtype=np.float32)
    abs_max = float(np.max(np.abs(arr))) if len(arr) > 0 else 1.0
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized.tobytes(), scale, 0
