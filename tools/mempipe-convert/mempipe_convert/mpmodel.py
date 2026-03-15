"""Core .mpmodel binary format — mirrors inference/model.go exactly."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


# ── Constants ────────────────────────────────────────────────────────────────

MAGIC = b"MPMD"
FORMAT_VERSION = 1
HEADER_SIZE = 64
WEIGHT_ALIGN = 64

# Header flags
FLAG_QUANTIZED_INT8 = 1 << 0
FLAG_QUANTIZED_FP16 = 1 << 1
FLAG_HAS_CALIBRATION = 1 << 2


# ── Op types (must match inference/model.go OpType enum) ─────────────────────

class OpType(IntEnum):
    MatMul = 0
    Add = 1
    ReLU = 2
    Sigmoid = 3
    Softmax = 4
    Conv2D = 5
    MaxPool2D = 6
    AvgPool2D = 7
    BatchNorm = 8
    Flatten = 9
    Reshape = 10
    Concat = 11
    Dense = 12  # fused MatMul + BiasAdd
    Quantize = 13
    Dequantize = 14
    # Transformer ops
    GELU = 15
    LayerNorm = 16
    Gather = 17
    BatchedMatMul = 18
    Mul = 19
    Sub = 20
    Transpose = 21
    Slice = 22
    Tanh = 23
    Where = 24
    Split = 25
    Div = 26
    Pow = 27
    IsNaN = 28
    And = 29
    GlobalAvgPool2D = 30
    HardSigmoid = 31
    HardSwish = 32


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Shape:
    dims: List[int]

    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class OpNode:
    op_type: OpType
    input_indices: List[int]
    output_indices: List[int]
    attrs: bytes = b""


@dataclass
class Metadata:
    name: str = ""
    input_shapes: List[Shape] = field(default_factory=list)
    output_shapes: List[Shape] = field(default_factory=list)
    quant_method: str = ""
    quant_scale: float = 0.0
    quant_zero: int = 0
    platform_hints: str = ""


@dataclass
class MpModelHeader:
    magic: bytes = MAGIC
    version: int = FORMAT_VERSION
    flags: int = 0
    metadata_offset: int = 0
    metadata_size: int = 0
    graph_offset: int = 0
    graph_size: int = 0
    weights_offset: int = 0
    weights_size: int = 0
    total_size: int = 0


@dataclass
class MpModel:
    metadata: Metadata = field(default_factory=Metadata)
    graph: List[OpNode] = field(default_factory=list)
    tensor_names: List[str] = field(default_factory=list)
    tensor_shapes: Dict[str, Shape] = field(default_factory=dict)
    weights_blob: bytes = b""

    def input_size(self) -> int:
        return sum(s.numel() for s in self.metadata.input_shapes)

    def output_size(self) -> int:
        return sum(s.numel() for s in self.metadata.output_shapes)


# ── Serialization ────────────────────────────────────────────────────────────


def _encode_metadata(md: Metadata) -> bytes:
    """Encode metadata to binary (matches Go encodeMetadata)."""
    buf = bytearray()

    # name
    name_bytes = md.name.encode("utf-8")
    buf += struct.pack("<H", len(name_bytes))
    buf += name_bytes

    # input shapes
    buf += struct.pack("<H", len(md.input_shapes))
    for s in md.input_shapes:
        buf += struct.pack("<H", len(s.dims))
        for d in s.dims:
            buf += struct.pack("<i", d)

    # output shapes
    buf += struct.pack("<H", len(md.output_shapes))
    for s in md.output_shapes:
        buf += struct.pack("<H", len(s.dims))
        for d in s.dims:
            buf += struct.pack("<i", d)

    # quant
    qm_bytes = md.quant_method.encode("utf-8")
    buf += struct.pack("<H", len(qm_bytes))
    buf += qm_bytes
    buf += struct.pack("<f", md.quant_scale)
    buf += struct.pack("<i", md.quant_zero)

    # platform hints
    ph_bytes = md.platform_hints.encode("utf-8")
    buf += struct.pack("<H", len(ph_bytes))
    buf += ph_bytes

    return bytes(buf)


def _encode_graph(
    nodes: List[OpNode],
    tensor_names: List[str],
    tensor_shapes: Optional[Dict[str, Shape]] = None,
) -> bytes:
    """Encode graph section to binary (matches Go encodeGraph)."""
    buf = bytearray()

    # tensor names
    buf += struct.pack("<H", len(tensor_names))
    for name in tensor_names:
        nb = name.encode("utf-8")
        buf += struct.pack("<H", len(nb))
        buf += nb

    # nodes
    buf += struct.pack("<H", len(nodes))
    for n in nodes:
        buf += struct.pack("<H", int(n.op_type))
        buf += struct.pack("<H", len(n.input_indices))
        for idx in n.input_indices:
            buf += struct.pack("<H", idx)
        buf += struct.pack("<H", len(n.output_indices))
        for idx in n.output_indices:
            buf += struct.pack("<H", idx)
        buf += struct.pack("<H", len(n.attrs))
        buf += n.attrs

    # tensor shapes
    shapes = tensor_shapes or {}
    buf += struct.pack("<H", len(shapes))
    for name, s in shapes.items():
        nb = name.encode("utf-8")
        buf += struct.pack("<H", len(nb))
        buf += nb
        buf += struct.pack("<H", len(s.dims))
        for d in s.dims:
            buf += struct.pack("<i", d)

    return bytes(buf)


def _align64(v: int) -> int:
    return (v + 63) & ~63


def _flags_from_metadata(md: Metadata) -> int:
    flags = 0
    if md.quant_method in ("int8_symmetric", "int8_asymmetric"):
        flags |= FLAG_QUANTIZED_INT8
    if md.quant_method == "fp16":
        flags |= FLAG_QUANTIZED_FP16
    return flags


def serialize_model(model: MpModel) -> bytes:
    """Serialize an MpModel to .mpmodel binary bytes."""
    meta_bytes = _encode_metadata(model.metadata)
    graph_bytes = _encode_graph(model.graph, model.tensor_names, model.tensor_shapes)

    meta_off = HEADER_SIZE
    graph_off = meta_off + len(meta_bytes)
    weights_off = _align64(graph_off + len(graph_bytes))
    total_size = weights_off + len(model.weights_blob)

    flags = _flags_from_metadata(model.metadata)

    buf = bytearray(total_size)

    # header (64 bytes)
    buf[0:4] = MAGIC
    struct.pack_into("<H", buf, 4, FORMAT_VERSION)
    struct.pack_into("<H", buf, 6, flags)
    struct.pack_into("<Q", buf, 8, meta_off)
    struct.pack_into("<Q", buf, 16, len(meta_bytes))
    struct.pack_into("<Q", buf, 24, graph_off)
    struct.pack_into("<Q", buf, 32, len(graph_bytes))
    struct.pack_into("<Q", buf, 40, weights_off)
    struct.pack_into("<Q", buf, 48, len(model.weights_blob))
    struct.pack_into("<Q", buf, 56, total_size)

    # sections
    buf[meta_off : meta_off + len(meta_bytes)] = meta_bytes
    buf[graph_off : graph_off + len(graph_bytes)] = graph_bytes
    buf[weights_off : weights_off + len(model.weights_blob)] = model.weights_blob

    return bytes(buf)


def load_model(data: bytes) -> MpModel:
    """Deserialize .mpmodel binary bytes to an MpModel."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Data too short: {len(data)} < {HEADER_SIZE}")

    magic = data[0:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    version = struct.unpack_from("<H", data, 4)[0]
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    meta_off = struct.unpack_from("<Q", data, 8)[0]
    meta_size = struct.unpack_from("<Q", data, 16)[0]
    graph_off = struct.unpack_from("<Q", data, 24)[0]
    graph_size = struct.unpack_from("<Q", data, 32)[0]
    weights_off = struct.unpack_from("<Q", data, 40)[0]
    weights_size = struct.unpack_from("<Q", data, 48)[0]

    # decode metadata
    md_bytes = data[meta_off : meta_off + meta_size]
    metadata = _decode_metadata(md_bytes)

    # decode graph
    g_bytes = data[graph_off : graph_off + graph_size]
    nodes, tensor_names, tensor_shapes = _decode_graph(g_bytes)

    # weights
    weights_blob = bytes(data[weights_off : weights_off + weights_size])

    return MpModel(
        metadata=metadata,
        graph=nodes,
        tensor_names=tensor_names,
        tensor_shapes=tensor_shapes,
        weights_blob=weights_blob,
    )


def _decode_metadata(data: bytes) -> Metadata:
    off = 0

    name_len = struct.unpack_from("<H", data, off)[0]
    off += 2
    name = data[off : off + name_len].decode("utf-8")
    off += name_len

    num_in = struct.unpack_from("<H", data, off)[0]
    off += 2
    input_shapes = []
    for _ in range(num_in):
        ndims = struct.unpack_from("<H", data, off)[0]
        off += 2
        dims = []
        for _ in range(ndims):
            dims.append(struct.unpack_from("<i", data, off)[0])
            off += 4
        input_shapes.append(Shape(dims))

    num_out = struct.unpack_from("<H", data, off)[0]
    off += 2
    output_shapes = []
    for _ in range(num_out):
        ndims = struct.unpack_from("<H", data, off)[0]
        off += 2
        dims = []
        for _ in range(ndims):
            dims.append(struct.unpack_from("<i", data, off)[0])
            off += 4
        output_shapes.append(Shape(dims))

    qm_len = struct.unpack_from("<H", data, off)[0]
    off += 2
    quant_method = data[off : off + qm_len].decode("utf-8")
    off += qm_len

    quant_scale = struct.unpack_from("<f", data, off)[0]
    off += 4
    quant_zero = struct.unpack_from("<i", data, off)[0]
    off += 4

    ph_len = struct.unpack_from("<H", data, off)[0]
    off += 2
    platform_hints = data[off : off + ph_len].decode("utf-8")

    return Metadata(
        name=name,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        quant_method=quant_method,
        quant_scale=quant_scale,
        quant_zero=quant_zero,
        platform_hints=platform_hints,
    )


def _decode_graph(data: bytes) -> tuple:
    off = 0

    num_names = struct.unpack_from("<H", data, off)[0]
    off += 2
    tensor_names = []
    for _ in range(num_names):
        n_len = struct.unpack_from("<H", data, off)[0]
        off += 2
        tensor_names.append(data[off : off + n_len].decode("utf-8"))
        off += n_len

    num_nodes = struct.unpack_from("<H", data, off)[0]
    off += 2
    nodes = []
    for _ in range(num_nodes):
        op_type = OpType(struct.unpack_from("<H", data, off)[0])
        off += 2
        num_in = struct.unpack_from("<H", data, off)[0]
        off += 2
        input_indices = []
        for _ in range(num_in):
            input_indices.append(struct.unpack_from("<H", data, off)[0])
            off += 2
        num_out = struct.unpack_from("<H", data, off)[0]
        off += 2
        output_indices = []
        for _ in range(num_out):
            output_indices.append(struct.unpack_from("<H", data, off)[0])
            off += 2
        attr_len = struct.unpack_from("<H", data, off)[0]
        off += 2
        attrs = data[off : off + attr_len]
        off += attr_len
        nodes.append(OpNode(op_type, input_indices, output_indices, attrs))

    # tensor shapes (may be absent)
    tensor_shapes: Dict[str, Shape] = {}
    if off < len(data):
        try:
            num_shapes = struct.unpack_from("<H", data, off)[0]
            off += 2
            for _ in range(num_shapes):
                n_len = struct.unpack_from("<H", data, off)[0]
                off += 2
                name = data[off : off + n_len].decode("utf-8")
                off += n_len
                ndims = struct.unpack_from("<H", data, off)[0]
                off += 2
                dims = []
                for _ in range(ndims):
                    dims.append(struct.unpack_from("<i", data, off)[0])
                    off += 4
                tensor_shapes[name] = Shape(dims)
        except struct.error:
            pass

    return nodes, tensor_names, tensor_shapes
