"""Tests for the Transformer-specific converter and extended ops."""

from __future__ import annotations

import struct
import pytest

from mempipe_convert.mpmodel import OpType, Shape


class TestOpTypeEnum:
    """Verify all Transformer OpType enum values match Go side."""

    def test_gelu(self):
        assert OpType.GELU == 15

    def test_layernorm(self):
        assert OpType.LayerNorm == 16

    def test_gather(self):
        assert OpType.Gather == 17

    def test_batched_matmul(self):
        assert OpType.BatchedMatMul == 18

    def test_mul(self):
        assert OpType.Mul == 19

    def test_sub(self):
        assert OpType.Sub == 20

    def test_transpose(self):
        assert OpType.Transpose == 21

    def test_slice(self):
        assert OpType.Slice == 22

    def test_tanh(self):
        assert OpType.Tanh == 23

    def test_where(self):
        assert OpType.Where == 24

    def test_split(self):
        assert OpType.Split == 25


class TestTransformerConverter:
    """Test transformer.py helper functions."""

    def test_encode_transpose_attrs(self):
        from mempipe_convert.transformer import _encode_transpose_attrs

        # Create a mock node with perm attribute
        class MockAttr:
            def __init__(self, name, ints):
                self.name = name
                self.ints = ints

        class MockNode:
            def __init__(self, attrs):
                self.attribute = attrs

        node = MockNode([MockAttr("perm", [0, 2, 1, 3])])
        result = _encode_transpose_attrs(node)

        # Decode: [ndims=4, perm=[0,2,1,3]]
        ndims = struct.unpack_from("<H", result, 0)[0]
        assert ndims == 4
        perm = [struct.unpack_from("<H", result, 2 + i * 2)[0] for i in range(4)]
        assert perm == [0, 2, 1, 3]

    def test_encode_split_attrs(self):
        from mempipe_convert.transformer import _encode_split_attrs

        class MockAttr:
            def __init__(self, name, i):
                self.name = name
                self.i = i

        class MockNode:
            def __init__(self, attrs, outputs):
                self.attribute = attrs
                self.output = outputs

        node = MockNode([MockAttr("axis", 2)], ["o1", "o2", "o3"])
        result = _encode_split_attrs(node)

        axis, num = struct.unpack("<HH", result)
        assert axis == 2
        assert num == 3

    def test_skip_ops_set(self):
        from mempipe_convert.transformer import _SKIP_OPS

        assert "Shape" in _SKIP_OPS
        assert "Unsqueeze" in _SKIP_OPS
        assert "Cast" in _SKIP_OPS
        assert "Identity" in _SKIP_OPS
        assert "Dropout" in _SKIP_OPS

    def test_transformer_onnx_op_map(self):
        from mempipe_convert.transformer import _TRANSFORMER_ONNX_OP_MAP

        # Verify key Transformer ops are mapped
        assert _TRANSFORMER_ONNX_OP_MAP["Gelu"] == OpType.GELU
        assert _TRANSFORMER_ONNX_OP_MAP["LayerNormalization"] == OpType.LayerNorm
        assert _TRANSFORMER_ONNX_OP_MAP["Gather"] == OpType.Gather
        assert _TRANSFORMER_ONNX_OP_MAP["Mul"] == OpType.Mul
        assert _TRANSFORMER_ONNX_OP_MAP["Sub"] == OpType.Sub
        assert _TRANSFORMER_ONNX_OP_MAP["Transpose"] == OpType.Transpose
        assert _TRANSFORMER_ONNX_OP_MAP["Tanh"] == OpType.Tanh
        assert _TRANSFORMER_ONNX_OP_MAP["Where"] == OpType.Where
        assert _TRANSFORMER_ONNX_OP_MAP["Split"] == OpType.Split

        # Also includes original ops
        assert _TRANSFORMER_ONNX_OP_MAP["MatMul"] == OpType.MatMul
        assert _TRANSFORMER_ONNX_OP_MAP["Add"] == OpType.Add
        assert _TRANSFORMER_ONNX_OP_MAP["Softmax"] == OpType.Softmax

    def test_converter_onnx_op_map_extended(self):
        """Verify the base converter also has Transformer ops."""
        from mempipe_convert.converter import _ONNX_OP_MAP

        assert _ONNX_OP_MAP["Gelu"] == OpType.GELU
        assert _ONNX_OP_MAP["LayerNormalization"] == OpType.LayerNorm
        assert _ONNX_OP_MAP["Gather"] == OpType.Gather


class TestSliceAttrsEncoding:
    """Test Slice attributes encoding with initializer resolution."""

    def test_encode_slice_attrs_with_initializers(self):
        import numpy as np
        from mempipe_convert.transformer import _encode_slice_attrs

        class MockNode:
            def __init__(self, inputs):
                self.input = inputs

        inits = {
            "starts": np.array([0], dtype=np.float32),
            "ends": np.array([128], dtype=np.float32),
            "axes": np.array([1], dtype=np.float32),
        }
        node = MockNode(["data", "starts", "ends", "axes"])
        result = _encode_slice_attrs(node, inits)

        axis, start, end = struct.unpack("<Hii", result)
        assert axis == 1
        assert start == 0
        assert end == 128
