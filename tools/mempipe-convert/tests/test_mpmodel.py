"""Tests for the .mpmodel serialization format (Python ↔ Go interop)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mempipe_convert.mpmodel import (
    MAGIC,
    FORMAT_VERSION,
    HEADER_SIZE,
    Metadata,
    MpModel,
    OpNode,
    OpType,
    Shape,
    load_model,
    serialize_model,
)


def _make_simple_model() -> MpModel:
    """Build a minimal MLP model for testing."""
    w1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    b1 = np.array([0.01, -0.01], dtype=np.float32)
    weights_blob = w1.tobytes() + b1.tobytes()

    return MpModel(
        metadata=Metadata(
            name="test-mlp",
            input_shapes=[Shape([1, 3])],
            output_shapes=[Shape([1, 2])],
        ),
        graph=[
            OpNode(OpType.Dense, [0, 1, 2], [3]),
            OpNode(OpType.Softmax, [3], [3]),
        ],
        tensor_names=["input", "w1", "b1", "output"],
        tensor_shapes={
            "w1": Shape([3, 2]),
            "b1": Shape([2]),
        },
        weights_blob=weights_blob,
    )


class TestRoundTrip:
    """Test serialize → load round-trip."""

    def test_basic_round_trip(self):
        model = _make_simple_model()
        data = serialize_model(model)
        model2 = load_model(data)

        assert model2.metadata.name == "test-mlp"
        assert len(model2.graph) == 2
        assert len(model2.tensor_names) == 4
        assert model2.tensor_names == ["input", "w1", "b1", "output"]
        assert model2.metadata.input_shapes[0].dims == [1, 3]
        assert model2.metadata.output_shapes[0].dims == [1, 2]
        assert len(model2.weights_blob) == len(model.weights_blob)

    def test_tensor_shapes_round_trip(self):
        model = _make_simple_model()
        data = serialize_model(model)
        model2 = load_model(data)

        assert "w1" in model2.tensor_shapes
        assert model2.tensor_shapes["w1"].dims == [3, 2]
        assert "b1" in model2.tensor_shapes
        assert model2.tensor_shapes["b1"].dims == [2]

    def test_header_magic(self):
        model = _make_simple_model()
        data = serialize_model(model)
        assert data[:4] == MAGIC

    def test_header_version(self):
        model = _make_simple_model()
        data = serialize_model(model)
        version = struct.unpack_from("<H", data, 4)[0]
        assert version == FORMAT_VERSION

    def test_graph_ops(self):
        model = _make_simple_model()
        data = serialize_model(model)
        model2 = load_model(data)

        assert model2.graph[0].op_type == OpType.Dense
        assert model2.graph[0].input_indices == [0, 1, 2]
        assert model2.graph[0].output_indices == [3]
        assert model2.graph[1].op_type == OpType.Softmax

    def test_weights_preserved(self):
        model = _make_simple_model()
        data = serialize_model(model)
        model2 = load_model(data)

        orig = np.frombuffer(model.weights_blob, dtype=np.float32)
        loaded = np.frombuffer(model2.weights_blob, dtype=np.float32)
        np.testing.assert_array_equal(orig, loaded)


class TestQuantization:
    """Test quantization metadata."""

    def test_int8_quant_flag(self):
        model = _make_simple_model()
        model.metadata.quant_method = "int8_symmetric"
        model.metadata.quant_scale = 0.05
        data = serialize_model(model)
        model2 = load_model(data)
        assert model2.metadata.quant_method == "int8_symmetric"

    def test_fp16_quant_flag(self):
        model = _make_simple_model()
        model.metadata.quant_method = "fp16"
        data = serialize_model(model)
        model2 = load_model(data)
        assert model2.metadata.quant_method == "fp16"


class TestValidation:
    """Test error handling."""

    def test_invalid_magic(self):
        data = bytearray(128)
        data[0:4] = b"BAAD"
        with pytest.raises(ValueError, match="Invalid magic"):
            load_model(bytes(data))

    def test_truncated_data(self):
        with pytest.raises(ValueError, match="too short"):
            load_model(b"\x01\x02\x03")

    def test_wrong_version(self):
        data = bytearray(128)
        data[0:4] = MAGIC
        struct.pack_into("<H", data, 4, 99)
        with pytest.raises(ValueError, match="Unsupported version"):
            load_model(bytes(data))


class TestShapeUtilities:
    """Test Shape helper methods."""

    def test_numel(self):
        assert Shape([2, 3, 4]).numel() == 24
        assert Shape([1]).numel() == 1
        assert Shape([]).numel() == 1  # scalar

    def test_model_sizes(self):
        model = _make_simple_model()
        assert model.input_size() == 3
        assert model.output_size() == 2


class TestOpType:
    """Test op type enum values match Go."""

    def test_enum_values(self):
        assert OpType.MatMul == 0
        assert OpType.Add == 1
        assert OpType.ReLU == 2
        assert OpType.Sigmoid == 3
        assert OpType.Softmax == 4
        assert OpType.Conv2D == 5
        assert OpType.Dense == 12
        assert OpType.Quantize == 13
        assert OpType.Dequantize == 14
