"""mempipe-convert: Convert ML models to MemPipe .mpmodel format."""

from mempipe_convert.converter import from_onnx
from mempipe_convert.inspect import inspect_model
from mempipe_convert.mpmodel import MpModel, MpModelHeader, OpType, Shape
from mempipe_convert.transformer import from_onnx_transformer

__all__ = [
    "from_onnx",
    "from_onnx_transformer",
    "inspect_model",
    "MpModel",
    "MpModelHeader",
    "OpType",
    "Shape",
]

__version__ = "0.1.0"


def from_pytorch(model, example_input, output_path: str, **kwargs):
    """Convert a PyTorch model to .mpmodel via ONNX export.

    Requires the 'pytorch' extra: ``uv pip install -e ".[pytorch]"``
    """
    import tempfile

    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch required: uv pip install -e '.[pytorch]'") from e

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        torch.onnx.export(
            model,
            example_input,
            tmp.name,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        from_onnx(tmp.name, output_path, **kwargs)


def from_keras(model, output_path: str, **kwargs):
    """Convert a Keras model to .mpmodel via tf2onnx → ONNX → .mpmodel.

    Requires the 'keras' extra: ``uv pip install -e ".[keras]"``
    """
    import tempfile

    try:
        import tf2onnx
    except ImportError as e:
        raise ImportError("tf2onnx required: uv pip install -e '.[keras]'") from e

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        import tensorflow as tf

        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=tmp.name)
        from_onnx(tmp.name, output_path, **kwargs)
