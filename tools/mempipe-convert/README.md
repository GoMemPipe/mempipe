# mempipe-convert

Convert ONNX, PyTorch, and Keras models to MemPipe's `.mpmodel` binary format for zero-allocation edge inference.

## Installation

```bash
uv pip install -e ".[all]"
```

Or minimal (ONNX only):

```bash
uv pip install -e .
```

## Usage

### CLI

```bash
# Convert ONNX model
mempipe-convert onnx model.onnx -o model.mpmodel

# Convert with INT8 quantization
mempipe-convert onnx model.onnx -o model.mpmodel --quantize int8

# Inspect a .mpmodel file
mempipe-convert inspect model.mpmodel

# Validate conversion accuracy
mempipe-convert validate model.mpmodel --reference model.onnx --input test_input.npy
```

### Python API

```python
from mempipe_convert import from_onnx, inspect_model

# Convert ONNX → .mpmodel
from_onnx("model.onnx", "model.mpmodel")

# Convert with quantization
from_onnx("model.onnx", "model.mpmodel", quantize="int8")

# Inspect
info = inspect_model("model.mpmodel")
print(info)
```

### PyTorch shortcut

```python
from mempipe_convert import from_pytorch
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
from_pytorch(model, torch.randn(1, 784), "mnist_mlp.mpmodel")
```

## Model Zoo

Pre-made conversion scripts in `zoo/`:

```bash
uv run zoo/convert_mnist.py
```

## Supported Operators

| ONNX Op | .mpmodel Op |
|---------|-------------|
| Gemm / MatMul | Dense / MatMul |
| Add | Add |
| Relu | ReLU |
| Sigmoid | Sigmoid |
| Softmax | Softmax |
| Conv | Conv2D |
| MaxPool | MaxPool2D |
| AveragePool | AvgPool2D |
| BatchNormalization | BatchNorm |
| Flatten | Flatten |
| Reshape | Reshape |
| Concat | Concat |
