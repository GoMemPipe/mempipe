# Tools

MemPipe ships with two companion tools: **memlint** (Go static analyzer) and
**mempipe-convert** (Python model converter).

---

## memlint — Zero-Allocation Static Analyzer

A `go/analysis`-based linter that enforces MemPipe's zero-allocation invariants
on annotated functions. It integrates directly with `go vet`.

### Installation

```bash
go install github.com/GoMemPipe/mempipe/tools/memlint/cmd/memlint@latest
```

### Usage

```bash
# Via go vet (recommended)
go vet -vettool=$(which memlint) ./...

# Direct invocation
memlint ./...
```

### Annotations

Mark performance-critical functions with directive comments:

```go
//mem:hot
func processFrame(r *runtime.TypedRegion[Frame]) {
    // ML001, ML003 enforced here
}

//mem:nogc
func updateSensor(r *runtime.TypedRegion[Sensor]) {
    // ML002, ML004 enforced here
}

// Both can be combined:
//mem:hot
//mem:nogc
func criticalPath(r *runtime.TypedRegion[State]) {
    // All rules enforced
}
```

### Rules

| Rule | Scope | Description |
|------|-------|-------------|
| **ML001** | `//mem:hot` | No `interface{}`/`any` parameters, variables, or fields — prevents boxing allocations |
| **ML002** | `//mem:nogc` | No `make()`, `new()`, or `append()` — prevents heap allocations |
| **ML003** | `//mem:hot` | No `reflect` package calls — prevents runtime type introspection overhead |
| **ML004** | `//mem:nogc` | No closures or `defer` — prevents heap escape of captured variables |
| **ML005** | Global | No `ReadField()`/`WriteField()` anywhere — deprecated, use `TypedRegion[T].Get()/Set()` |

### Suppression

Use `//mem:allow()` to suppress a specific rule on the next line:

```go
//mem:nogc
func specialCase() {
    //mem:allow(ML002)
    buf := make([]byte, 16)  // intentional, won't trigger ML002
    _ = buf
}
```

### Example Diagnostics

```
engine.go:42:5: ML001: interface{}/any parameter in //mem:hot function
engine.go:58:9: ML002: make() in //mem:nogc function causes heap allocation
engine.go:73:3: ML003: reflect.TypeOf() in //mem:hot function
engine.go:81:5: ML004: closure in //mem:nogc function may cause heap escape
engine.go:85:5: ML004: defer in //mem:nogc function may cause heap escape
pipeline.go:22:3: ML005: ReadField() is deprecated; use TypedRegion[T].Get()/Set() instead
```

### Implementation

memlint is built on the standard `golang.org/x/tools/go/analysis` framework:

- **Pre-scan phase**: Collects `//mem:hot` and `//mem:nogc` directives from
  function doc comments
- **Walk phase**: AST inspection of each annotated function body
- **Global phase**: ML005 runs across all files regardless of annotations
- **Tests**: Uses `analysistest.Run` with testdata fixtures

---

## mempipe-convert — Model Converter

A Python tool that converts ONNX models (and optionally PyTorch/Keras via
ONNX export) to MemPipe's `.mpmodel` binary format.

### Installation

```bash
# Core (ONNX only)
pip install mempipe-convert

# With validation support
pip install "mempipe-convert[validation]"

# With PyTorch export
pip install "mempipe-convert[pytorch]"

# Everything
pip install "mempipe-convert[all]"

# Development
pip install "mempipe-convert[dev]"
```

**Requirements**: Python ≥ 3.9, numpy ≥ 1.24, onnx ≥ 1.14

### CLI Commands

#### `onnx` — Convert ONNX to .mpmodel

```bash
mempipe-convert onnx model.onnx -o model.mpmodel
mempipe-convert onnx model.onnx -o model.mpmodel --quantize int8
mempipe-convert onnx model.onnx -o model.mpmodel --name "my_model" --platform "wasm"

# Transformer models with pattern fusion (GELU, LayerNorm)
mempipe-convert onnx gpt2.onnx -o gpt2.mpmodel --transformer --seq-len 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | (required) | Output `.mpmodel` path |
| `--name` | filename stem | Model name in metadata |
| `--quantize` | none | `int8` or `fp16` |
| `--platform` | `""` | Platform hints string |
| `--transformer` | `false` | Enable transformer pattern fusion |
| `--seq-len` | `128` | Fixed sequence length for dynamic dims |

#### `inspect` — Inspect .mpmodel

```bash
mempipe-convert inspect model.mpmodel
```

Output:
```
Model: mobilenet_v3
  Inputs:  [[1, 3, 224, 224]]
  Outputs: [[1, 1000]]
  Ops:     75 (Conv2D, BatchNorm, ReLU, Add, AvgPool2D, Dense, ...)
  Tensors: 213
  Params:  5,483,032 (21,932,128 bytes)
  Est. FLOPs:       219,000,000
  Est. Activations: 12,845,056 bytes
```

Reports: input/output shapes, operator breakdown, tensor count, parameter
count, estimated FLOPs, estimated activation memory, quantization method, and
per-tensor shape details.

#### `quantize` — Post-Training Quantization

```bash
mempipe-convert quantize model.mpmodel -o model_int8.mpmodel --method dynamic
```

| Method | Description |
|--------|-------------|
| `dynamic` | Per-tensor symmetric INT8 (default) |
| `static` | Calibration-based INT8 |
| `fp16` | Float16 weight compression |

Reports input/output size and compression ratio.

#### `validate` — Validate Against ONNX Reference

```bash
mempipe-convert validate model.mpmodel --reference model.onnx --atol 1e-4
mempipe-convert validate model.mpmodel --reference model.onnx --input test.npy
```

| Flag | Default | Description |
|------|---------|-------------|
| `--reference` | (required) | Reference `.onnx` file |
| `--input` | random | Test input `.npy` file |
| `--atol` | `1e-4` | Absolute tolerance for comparison |

Requires `onnxruntime` (`pip install "mempipe-convert[validation]"`).

### ONNX Op Mapping

28 ONNX operations are mapped to `.mpmodel` op types:

| ONNX Op | mpmodel Op | Category |
|---------|------------|----------|
| Gemm | Dense | Core |
| MatMul | MatMul | Core |
| Add | Add | Core |
| Relu | ReLU | Core |
| Sigmoid | Sigmoid | Core |
| Softmax | Softmax | Core |
| Conv | Conv2D | Core |
| MaxPool | MaxPool2D | Core |
| AveragePool | AvgPool2D | Core |
| BatchNormalization | BatchNorm | Core |
| Flatten | Flatten | Core |
| Reshape | Reshape | Core |
| Concat | Concat | Core |
| Gelu | GELU | Transformer |
| LayerNormalization | LayerNorm | Transformer |
| Gather | Gather | Transformer |
| Mul | Mul | Transformer |
| Sub | Sub | Transformer |
| Transpose | Transpose | Transformer |
| Slice | Slice | Transformer |
| Tanh | Tanh | Transformer |
| Where | Where | Transformer |
| Split | Split | Transformer |
| Div | Div | Extended |
| Pow | Pow | Extended |
| IsNaN | IsNaN | Extended |
| And | And | Extended |

Unsupported ops are skipped with a warning. The `--transformer` flag enables
additional pattern fusion for GELU and LayerNorm subgraphs.

### Python API

```python
from mempipe_convert.converter import from_onnx
from mempipe_convert.inspect import inspect_model

# Convert
model = from_onnx(
    "model.onnx",
    "model.mpmodel",
    quantize="int8",
    name="my_model",
    platform_hints="wasm",
)

# Inspect
info = inspect_model("model.mpmodel")
print(info.num_ops, info.total_params, info.estimated_flops)

# Transformer conversion
from mempipe_convert.transformer import from_onnx_transformer
model = from_onnx_transformer(
    "gpt2.onnx",
    "gpt2.mpmodel",
    fuse_patterns=True,
    seq_len=128,
)
```

### Model Zoo

Pre-conversion scripts are available in `tools/mempipe-convert/zoo/`:

```bash
cd tools/mempipe-convert/zoo
python mobilenet_v3.py   # downloads + converts MobileNet-v3
python gpt2.py           # downloads + converts GPT-2
```

---

## See Also

- [MPMODEL_FORMAT.md](MPMODEL_FORMAT.md) — Binary format specification
- [INFERENCE.md](INFERENCE.md) — Inference engine documentation
- [CONDITIONAL_COMPILATION.md](CONDITIONAL_COMPILATION.md) — `//mem:hot` / `//mem:nogc` annotations
