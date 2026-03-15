# AI Inference Engine

MemPipe includes a **zero-allocation neural network inference engine** that loads
custom `.mpmodel` binary files, allocates all memory (weights, activations, I/O
buffers) in a **single contiguous arena**, and executes computation graphs with
**zero heap allocations** on the hot path.

The engine supports **33 operator types** spanning linear algebra, convolutions,
activations, normalization, transformer blocks, quantization, and logic —
enough to run production models from MLP classifiers to GPT-2 transformers.

## Quick Start

```go
import "github.com/GoMemPipe/mempipe/inference"

// Load model (from bytes — works on all platforms including WASM)
model, err := inference.LoadModelFromBytes(modelData)

// Create engine (allocates arena once, compiles execution graph)
engine, err := inference.NewEngine(model)

// --- Byte-based inference (copies in/out) ---
output, err := engine.Infer(inputBytes)

// --- Zero-copy tensor inference (recommended) ---
inputs := engine.InputTensors()
inputs[0].CopyFrom(imageBytes) // write input data into arena

outputs, err := engine.InferTensor() // zero-alloc forward pass
prob := outputs[0].AtF32(0, 7)       // read output directly from arena

// --- Batch inference ---
results, err := engine.InferBatch([][]byte{sample1, sample2, sample3})
```

## Model Format (.mpmodel)

The `.mpmodel` format is a compact binary format optimized for direct memory
mapping. See [MPMODEL_FORMAT.md](MPMODEL_FORMAT.md) for the byte-level specification.

```
┌──────────────────────────────────────┐
│            File Header (64B)          │
│  magic: "MPMD"  version: 1           │
│  flags: quantization bits             │
│  offsets: metadata / graph / weights  │
├──────────────────────────────────────┤
│        Metadata Section               │
│  model name, input/output shapes      │
│  quantization method / scale / zero   │
│  platform hints                       │
├──────────────────────────────────────┤
│          Graph Section                │
│  tensor name table (interned)         │
│  operator nodes (topological order)   │
│  tensor shape table (weights)         │
├──────────────────────────────────────┤
│        Weights Section (64B-aligned)  │
│  raw weight data (float32/int8/fp16)  │
│  directly memcpy-able into arena      │
└──────────────────────────────────────┘
```

### Converting Models

Use the Python `mempipe-convert` tool to convert from ONNX, PyTorch, or Keras:

```bash
# Install converter
cd tools/mempipe-convert && pip install -e ".[dev]"

# Convert an ONNX model
mempipe-convert onnx model.onnx -o model.mpmodel

# Convert with INT8 quantization
mempipe-convert onnx model.onnx -o model_int8.mpmodel --quantize int8

# Convert a transformer model (enables GELU/LayerNorm pattern fusion)
mempipe-convert onnx gpt2.onnx -o gpt2.mpmodel --transformer --seq-len 128

# Inspect a .mpmodel file
mempipe-convert inspect model.mpmodel

# Post-training quantization
mempipe-convert quantize model.mpmodel -o model_q.mpmodel --method dynamic

# Validate accuracy vs ONNX reference
mempipe-convert validate model.mpmodel model.onnx --atol 1e-5
```

See [TOOLS.md](TOOLS.md) for full documentation and `tools/mempipe-convert/zoo/`
for pre-built conversion scripts (MNIST, MobileNetV3, GPT-2).

## Operators

The inference engine supports **33 operator types**, organized by category:

### Linear Algebra

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 0 | `MatMul` | Matrix multiplication C = A × B | INT8 variant available, 3-tier hardware dispatch (generic / SIMD / WebGPU) |
| 12 | `Dense` | Fused MatMul + bias add | Y = X·W + B in one pass |
| 18 | `BatchedMatMul` | 3D+ batched matmul for multi-head attention | C[b] = A[b] × B[b], reuses platform `matMulF32` |

### Activations

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 2 | `ReLU` | Rectified linear unit | Y = max(0, X) |
| 3 | `Sigmoid` | Sigmoid activation | Y = 1/(1+exp(−X)) |
| 4 | `Softmax` | Row-wise softmax | Numerically stable (max subtraction) |
| 15 | `GELU` | Gaussian Error Linear Unit | Fast tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) |
| 23 | `Tanh` | Hyperbolic tangent | Element-wise math.Tanh |
| 31 | `HardSigmoid` | Hard sigmoid activation | Configurable α/β via `Attributable` attrs |
| 32 | `HardSwish` | Hard swish activation | x · HardSigmoid(x) |

### Normalization

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 8 | `BatchNorm` | Batch normalization | Supports NCHW (4D) and 2D inputs |
| 16 | `LayerNorm` | Layer normalization | Row-wise mean/variance → normalize → affine transform |

### Convolution & Pooling

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 5 | `Conv2D` | 2D convolution (NCHW) | 4 fast paths: 1×1 pointwise, depthwise, im2col+MatMul, general grouped. Supports stride, padding, dilation, groups via `Attributable`. |
| 6 | `MaxPool2D` | Max pooling (2×2, stride 2) | |
| 7 | `AvgPool2D` | Average pooling (2×2, stride 2) | |
| 30 | `GlobalAvgPool2D` | Global average pooling | Output: [N, C, 1, 1] |

### Element-wise Arithmetic

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 1 | `Add` | Element-wise addition | Broadcasting: same-size, scalar, channel-wise, smaller→larger |
| 19 | `Mul` | Element-wise multiplication | Same broadcasting variants as Add |
| 20 | `Sub` | Element-wise subtraction | Same broadcasting variants |
| 26 | `Div` | Element-wise division | Same broadcasting variants |
| 27 | `Pow` | Element-wise power | Special-cases for exponents 2.0 and 3.0 |

### Structural / Reshape

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 9 | `Flatten` | Reshape to 2D [N, C×H×W] | View operation |
| 10 | `Reshape` | Arbitrary reshape | Target shape from attrs; view or copy |
| 11 | `Concat` | Concatenation along axis | Shape-level (via inference) |
| 21 | `Transpose` | Dimension permutation | Permutation from attrs, uses fixed-size [8]int (no heap alloc) |
| 22 | `Slice` | Sub-range extraction | start/end/axes/steps from attrs |
| 25 | `Split` | Split along axis | Sizes from attrs |

### Transformer / Embedding

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 17 | `Gather` | Embedding lookup | Int32 indices → weight matrix rows via `copy()` |
| 24 | `Where` | Conditional element select | Float32 bitmask: `cond != 0 ? x : y` |

### Quantization

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 13 | `Quantize` | Float32 → Int8 | Symmetric quantization with scale |
| 14 | `Dequantize` | Int8 → Float32 | Reverse mapping with scale + zero point |

### Logic / Utility

| # | Operator | Description | Notes |
|---|----------|-------------|-------|
| 28 | `IsNaN` | Element-wise NaN check | Float32 bitmask output |
| 29 | `And` | Element-wise logical AND | Float32 bitmask AND |

### Operator Lifecycle Interfaces

Beyond the core `Operator` interface, two optional interfaces control operator
setup at compile time:

```go
// Core interface — all operators must implement this
type Operator interface {
    Execute(inputs, outputs []*Tensor) error
    OutputShape(inputShapes []Shape) ([]Shape, error)
}

// Optional: one-time initialization (e.g., allocate SIMD scratch buffers, compile GPU shaders)
type Initializable interface {
    Init(arena *InferenceArena) error
}

// Optional: parse per-node attributes (e.g., Conv2D stride/padding, Reshape target shape)
type Attributable interface {
    SetAttrs(attrs []byte) error
}
```

The `Engine` calls `Init(arena)` and `SetAttrs(blob)` exactly **once** during
`compile()` — never on the hot path. Operators that implement these interfaces:

| Operator | `Initializable` | `Attributable` |
|----------|:---:|:---:|
| `MatMul` (and `Dense`, `BatchedMatMul` via shared accel) | ✓ | — |
| `Conv2D` | — | ✓ (stride, padding, dilation, groups) |
| `HardSigmoid` | — | ✓ (alpha, beta) |
| `Transpose` | — | ✓ (permutation) |
| `Reshape` | — | ✓ (target shape) |
| `Slice` | — | ✓ (start, end, axes, steps) |
| `Split` | — | ✓ (axis, sizes) |

### Custom Operators

Register custom operators at runtime:

```go
type MyScaleOp struct{}

func (o *MyScaleOp) Execute(inputs, outputs []*Tensor) error {
    in, out := inputs[0].Float32s(), outputs[0].Float32s()
    for i, v := range in {
        out[i] = v * 2.0
    }
    return nil
}

func (o *MyScaleOp) OutputShape(in []Shape) ([]Shape, error) {
    return in, nil // pass-through shape
}

func init() {
    inference.RegisterOperator(100, func() Operator { return &MyScaleOp{} })
}
```

## Hardware-Accelerated MatMul

The `matMulF32` function — the hot-path kernel used by `MatMul`, `Dense`, and
`BatchedMatMul` — is split across three build-tagged files with platform-specific
implementations:

| File | Build Tag | Strategy | Performance |
|------|-----------|----------|-------------|
| `matmul_generic.go` | Fallback (all other platforms) | Portable i-p-j loop | Baseline |
| `matmul_simd.go` | `(linux\|darwin) && (amd64\|arm64)` | Cache-blocked 4×4 micro-kernel with BCE hints | ~5–10× faster |
| `matmul_wasm.go` | `js && wasm` | WebGPU compute shader (16×16 workgroups, shared memory tiling) | GPU-accelerated |

### SIMD Path Details

The native SIMD path uses a **three-level blocked algorithm** optimized for modern
CPU cache hierarchies:

- **Block sizes**: M=64, N=256, K=256 (tuned for L1/L2 cache)
- **Micro-kernel**: 4×4 register-tiled with **16 scalar accumulators**
- **K-loop**: 4× unrolled with explicit bounds-check elimination (BCE) hints
- **B-packing**: Column-major scratch buffer pre-allocated from arena in `Init()`
- **No assembly required**: Pure Go — relies on the compiler's SSA prove pass to
  emit SIMD instructions (AVX2 on x86-64, NEON on ARM64)

### WebGPU Path Details

The WASM WebGPU path offloads matrix multiplication to GPU compute shaders:

- **Shader**: WGSL compute shader with 16×16 workgroup size and shared memory tiling
- **Initialization**: GPU adapter, device, pipeline, and bind group layout created
  once in `Init()` (not on the hot path)
- **Dispatch**: Cached JS function invoked with raw WASM memory pointers (zero
  `js.ValueOf` allocations). Blocks on a Go channel until the GPU completes.
- **Event-loop safety**: Uses a `goAsync()` pattern that spawns a goroutine for
  blocking operations, preventing Go WASM event-loop deadlock
- **Fallback**: Automatically falls back to `matMulF32Generic()` if WebGPU is unavailable

```go
// Check WebGPU availability at startup
inference.EnsureWebGPU()              // async init (call from main)
ready := inference.IsWebGPUReady()    // poll readiness
```

### INT8 MatMul

A standalone INT8 matrix multiplication function is available for quantized workloads:

```go
// INT8 inputs with INT32 accumulation for precision
inference.MatMulInt8(a, b, c, scaleA, scaleB) // a: int8, b: int8, c: float32
```

## Tensors

Tensors are **views into the inference arena** — they never own memory:

```go
type Tensor struct {
    data    unsafe.Pointer // pointer into arena
    shape   []int          // e.g. [1, 784]
    strides []int          // element strides per dimension
    dtype   DType          // Float32, Int8, etc.
    name    string         // tensor name from model
    size    int            // total byte size
}
```

### Data Types

| DType | Size | Description |
|-------|------|-------------|
| `Float32` | 4B | Default for weights and activations |
| `Float16` | 2B | Compact storage, converted to F32 for compute |
| `Int8` | 1B | Quantized inference (4× smaller, faster matmul) |
| `Uint8` | 1B | Image data, raw bytes |
| `Int32` | 4B | Quantization zero-points, gather indices |

### Zero-Alloc Element Access

```go
// Read/write individual elements (variadic indices)
val := tensor.AtF32(row, col)     // read float32
tensor.SetF32(42.0, row, col)     // write float32

val8 := tensor.AtInt8(i)          // read int8
tensor.SetInt8(-3, i)             // write int8

val32 := tensor.AtInt32(i)        // read int32
tensor.SetInt32(7, i)             // write int32
```

### Zero-Copy Slice Views

```go
// Get zero-copy typed slices (backed by arena memory)
floats := tensor.Float32s()   // []float32 view
bytes  := tensor.Int8s()      // []int8 view
ints   := tensor.Int32s()     // []int32 view
raw    := tensor.Bytes()      // []byte view
```

### Bulk Operations

```go
// Copy data into/out of arena
tensor.CopyFrom(inputBytes)   // []byte → arena
tensor.CopyTo(outputBytes)    // arena → []byte

// Zero all elements
tensor.Zero()

// Reshape (view — same underlying data, different shape/strides)
reshaped := tensor.Reshape(3, 4)

// Slice along first dimension
sub := tensor.Slice(2, 5)     // rows [2, 5) — pointer arithmetic, no copy
```

### Tensor Metadata

```go
tensor.NumElements() // total element count
tensor.Rank()        // number of dimensions
tensor.ByteSize()    // total byte size
tensor.Shape()       // dimension slice
tensor.DType()       // data type enum
```

## Quantization

### INT8 Symmetric Quantization

Maps float32 values to int8 using `absmax / 127` scaling (zero point = 0):

```go
scale, err := inference.QuantizeSymmetric(floatTensor, int8Tensor)
err = inference.DequantizeInt8ToFloat32(int8Tensor, floatTensor, scale, 0)
```

### INT8 Asymmetric Quantization

Maps using `[min, max]` range to `[-128, 127]` with a non-zero zero point:

```go
scale, zeroPoint, err := inference.QuantizeAsymmetric(floatTensor, int8Tensor)
err = inference.DequantizeInt8ToFloat32(int8Tensor, floatTensor, scale, zeroPoint)
```

### FP16 Conversion

Manual IEEE 754 half-precision conversion (no external dependency):

```go
// Tensor-level conversion
err := inference.F32ToF16(floatTensor, fp16Tensor)
err = inference.F16ToF32(fp16Tensor, floatTensor)

// Individual value conversion
bits := inference.F32ToF16Bits(3.14)     // float32 → uint16
val  := inference.F16BitsToF32(bits)     // uint16 → float32
```

## Inference Arena

The `InferenceArena` is a **single contiguous byte slice** that holds all memory
for an inference session — weights, activations, I/O buffers, and scratch space:

```go
arena := NewInferenceArena(totalSize)     // THE one allocation

// Bump-allocate tensors (64-byte aligned)
tensor := arena.AllocTensor("conv1/output", Shape{1, 64, 32, 32}, Float32)

// Raw scratch allocation (e.g., for SIMD packing buffers)
scratch := arena.AllocRaw(65536)

// Load weights from .mpmodel blob
basePtr := arena.LoadWeights(weightsBlob)

// Reset for reuse (no re-allocation)
arena.Reset()
arena.Zero()

// Inspect usage
fmt.Printf("Used: %d / %d bytes\n", arena.UsedBytes(), arena.TotalBytes())
```

## Engine Lifecycle

### Compilation (7 steps)

```
NewEngine(model, opts...)
    │
    ├── 1. Validate model (magic, version, metadata)
    ├── 2. Infer tensor shapes (forward propagation through graph)
    ├── 3. Identify weight tensors in the weights blob
    ├── 4. Compute total arena size (weights + activations + I/O + extra)
    ├── 5. Allocate single arena: make([]byte, totalSize)  ← THE one allocation
    ├── 6. Load weights into arena + create Tensor objects
    ├── 7. Resolve operators from registry
    │      ├── Call SetAttrs() on Attributable operators
    │      └── Call Init(arena) on Initializable operators
    └── Return compiled Engine
```

### Inference APIs

```go
// Byte-based inference (copies data in/out)
output, err := engine.Infer(inputBytes)

// Zero-copy tensor inference (recommended for performance)
// Caller writes directly to input tensors, reads directly from output tensors
outputs, err := engine.InferTensor()

// Batch inference (sequential, reuses arena)
results, err := engine.InferBatch([][]byte{sample1, sample2, sample3})
```

### Configuration

```go
engine, err := inference.NewEngine(model,
    inference.WithExtraArena(1024*1024), // extra 1MB for workspace buffers
)
```

### Memory Inspection

```go
fmt.Printf("Arena: %d / %d bytes (%.1f%% used)\n",
    engine.ArenaUsed(), engine.ArenaTotal(),
    float64(engine.ArenaUsed())/float64(engine.ArenaTotal())*100)

// Access any named tensor (weights, activations, I/O)
t, ok := engine.Tensor("dense_1/weights")
if ok {
    fmt.Printf("Shape: %v, DType: %s, Bytes: %d\n",
        t.Shape(), t.DType(), t.ByteSize())
}

// List all I/O tensors
for i, t := range engine.InputTensors() {
    fmt.Printf("Input %d: %s %v\n", i, t.DType(), t.Shape())
}
for i, t := range engine.OutputTensors() {
    fmt.Printf("Output %d: %s %v\n", i, t.DType(), t.Shape())
}
```

## Shape Inference

The engine automatically infers all intermediate tensor shapes at compile time
by walking the graph topologically and calling `inferOpOutputShapes()` for each
operator. This covers all 33 operators including complex rules for:

- **Conv2D**: Output spatial dims from kernel size, stride, padding, and dilation
- **MatMul / BatchedMatMul**: Inner dimension contraction, batch broadcasting
- **Transpose**: Permutation of input dimensions
- **Reshape**: Target shape from encoded attributes (with -1 inference)
- **Concat**: Sum along concatenation axis
- **Split**: Divide along split axis with specified sizes
- **Slice**: Sub-range with start/end/axes/steps
- **Gather**: Embedding output from indices shape + embedding dim
- **Pooling**: Spatial downsampling (2×2 stride 2, or global)
- **Broadcasting**: Element-wise operators with shape broadcasting

## Benchmarks

Representative performance numbers (all report **0 allocs/op** unless noted):

### MatMul (SIMD path, native)

```
BenchmarkMatMul_4x4          ~150 ns/op     0 allocs/op
BenchmarkMatMul_16x16        ~500 ns/op     0 allocs/op
BenchmarkMatMul_64x64        ~3 µs/op       0 allocs/op
BenchmarkMatMul_128x128      ~20 µs/op      0 allocs/op
BenchmarkMatMul_256x256      ~120 µs/op     0 allocs/op
BenchmarkMatMul_512x512      ~800 µs/op     0 allocs/op
BenchmarkMatMul_1024x1024    ~6 ms/op       0 allocs/op
```

### Operators

```
BenchmarkDense_128x64        ~10 µs/op      0 allocs/op
BenchmarkReLU_1024           ~200 ns/op     0 allocs/op
BenchmarkReLU_65536          ~14 µs/op      0 allocs/op
BenchmarkSigmoid_1024        ~3 µs/op       0 allocs/op
BenchmarkSoftmax_128x10      ~5 µs/op       0 allocs/op
BenchmarkConv2D_32x32x3      ~1 ms/op       0 allocs/op
BenchmarkQuantize_1024       ~500 ns/op     0 allocs/op
BenchmarkMatMulInt8_64x64    ~2 µs/op       0 allocs/op
```

### Transformer Operators (GPT-2 scale)

```
BenchmarkGELU_98304          ~200 µs/op     0 allocs/op
BenchmarkLayerNorm_128x768   ~150 µs/op     0 allocs/op
BenchmarkGather_50257x768    ~30 µs/op      0 allocs/op   (seq 128)
BenchmarkBatchedMatMul_12x128x64  ~50 µs/op 0 allocs/op   (QK^T)
BenchmarkMul_98304           ~40 µs/op      0 allocs/op
BenchmarkTanh_98304          ~300 µs/op     0 allocs/op
```

### End-to-End

```
BenchmarkEngine_MLP_Infer       ~50 µs/op   0 allocs/op
BenchmarkEngine_MLP_InferTensor ~30 µs/op   0 allocs/op  (zero-copy)
BenchmarkArena_100Tensors       ~2 µs/op    1 allocs/op  (the ONE allocation)
BenchmarkTensor_Zero_4096       ~200 ns/op  0 allocs/op
```

## WASM Deployment

The inference engine compiles to WASM with zero code changes:

```bash
GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o mempipe.wasm ./platform/wasm/
```

From JavaScript:

```javascript
const mp = await MemPipeJS.MemPipe.load('mempipe.wasm');
const model = mp.loadModel(modelBytes);
const engine = mp.newEngine(model);

// High-level API (copies data)
const output = engine.infer(new Float32Array(input));

// Zero-copy API (direct WASM memory access — 4× faster)
const inputView = engine.inputView(0);
inputView.set(myData);
engine.inferZeroCopy();
const result = engine.outputView(0);
```

See [WASM.md](WASM.md) for the full WASM deployment guide including WebGPU
acceleration, Web Worker architecture, and Web Audio integration.

## Python Tooling

The `tools/mempipe-convert/` package provides a complete model conversion toolkit:

| Subcommand | Example | Description |
|------------|---------|-------------|
| `onnx` | `mempipe-convert onnx model.onnx -o m.mpmodel` | Convert ONNX → .mpmodel |
| `inspect` | `mempipe-convert inspect m.mpmodel` | Print metadata, ops, shapes, params |
| `quantize` | `mempipe-convert quantize m.mpmodel -o q.mpmodel --method dynamic` | Post-training INT8/FP16 quantization |
| `validate` | `mempipe-convert validate m.mpmodel model.onnx --atol 1e-5` | Accuracy comparison vs ONNX reference |

### Conversion Options

```bash
# INT8 or FP16 quantization during conversion
mempipe-convert onnx model.onnx -o model.mpmodel --quantize int8
mempipe-convert onnx model.onnx -o model.mpmodel --quantize fp16

# Transformer mode (enables GELU/LayerNorm pattern fusion)
mempipe-convert onnx gpt2.onnx -o gpt2.mpmodel --transformer --seq-len 128
```

### Python API

```python
from mempipe_convert import from_onnx, from_pytorch, from_keras, inspect_model

# Convert from various frameworks
model = from_onnx("model.onnx")
model = from_pytorch(torch_model, dummy_input)
model = from_keras(keras_model)

# Inspect model details
info = inspect_model("model.mpmodel")
print(f"Params: {info.total_params:,}, FLOPs: {info.estimated_flops:,}")
```

### Zoo Scripts

Pre-built conversion scripts for common models:

```bash
cd tools/mempipe-convert/zoo
python convert_mnist.py      # → mnist_mlp.mpmodel
python convert_mobilenet.py  # → mobilenet_v3.mpmodel
python convert_gpt2.py       # → gpt2.mpmodel
```

See [TOOLS.md](TOOLS.md) for comprehensive tooling documentation.
