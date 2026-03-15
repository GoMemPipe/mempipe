# `.mpmodel` Binary Format Specification

**Version**: 1  
**Magic**: `MPMD` (4 bytes)  
**Byte order**: Little-endian throughout

## Overview

`.mpmodel` is a compact binary format for storing neural network models
for MemPipe's zero-allocation inference engine. It is designed to be:

- **Directly loadable** — weights section is 64-byte aligned and can be
  memory-mapped or copied directly into inference arenas
- **Cross-platform** — works on native (x86/ARM), WASM, and embedded targets
- **Simple** — no external dependencies to parse (pure struct reads)

## File Layout

```
┌──────────────────────────────────┐  Offset 0
│         Header (64 bytes)        │
├──────────────────────────────────┤  Offset 64
│        Metadata Section          │
├──────────────────────────────────┤
│         Graph Section            │
├─── (padding to 64-byte align) ───┤
│        Weights Section           │
└──────────────────────────────────┘
```

## Header (64 bytes)

| Offset | Size | Type   | Field             | Description                        |
|--------|------|--------|-------------------|------------------------------------|
| 0      | 4    | char[] | `magic`           | `"MPMD"` (0x4D504D44)             |
| 4      | 2    | u16    | `version`         | Format version (currently 1)       |
| 6      | 2    | u16    | `flags`           | Bit flags (see below)              |
| 8      | 8    | u64    | `metadata_offset` | Byte offset of metadata section    |
| 16     | 8    | u64    | `metadata_size`   | Byte size of metadata section      |
| 24     | 8    | u64    | `graph_offset`    | Byte offset of graph section       |
| 32     | 8    | u64    | `graph_size`      | Byte size of graph section         |
| 40     | 8    | u64    | `weights_offset`  | Byte offset of weights section     |
| 48     | 8    | u64    | `weights_size`    | Byte size of weights section       |
| 56     | 8    | u64    | `total_size`      | Total file size in bytes           |

### Header Flags

| Bit | Name                 | Description                            |
|-----|----------------------|----------------------------------------|
| 0   | `FLAG_QUANTIZED_INT8`| Weights are INT8 quantized             |
| 1   | `FLAG_QUANTIZED_FP16`| Weights are FP16 compressed            |
| 2   | `FLAG_HAS_CALIBRATION`| Calibration data present (reserved)   |

## Metadata Section

Compact binary encoding (no JSON dependency). Fields are sequential with
length-prefixed strings.

| Field            | Encoding                                          |
|------------------|---------------------------------------------------|
| `name`           | `u16 length` + `length` UTF-8 bytes               |
| `input_shapes`   | `u16 count`, then for each: `u16 ndims`, `i32 × ndims` |
| `output_shapes`  | Same encoding as `input_shapes`                   |
| `quant_method`   | `u16 length` + `length` UTF-8 bytes               |
| `quant_scale`    | `f32`                                              |
| `quant_zero`     | `i32`                                              |
| `platform_hints` | `u16 length` + `length` UTF-8 bytes               |

### Quantization Method Strings

| Value              | Description                             |
|--------------------|-----------------------------------------|
| `""`               | No quantization (FP32 weights)          |
| `"int8_symmetric"` | Symmetric INT8 (scale only, zero = 0)   |
| `"int8_asymmetric"`| Asymmetric INT8 (scale + zero point)    |
| `"fp16"`           | FP16 weight compression                 |

## Graph Section

### Tensor Name Table

| Field         | Encoding                                         |
|---------------|--------------------------------------------------|
| `num_names`   | `u16`                                            |
| For each name | `u16 length` + `length` UTF-8 bytes              |

Tensor names are referenced by index throughout the graph. Convention:
- First `N` names are model inputs (matching `metadata.input_shapes` order)
- Last `M` names are model outputs (matching `metadata.output_shapes` order)
- Intermediate names are weights, biases, and activation tensors

### Op Node Table

| Field          | Encoding                                        |
|----------------|-------------------------------------------------|
| `num_nodes`    | `u16`                                           |
| For each node: |                                                 |
| `op_type`      | `u16` — see Op Types table below                |
| `num_inputs`   | `u16`                                           |
| `input_indices`| `u16 × num_inputs` — indices into tensor names  |
| `num_outputs`  | `u16`                                           |
| `output_indices`| `u16 × num_outputs` — indices into tensor names |
| `attrs_len`    | `u16`                                           |
| `attrs`        | `attrs_len` raw bytes (operator-specific)        |

### Tensor Shape Table

Appended after the op node table. Stores known shapes for weight/parameter
tensors needed for shape inference.

| Field          | Encoding                                        |
|----------------|-------------------------------------------------|
| `num_shapes`   | `u16`                                           |
| For each shape:|                                                 |
| `name`         | `u16 length` + `length` UTF-8 bytes             |
| `ndims`        | `u16`                                           |
| `dims`         | `i32 × ndims`                                   |

### Op Types

All 33 supported operator types with their `op_type` values:

#### Core Operators (0–14)

| Value | Name        | Inputs                      | Outputs | Notes                    |
|-------|-------------|-----------------------------|---------|--------------------------|
| 0     | MatMul      | A[M,K], B[K,N]             | C[M,N]  | C = A × B (INT8 variant available) |
| 1     | Add         | A, B                        | C       | Element-wise with broadcasting |
| 2     | ReLU        | X                           | Y       | Y = max(0, X)            |
| 3     | Sigmoid     | X                           | Y       | Y = 1/(1+exp(−X))        |
| 4     | Softmax     | X[..., N]                   | Y       | Row-wise, numerically stable |
| 5     | Conv2D      | X[N,C,H,W], K[Co,Ci,Kh,Kw] (+bias) | Y | See attrs encoding below |
| 6     | MaxPool2D   | X[N,C,H,W]                 | Y       | 2×2, stride 2            |
| 7     | AvgPool2D   | X[N,C,H,W]                 | Y       | 2×2, stride 2            |
| 8     | BatchNorm   | X, γ, β, μ, σ²             | Y       | NCHW or 2D               |
| 9     | Flatten     | X[N,C,H,W]                 | Y[N,C×H×W] | View reshape          |
| 10    | Reshape     | X                           | Y       | Target shape in attrs    |
| 11    | Concat      | A, B, ...                   | Y       | Along specified axis     |
| 12    | Dense       | X[M,K], W[K,N], B[N]       | Y[M,N]  | Fused MatMul + BiasAdd   |
| 13    | Quantize    | X (FP32)                    | Y (INT8)| Symmetric scale          |
| 14    | Dequantize  | X (INT8)                    | Y (FP32)| Scale + zero point       |

#### Transformer Operators (15–25)

| Value | Name          | Inputs              | Outputs | Notes                    |
|-------|---------------|----------------------|---------|--------------------------|
| 15    | GELU          | X                    | Y       | Fast tanh approximation  |
| 16    | LayerNorm     | X, γ, β             | Y       | Row-wise normalization   |
| 17    | Gather        | Weight[V,D], Idx     | Y       | Embedding lookup (int32 indices) |
| 18    | BatchedMatMul | A[B,M,K], B[B,K,N]  | C[B,M,N]| Multi-head attention matmul |
| 19    | Mul           | A, B                 | C       | Element-wise multiply with broadcasting |
| 20    | Sub           | A, B                 | C       | Element-wise subtract with broadcasting |
| 21    | Transpose     | X                    | Y       | Dimension permutation (see attrs) |
| 22    | Slice         | X                    | Y       | Sub-range extraction (see attrs) |
| 23    | Tanh          | X                    | Y       | Element-wise tanh        |
| 24    | Where         | Cond, X, Y           | Z       | Conditional select (float32 bitmask) |
| 25    | Split         | X                    | Y₁,Y₂,...| Split along axis (see attrs) |

#### Extended Operators (26–32)

| Value | Name            | Inputs    | Outputs | Notes                    |
|-------|-----------------|-----------|---------|--------------------------|
| 26    | Div             | A, B      | C       | Element-wise divide with broadcasting |
| 27    | Pow             | A, B      | C       | Element-wise power (fast paths for ²/³) |
| 28    | IsNaN           | X         | Y       | Float32 bitmask output   |
| 29    | And             | A, B      | C       | Float32 bitmask AND      |
| 30    | GlobalAvgPool2D | X[N,C,H,W]| Y[N,C,1,1] | Spatial average pooling |
| 31    | HardSigmoid     | X         | Y       | Configurable α/β (see attrs) |
| 32    | HardSwish       | X         | Y       | x · HardSigmoid(x)      |

### Operator Attribute Encoding

Some operators store configuration in the `attrs` blob. The encoding is
operator-specific raw bytes (little-endian):

#### Conv2D Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 4    | i32  | `stride_h` | Vertical stride |
| 4      | 4    | i32  | `stride_w` | Horizontal stride |
| 8      | 4    | i32  | `pad_h` | Vertical padding |
| 12     | 4    | i32  | `pad_w` | Horizontal padding |
| 16     | 4    | i32  | `dilation_h` | Vertical dilation |
| 20     | 4    | i32  | `dilation_w` | Horizontal dilation |
| 24     | 4    | i32  | `groups` | Convolution groups (1 = standard, C = depthwise) |

#### Reshape Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 2    | u16  | `ndims` | Number of target dimensions |
| 2      | 4×n  | i32[] | `dims` | Target shape (-1 for inferred) |

#### Transpose Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 2    | u16  | `ndims` | Number of permutation entries |
| 2      | 4×n  | i32[] | `perm` | Permutation indices |

#### Slice Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 2    | u16  | `naxes` | Number of slice axes |
| 2      | varies | i32[] | `starts` | Start indices (n values) |
| 2+4n   | varies | i32[] | `ends` | End indices (n values) |
| 2+8n   | varies | i32[] | `axes` | Axis indices (n values) |
| 2+12n  | varies | i32[] | `steps` | Step sizes (n values) |

#### Split Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 4    | i32  | `axis` | Split axis |
| 4      | 2    | u16  | `num_splits` | Number of output pieces |
| 6      | 4×n  | i32[] | `sizes` | Size of each split |

#### HardSigmoid Attributes

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0      | 4    | f32  | `alpha` | Slope coefficient (default 0.2) |
| 4      | 4    | f32  | `beta` | Offset coefficient (default 0.5) |

## Weights Section

- **Alignment**: Must start at a 64-byte aligned offset
- **Format**: Raw bytes, packed in tensor name table order
  - Only tensors that are initializers (weights/biases) are stored
  - Default dtype is FP32 (4 bytes per element)
  - If `FLAG_QUANTIZED_INT8` is set, weights are INT8 (1 byte per element)
  - If `FLAG_QUANTIZED_FP16` is set, weights are FP16 (2 bytes per element)
- **Arena loading**: The entire weights section can be `memcpy`'d into the
  inference arena in a single operation

## Versioning Rules

1. **Version 1** is the initial format described in this document
2. New fields may be appended to sections in minor revisions
3. Breaking changes require a version bump
4. Parsers must reject files with `version > supported_version`

## Memory Model

The format is designed for MemPipe's arena-based memory model:

```
┌──────────────────────────────────────────────┐
│              Single Arena Allocation          │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Weights  │  │Activations│  │ I/O Buffers│  │
│  │(from file)│  │(inferred) │  │(from shapes)│ │
│  └──────────┘  └──────────┘  └───────────┘  │
│                                              │
│  Total size = weights + activations + I/O    │
│  All 64-byte aligned                         │
└──────────────────────────────────────────────┘
```

1. **Compile time**: Shape inference computes all intermediate tensor sizes
2. **Arena sizing**: `weights_size + Σ(activation_sizes) + Σ(io_sizes)`, all 64-byte aligned
3. **Single allocation**: One `make([]byte, total)` — zero GC pressure
4. **Zero-copy inference**: `InferTensor()` returns views into arena memory

## Implementation References

| Language | File                              | Description              |
|----------|-----------------------------------|--------------------------|
| Go       | `inference/model.go`              | Loader, serializer       |
| Go       | `inference/tensor.go`             | Arena-backed tensors     |
| Go       | `inference/engine.go`             | Inference execution      |
| Python   | `tools/mempipe-convert/mempipe_convert/mpmodel.py` | Format codec |
| Python   | `tools/mempipe-convert/mempipe_convert/converter.py` | ONNX converter |
