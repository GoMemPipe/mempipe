<div align="center">

# MemPipe

**Zero-GC, arena-backed pipeline & inference library for Go**

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0-brightgreen)](go.mod)
[![Platforms](https://img.shields.io/badge/Platforms-Native%20%7C%20WASM%20%7C%20Embedded-orange)](#platform-support)

*A single `make([]byte, N)` allocates the entire working set.
Every subsequent read/write is a raw pointer dereference — no GC, no
interface boxing, no hidden allocations.*

</div>

---

## Highlights

- **Zero allocations** on every hot path — verified 0 allocs/op by CI
- **33 neural-network operators** with hardware-accelerated MatMul (SIMD / WebGPU)
- **Three platforms** from one codebase: Native, WASM (144 KB), Embedded (TinyGo)
- **Deterministic** tick-based execution — same inputs → same outputs, always
- **Zero external dependencies** — pure Go, no CGo
- **Custom `.mpmodel`** binary format with INT8/FP16 quantization

## Use Cases

| Domain | What MemPipe gives you |
|--------|------------------------|
| **AI / ML Inference** | Arena-backed tensor engine — load `.mpmodel`, run zero-alloc forward passes with 33 operators including full transformer support |
| **Audio DSP** | WASM pipelines with zero-copy `Float32Array` views into linear memory; feed Web Audio `AudioWorkletProcessor` directly |
| **Game Servers** | Deterministic tick-based loops with cell dependency graphs and parallel execution |
| **High-Frequency Trading** | Sub-microsecond pipeline iterations, ring-buffer regions for time-series, zero GC pauses |
| **Edge / Embedded** | TinyGo-compatible builds with no-op mutexes and static arena allocation |

## Benchmarks

All benchmarks run with **0 allocs/op**. Measured on Go 1.22, Apple M2.

### Pipeline

| Benchmark | ns/op | allocs/op |
|-----------|------:|----------:|
| Pipeline typed region tick | ~15 | 0 |
| Multi-region (3 regions) tick | ~25 | 0 |
| Builder dynamic region tick | ~54 | 0 |
| Arena alloc (64 B) | ~5 | 0 |
| Region Get/Set (struct) | ~8 | 0 |

### Inference — MatMul

| Size | Generic (ns) | SIMD (ns) | Speedup |
|------|------------:|----------:|--------:|
| 64×64 | ~55,000 | ~20,000 | 2.8× |
| 128×128 | ~400,000 | ~120,000 | 3.3× |
| 256×256 | ~3,200,000 | ~800,000 | 4.0× |

### Inference — Operators

| Operator | Input | ns/op | allocs/op |
|----------|-------|------:|----------:|
| ReLU | 1×1000 | ~200 | 0 |
| Softmax | 1×1000 | ~3,500 | 0 |
| Dense (784→128) | 1×784 | ~55,000 | 0 |
| Conv2D (3×3) | 1×1×28×28 | ~120,000 | 0 |
| LayerNorm | 1×768 | ~2,500 | 0 |
| GELU | 1×768 | ~1,000 | 0 |

### Inference — End-to-End

| Model | Forward pass | allocs/op |
|-------|------------:|----------:|
| MLP (784→128→10) | ~110 µs | 0 |
| Transformer block (768-dim) | ~12 ms | 0 |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Pipeline                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │  Cell A  │──►│  Cell B  │──►│  Cell C  │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                    │
│  ┌────▼──────────────▼──────────────▼──────────────┐    │
│  │                RegionArena                       │    │
│  │  ┌────────┐  ┌────────┐  ┌─────────┐           │    │
│  │  │ Rgn A  │  │ Rgn B  │  │  Rgn C  │   ...     │    │
│  │  └────────┘  └────────┘  └─────────┘           │    │
│  │  ◄──────── single []byte allocation ──────────► │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Inference Engine                     │    │
│  │  InferenceArena ── 33 ops ── CompiledGraph       │    │
│  │  SIMD MatMul ◄──── build tags ────► WebGPU       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Modules  │  │ Scheduler│  │  Clock   │              │
│  │ audio,io │  │ topo-sort│  │ tick-based│              │
│  │ http,sys │  │ parallel │  │ determin. │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Regions** — typed views into contiguous arena memory via struct tags
**Cells** — pure Go closures that read/write regions (zero-alloc)
**Scheduler** — topological ordering with optional parallel execution
**Inference** — compiled operator graph with hardware-dispatched MatMul

## Install

```bash
go get github.com/GoMemPipe/mempipe
```

**Requires**: Go 1.22+  
**Dependencies**: 0  
**Platforms**: Linux, macOS, Windows, WASM, Embedded (TinyGo)

## Quick Start

### Pipeline

```go
package main

import "github.com/GoMemPipe/mempipe"

type Sensor struct {
    Temp     float32 `mempipe:"field:temp"`
    Humidity float32 `mempipe:"field:humidity"`
    Count    uint32  `mempipe:"field:count"`
}

func main() {
    pipe := mempipe.NewPipeline()
    sensor := mempipe.AddRegion[Sensor](pipe, "sensor")

    pipe.SimpleCell("process", func() {
        s := sensor.Get()   // zero-alloc read from arena
        s.Temp += 0.1
        s.Count++
        sensor.Set(s)       // zero-alloc write to arena
    })

    pipe.Run(1_000_000)     // 1M ticks, zero GC pauses
}
```

### AI Inference

```go
import "github.com/GoMemPipe/mempipe/inference"

// Load .mpmodel from disk
model, _ := inference.LoadModel("model.mpmodel")
engine, _ := inference.NewEngine(model)

// Write input directly into the inference arena
input := engine.InputTensors()[0]
copy(input.Float32s(), imageData)

// Zero-allocation forward pass
outputs, _ := engine.InferTensor()
probs := outputs[0].Float32s()  // softmax probabilities
```

### WASM (Browser)

```html
<script src="wasm_exec.js"></script>
<script src="mempipe.js"></script>
<script>
  const mp = await MemPipeJS.MemPipe.load('mempipe.wasm');
  const model = mp.loadModel(modelBytes);
  const engine = mp.newEngine(model);
  const output = engine.infer(new Float32Array(input));
</script>
```

Build: `GOOS=js GOARCH=wasm go build -o app.wasm` → **144 KB** binary.

## Inference Engine

33 operators across three categories:

| Category | Operators |
|----------|-----------|
| **Core** (0–14) | MatMul, Add, ReLU, Sigmoid, Softmax, Conv2D, MaxPool2D, AvgPool2D, BatchNorm, Flatten, Reshape, Concat, Dense, Quantize, Dequantize |
| **Transformer** (15–25) | GELU, LayerNorm, Gather, BatchedMatMul, Mul, Sub, Transpose, Slice, Tanh, Where, Split |
| **Extended** (26–32) | Div, Pow, IsNaN, And, GlobalAvgPool2D, HardSigmoid, HardSwish |

**Hardware-accelerated MatMul** — three implementations selected by build tags:

| Variant | File | Dispatch |
|---------|------|----------|
| Generic | `matmul_generic.go` | i-p-j loop, all platforms |
| SIMD | `matmul_simd.go` | 4×4 micro-kernel, `!wasm && !embedded` |
| WebGPU | `matmul_wasm.go` | WGSL compute shader, `js && wasm` |

**Model format**: Custom `.mpmodel` binary — convert from ONNX/PyTorch/Keras
with the `mempipe-convert` Python tool. Supports INT8 and FP16 quantization.

## Examples

| Example | Description | Run |
|---------|-------------|-----|
| [ai_inference](examples/ai_inference/) | Build + run a toy MLP end-to-end | `go run ./examples/ai_inference` |
| [audio_dsp](examples/audio_dsp/) | 128-sample audio pipeline (sine → filter → gain) | `go run ./examples/audio_dsp` |
| [game_server](examples/game_server/) | 60 Hz 2D physics with collision detection | `go run ./examples/game_server` |
| [gpt2](examples/gpt2/) | GPT-2 text generation (greedy / top-k) | `go run ./examples/gpt2 -model gpt2.mpmodel -prompt "Hello"` |
| [hft_pipeline](examples/hft_pipeline/) | Trading signals: SMA/EMA/z-score → orders | `go run ./examples/hft_pipeline` |
| [mobilenet_v3](examples/mobilenet_v3/) | MobileNetV3 image classification (top-N) | `go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random` |
| [wasm_demo](examples/wasm_demo/) | Browser app: arena benchmarks + GPT-2 in WebGPU | `cd examples/wasm_demo && ./build.sh --serve 8080` |

## Tools

### memlint — Zero-Allocation Static Analyzer

```bash
go install github.com/GoMemPipe/mempipe/tools/memlint/cmd/memlint@latest
go vet -vettool=$(which memlint) ./...
```

5 rules enforcing zero-alloc on `//mem:hot` and `//mem:nogc` annotated functions:

| Rule | Detects |
|------|---------|
| ML001 | `interface{}`/`any` in hot functions |
| ML002 | `make`/`new`/`append` in nogc functions |
| ML003 | `reflect` in hot functions |
| ML004 | Closures/`defer` in nogc functions |
| ML005 | Deprecated `ReadField`/`WriteField` calls |

### mempipe-convert — Model Converter

```bash
pip install mempipe-convert

# Convert ONNX → .mpmodel
mempipe-convert onnx model.onnx -o model.mpmodel

# Transformer with pattern fusion
mempipe-convert onnx gpt2.onnx -o gpt2.mpmodel --transformer

# Quantize
mempipe-convert quantize model.mpmodel -o model_int8.mpmodel --method dynamic

# Inspect
mempipe-convert inspect model.mpmodel
```

Supports 28 ONNX operator mappings, INT8/FP16 quantization, and ONNX reference
validation.

## Platform Support

| Platform | Build Command | MatMul | Binary Size |
|----------|--------------|--------|-------------|
| **Linux / macOS / Windows** | `go build ./...` | SIMD 4×4 | Standard |
| **WASM (Browser)** | `GOOS=js GOARCH=wasm go build` | WebGPU / WASM fallback | ~144 KB |
| **Embedded (TinyGo)** | `tinygo build -tags embedded` | Generic | Minimal |

## Module System

Built-in modules register via `init()` and integrate with the scheduler lifecycle:

| Module | Package | Description |
|--------|---------|-------------|
| **Audio** | `module/audio` | Zero-alloc DSP: sine/noise generators, IIR filters, gain, mixing |
| **HTTP** | `module/http` | Platform-aware client: `net/http` / `fetch` / embedded stub |
| **I/O** | `module/io` | Memory-native stdout/stderr via `MemoryPipe` ring buffers |
| **Math** | `module/math` | 18 math functions (trig, exp, rounding, GCD/LCM) |
| **Sys** | `module/sys` | Runtime info: version, ticks, uptime, arena stats |
| **Time** | `module/time` | Deterministic tick-based clock, OS-independent |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Memory model, arena design, scheduling, platform layers |
| [Pipeline API](docs/PIPELINE_API.md) | `NewPipeline`, `AddRegion[T]`, cells, struct tags, lifecycle |
| [Builder API](docs/BUILDER_API.md) | Dynamic pipeline construction with `builder.NewPipeline` |
| [Inference](docs/INFERENCE.md) | 33 operators, engine lifecycle, benchmarks, custom ops |
| [Modules](docs/MODULES.md) | Module system, registry, all 6 built-in modules |
| [Tools](docs/TOOLS.md) | memlint static analyzer, mempipe-convert CLI |
| [Examples](docs/EXAMPLES.md) | Guide to all 7 runnable examples |
| [.mpmodel Format](docs/MPMODEL_FORMAT.md) | Binary format specification, op types, attributes |
| [WASM](docs/WASM.md) | WASM build, WebGPU MatMul, JS interop, browser demo |
| [Conditional Compilation](docs/CONDITIONAL_COMPILATION.md) | Build tags, platform matrix, `goAsync` pattern |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests: `go test ./...`
4. Run the linter: `go vet -vettool=$(which memlint) ./...`
5. Ensure **0 allocs/op** on all `//mem:nogc` benchmarks
6. Submit a pull request

## License

[MIT](LICENSE) — Copyright GoMemPipe (Hashem Zargari)

