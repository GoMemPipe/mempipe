# Examples

MemPipe ships with seven runnable examples demonstrating pipeline construction,
AI inference, real-time DSP, high-frequency trading, game simulation, and
WASM deployment. Every example operates with **zero heap allocations** on the
hot path.

| Example | Description | Key Features |
|---------|-------------|--------------|
| [ai_inference](#ai-inference) | Build, serialize, and run a toy MLP | Full model lifecycle, byte + tensor inference |
| [audio_dsp](#audio-dsp) | 128-sample audio processing pipeline | Sine gen, low-pass filter, gain, metering |
| [game_server](#game-server) | 60 Hz tick-based 2D physics simulation | Euler integration, circle collision, bouncing |
| [gpt2](#gpt2) | GPT-2 autoregressive text generation | Transformer inference, temperature, top-k |
| [hft_pipeline](#hft-pipeline) | High-frequency trading signal pipeline | SMA/EMA/z-score, mean-reversion orders |
| [mobilenet_v3](#mobilenet-v3) | MobileNetV3 image classification | Image preprocessing, top-N predictions |
| [wasm_demo](#wasm-demo) | Browser-based WASM + WebGPU demo | Worker.js, GPT-2 in browser, arena benchmarks |

---

## AI Inference

**Path**: `examples/ai_inference/`

Programmatically builds a toy MNIST-style MLP (784 → 128 → 10), serializes it
to `.mpmodel`, loads it back, and runs 100K zero-allocation inferences.

```bash
go run ./examples/ai_inference
```

**Demonstrates**:
- Full model lifecycle: build graph → `inference.SerializeModel()` → `inference.LoadModelFromBytes()` → `inference.NewEngine()`
- Byte-based inference via `engine.Infer(bytes)` (copies I/O)
- Zero-copy tensor inference via `engine.InferTensor()` with direct arena memory access
- Xavier weight initialization, manual graph construction with `OpDense`, `OpReLU`, `OpSoftmax`

---

## Audio DSP

**Path**: `examples/audio_dsp/`

Real-time audio processing pipeline: generates a 440 Hz sine wave, applies a
1-pole low-pass filter and gain stage, then meters the output — all in
128-sample blocks matching Web Audio's `AudioWorklet` quantum.

```bash
go run ./examples/audio_dsp
```

**Demonstrates**:
- `mempipe.NewPipeline()` + `mempipe.AddRegion[T]()` for typed arena-backed state
- `pipe.Cell()` with explicit input/output string dependencies for topological ordering
- `pipe.OnIteration()` for per-frame monitoring
- `RegionHandle[T].Get()` / `.Set()` for zero-alloc struct read/write
- Designed for WASM `AudioWorkletProcessor` integration

---

## Game Server

**Path**: `examples/game_server/`

Tick-based 2D game simulation with 8 entities arranged in a circle: Euler
physics integration, wall bouncing, O(n²) circle collision detection with
elastic response, and aggregate state tracking — running at 60 Hz for 600
ticks.

```bash
go run ./examples/game_server
```

**Demonstrates**:
- 4-cell pipeline: `input_processor → physics → collision → state_sync`
- `AddRegion[T]()` with game-domain structs (`InputState`, `Physics`, `CollisionResult`, `GameState`)
- `pipe.Cell()` with dependency chains for deterministic tick ordering
- `mempipe.WithWorkers(1)` pipeline option

---

## GPT-2

**Path**: `examples/gpt2/`

Runs a real GPT-2 transformer model from a converted `.mpmodel` file with
autoregressive text generation. Supports greedy argmax, temperature scaling,
and top-k sampling.

```bash
# Convert the model first
mempipe-convert onnx --transformer gpt2.onnx -o gpt2.mpmodel

# Greedy generation
go run ./examples/gpt2 -model gpt2.mpmodel -prompt "The quick brown fox" -n 50

# With temperature + top-k sampling
go run ./examples/gpt2 -model gpt2.mpmodel -prompt "Once upon a" -n 100 -temp 0.8 -topk 40

# With pre-tokenized BPE IDs
go run ./examples/gpt2 -model gpt2.mpmodel -tokens 464,2068,7586 -n 50
```

**CLI Flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to `.mpmodel` file |
| `-prompt` | `""` | Text prompt (uses simple whitespace tokenizer) |
| `-tokens` | `""` | Pre-tokenized comma-separated BPE token IDs |
| `-n` | `20` | Number of tokens to generate |
| `-temp` | `1.0` | Temperature for softmax sampling |
| `-topk` | `0` | Top-k filtering (0 = greedy argmax) |
| `-seed` | `42` | Random seed for reproducibility |
| `-v` | `false` | Verbose output |

**Demonstrates**:
- Autoregressive generation loop with sliding window context
- Softmax with temperature scaling, top-k filtering, categorical sampling
- `inference.LoadModel()` file-based model loading
- `engine.InferTensor()` zero-allocation hot path

---

## HFT Pipeline

**Path**: `examples/hft_pipeline/`

High-frequency trading signal pipeline: simulates a market feed with geometric
Brownian motion, computes SMA/EMA/σ/z-score/momentum via a ring buffer, and
generates mean-reversion buy/sell orders with P&L tracking across 10K ticks.

```bash
go run ./examples/hft_pipeline
```

**Demonstrates**:
- 3-cell pipeline: `market_data → signal_compute → order_gen`
- Ring-buffer pattern (`[windowSize]float32` on stack) for windowed time-series
- Mean-reversion strategy triggered by z-score threshold (±1.5)
- `pipe.OnIteration()` with tick counter for periodic reporting

---

## MobileNet V3

**Path**: `examples/mobilenet_v3/`

Runs MobileNetV3-Large (ImageNet 1000-class) image classification. Supports
PNG/JPEG/GIF input with center-crop + bilinear resize preprocessing, outputs
top-N predictions and saves an annotated result image.

```bash
# Convert the model first
mempipe-convert onnx MobileNet-v3.onnx -o mnv3.mpmodel

# Classify an image
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -image photo.png -top 5

# Quick smoke test with random input
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -v

# Benchmark 100 iterations
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -iter 100 -v
```

**CLI Flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to `.mpmodel` file |
| `-image` | `""` | Input image path (PNG/JPEG/GIF) |
| `-out` | `""` | Output annotated image path |
| `-random` | `false` | Use random input tensor |
| `-top` | `5` | Number of top predictions |
| `-iter` | `1` | Benchmark iterations |
| `-seed` | `42` | Random seed |
| `-no-save` | `false` | Skip saving result image |
| `-v` | `false` | Verbose output |

**Demonstrates**:
- Image preprocessing: center-crop → bilinear interpolation → NCHW → `[0,1]`
- `engine.InferTensor()` with multi-iteration benchmarking
- Softmax + top-k index extraction for classification
- TinyGo-compatible for embedded/microcontroller targets

---

## WASM Demo

**Path**: `examples/wasm_demo/`

Interactive browser-based single-page app with two demos:
1. **Demo 1**: Arena allocation, zero-copy memory, and operator benchmarks
2. **Demo 2**: GPT-2 text generation with WebGPU-accelerated MatMul

```bash
# Automated build + serve
cd examples/wasm_demo
chmod +x build.sh
./build.sh --serve 8080

# Manual build
GOOS=js GOARCH=wasm go build -o examples/wasm_demo/demo.wasm ./examples/wasm_demo/
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" examples/wasm_demo/
cd examples/wasm_demo && python3 -m http.server 8080
```

Open http://localhost:8080 (Chrome 113+ for WebGPU).

**Architecture** (3 layers):
```
Browser UI (index.html)
    ↓ postMessage
Web Worker (worker.js)
    ↓ Go() / syscall/js
Go WASM binary (main.go)
```

**Demonstrates**:
- `//go:build js && wasm` + `syscall/js` interop
- Go functions exported to JS via `js.Global().Set()`
- `inference.NewInferenceArena()` for direct WASM memory management
- `inference.GetOperator()` for standalone operator benchmarking
- `inference.IsWebGPUReady()` runtime capability detection
- Worker-based asynchronous generation loop

---

## Running All Examples

```bash
# Pipeline examples (no model files needed)
go run ./examples/ai_inference
go run ./examples/audio_dsp
go run ./examples/game_server
go run ./examples/hft_pipeline

# Inference examples (requires model conversion)
pip install mempipe-convert
mempipe-convert onnx gpt2.onnx -o gpt2.mpmodel --transformer
mempipe-convert onnx MobileNet-v3.onnx -o mnv3.mpmodel

go run ./examples/gpt2 -model gpt2.mpmodel -prompt "Hello world" -n 30
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -v
```

## See Also

- [PIPELINE_API.md](PIPELINE_API.md) — Pipeline construction API
- [BUILDER_API.md](BUILDER_API.md) — Builder API for dynamic pipelines
- [INFERENCE.md](INFERENCE.md) — Inference engine reference
- [WASM.md](WASM.md) — WASM deployment guide
