# WASM Deployment Guide

MemPipe compiles to WebAssembly for browser and Node.js deployment.  The
WASM binary is **144KB** (stripped) and provides zero-copy access to
arena-backed tensor memory from JavaScript.

## Building

```bash
# Standard Go WASM build
GOOS=js GOARCH=wasm go build -ldflags="-s -w" \
    -o dist/mempipe.wasm ./platform/wasm/

# Copy Go WASM runtime
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" dist/

# Copy JS wrapper
cp platform/wasm/js/mempipe.js dist/
cp platform/wasm/js/mempipe.d.ts dist/

# Or use the Makefile:
make wasm
```

### Size Optimization

The default build uses `-ldflags="-s -w"` to strip debug info.  Additional
size reductions:

| Technique | Effect |
|-----------|--------|
| `-ldflags="-s -w"` | Strip symbols + DWARF (~30% smaller) |
| `//go:build` exclusions | Exclude unused modules (HTTP, sys) |
| TinyGo | Much smaller binary: `make tinygo-wasm` |

## Browser Integration

### Minimal HTML Setup

```html
<!DOCTYPE html>
<html>
<head>
    <script src="wasm_exec.js"></script>
    <script src="mempipe.js"></script>
</head>
<body>
<script>
async function main() {
    const mp = await MemPipeJS.MemPipe.load('mempipe.wasm');
    console.log('MemPipe', mp.version, 'on', mp.platform);

    // Load a .mpmodel file
    const resp = await fetch('model.mpmodel');
    const modelBytes = new Uint8Array(await resp.arrayBuffer());
    const model = mp.loadModel(modelBytes);

    // Create inference engine
    const engine = mp.newEngine(model);

    // Run inference
    const input = new Float32Array(784); // e.g., 28x28 image
    input.fill(0.5);
    const output = engine.infer(input);
    console.log('Predictions:', output);

    // Clean up
    engine.free();
    model.free();
}
main();
</script>
</body>
</html>
```

### ES Module Import

```javascript
import { MemPipe } from './mempipe.js';

const mp = await MemPipe.load('mempipe.wasm');
```

### Node.js

```javascript
require('./wasm_exec.js');
const { MemPipe } = require('./mempipe.js');
const fs = require('fs');

const wasmBytes = fs.readFileSync('mempipe.wasm');
const mp = await MemPipe.load(wasmBytes);
```

## Zero-Copy Memory Access

The key advantage of WASM deployment is **zero-copy data sharing** between
JavaScript and the inference arena.  Instead of copying Float32Arrays back
and forth, JavaScript can create typed array views directly over WASM linear
memory.

### How It Works

```
WASM Linear Memory (one contiguous ArrayBuffer)
┌────────────────────────────────────────────────────┐
│  ...  │  Arena  │  Input Tensor  │  Output Tensor  │
│       │ weights │  ◄─── JS view ─│── JS view ──►   │
└────────────────────────────────────────────────────┘
         ▲                ▲                ▲
         │                │                │
    engine.arenaInfo()  inputView(0)    outputView(0)
```

### Using Zero-Copy

```javascript
const engine = mp.newEngine(model);

// Get a Float32Array that directly points into WASM memory
const inputView = engine.inputView(0);  // no copy!
const shape = engine.inputShape(0);     // e.g., [1, 784]

// Write input data directly into WASM memory
inputView.set(myImageData);  // one memcpy, no JS→Go marshalling

// Run inference (operates on data already in arena)
engine.inferZeroCopy();

// Read output directly from WASM memory
const outputView = engine.outputView(0);  // no copy!
const prediction = outputView[7];  // class 7 probability
```

### Performance Comparison

| Method | Input Copy | Compute | Output Copy | Total |
|--------|-----------|---------|-------------|-------|
| `engine.infer(data)` | JS→Go marshal | InferTensor | Go→JS marshal | ~200µs |
| Zero-copy | memcpy into view | InferTensor | read from view | ~50µs |

## Web Audio Integration

MemPipe's zero-copy WASM memory is ideal for real-time audio processing
with the Web Audio API's `AudioWorkletProcessor`.

### Audio Worklet Example

```javascript
// audio_worklet.js — runs in AudioWorklet thread
class MemPipeProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Engine and views set up via message from main thread
        this.inputView = null;
        this.outputView = null;
        this.port.onmessage = (e) => {
            if (e.data.type === 'init') {
                this.inputView = e.data.inputView;
                this.outputView = e.data.outputView;
            }
        };
    }

    process(inputs, outputs) {
        const input = inputs[0][0];   // 128 samples
        const output = outputs[0][0];

        if (this.inputView && input) {
            // Write audio samples directly into WASM arena
            this.inputView.set(input);

            // Run DSP pipeline (zero-copy)
            mempipe.inferZeroCopy(this.engineHandle);

            // Read processed samples from WASM arena
            output.set(this.outputView.subarray(0, 128));
        }

        return true;
    }
}

registerProcessor('mempipe-processor', MemPipeProcessor);
```

### SharedArrayBuffer

For the best performance with Web Audio, use `SharedArrayBuffer` so the
AudioWorklet thread can access WASM memory without copying:

```javascript
// Main thread
const mp = await MemPipeJS.MemPipe.load('mempipe.wasm');
const { ptr, size } = engine.arenaInfo();

// SharedArrayBuffer requires COOP/COEP headers:
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
```

**Required HTTP headers** for SharedArrayBuffer:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## API Reference

### MemPipe

| Method | Returns | Description |
|--------|---------|-------------|
| `MemPipe.load(source)` | `Promise<MemPipe>` | Load WASM module from URL or bytes |
| `.version` | `string` | Library version |
| `.platform` | `string` | `"wasm"` |
| `.loadModel(bytes)` | `Model` | Parse .mpmodel from Uint8Array |
| `.newEngine(model)` | `InferenceEngine` | Create inference engine |

### Model

| Method | Description |
|--------|-------------|
| `.free()` | Release model memory |

### InferenceEngine

| Method | Returns | Description |
|--------|---------|-------------|
| `.infer(input)` | `Float32Array` | Copy-based inference |
| `.inferZeroCopy()` | `void` | Zero-copy inference (use views) |
| `.inputView(i)` | `Float32Array` | Direct WASM memory view for input i |
| `.outputView(i)` | `Float32Array` | Direct WASM memory view for output i |
| `.inputShape(i)` | `number[]` | Input tensor dimensions |
| `.outputShape(i)` | `number[]` | Output tensor dimensions |
| `.arenaInfo()` | `{ptr, size}` | Arena memory location and size |
| `.free()` | `void` | Release engine |

## Browser Compatibility

| Browser | WASM | SharedArrayBuffer | AudioWorklet | WebGPU |
|---------|------|-------------------|--------------|--------|
| Chrome 90+ | ✅ | ✅ | ✅ | ✅ (113+) |
| Firefox 89+ | ✅ | ✅ | ✅ | 🔜 (Nightly) |
| Safari 15+ | ✅ | ✅ | ✅ | ✅ (18+) |
| Edge 90+ | ✅ | ✅ | ✅ | ✅ (113+) |
| Node.js 16+ | ✅ | ✅ | N/A | N/A |

## WebGPU-Accelerated MatMul

When running in browsers with WebGPU support, MemPipe automatically offloads
matrix multiplication to GPU compute shaders for significant performance gains.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Go WASM Runtime                     │
│                                                      │
│  matMulF32()                                         │
│    │                                                 │
│    ├── Is WebGPU ready? ──No──► matMulF32Generic()   │
│    │                                                 │
│    └── Yes ──► Invoke cached JS dispatch function    │
│                    │                                 │
│                    ▼                                 │
│  ┌─────── JavaScript Async Pipeline ───────────┐    │
│  │  1. writeBuffer(A, B → GPU)                  │    │
│  │  2. dispatchCompute(workgroups)              │    │
│  │  3. mapAsync(C ← GPU)                       │    │
│  │  4. copy C back to WASM memory              │    │
│  │  5. callback → Go doneChan                   │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ◄── Go goroutine blocks on <-doneChan ──►           │
└─────────────────────────────────────────────────────┘
```

### WGSL Compute Shader

The GPU shader uses **16×16 workgroups** with shared-memory tiling for
good occupancy and memory coalescing:

```wgsl
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

var<workgroup> tileA : array<array<f32, 16>, 16>;
var<workgroup> tileB : array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3u, ...) {
    // Tiled matrix multiplication with shared memory
    // ...
}
```

### Initialization

WebGPU initialization is **asynchronous** and must complete before inference:

```go
// In main() — before select{}
inference.EnsureWebGPU()

// Check readiness (optional)
if inference.IsWebGPUReady() {
    fmt.Println("WebGPU MatMul active")
}
```

The initialization sequence:
1. Request GPU adapter with maximum buffer size limits
2. Request GPU device
3. Compile WGSL compute shader into pipeline
4. Create bind group layout and persistent `js.Func` callback
5. Install JS-side async dispatch orchestrator via `eval()`

All handles are created **once** and cached — the hot path only invokes a
single cached JS function with raw integer pointers (zero `js.ValueOf` allocations).

### Fallback

If WebGPU is unavailable (older browser, Node.js, WebGPU API not exposed),
the engine automatically falls back to `matMulF32Generic()` — the portable
pure-Go implementation. No code changes needed.

## WASM Demo Application

The `examples/wasm_demo/` directory contains a complete browser-based demo
with two interactive sections:

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Browser (Main Thread)               │
│  ┌────────────────────────────────────────┐           │
│  │           index.html (UI)              │           │
│  │  - Demo 1: Arena / Operators           │           │
│  │  - Demo 2: GPT-2 Text Generation      │           │
│  │  - callWorker(method, args) RPC        │           │
│  └────────────┬───────────────────────────┘           │
│               │ postMessage                            │
│  ┌────────────▼───────────────────────────┐           │
│  │          worker.js (Web Worker)         │           │
│  │  - Loads wasm_exec.js + demo.wasm       │           │
│  │  - Instantiates Go WASM runtime         │           │
│  │  - Polls for mempipeDemo global         │           │
│  │  - RPC dispatch: method → Go function   │           │
│  │  - Streams progress for generation      │           │
│  └────────────┬───────────────────────────┘           │
│               │ syscall/js                             │
│  ┌────────────▼───────────────────────────┐           │
│  │         main.go (Go WASM)               │           │
│  │  - goAsync() Promise wrapper            │           │
│  │  - Demo 1: arena, operators, zero-copy  │           │
│  │  - Demo 2: GPT-2 load, generate, bench  │           │
│  │  - EnsureWebGPU() at startup            │           │
│  └─────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────┘
```

### Building the Demo

```bash
cd examples/wasm_demo
./build.sh        # compile demo.wasm + copy assets
./build.sh serve  # compile + start local HTTP server
```

### `goAsync()` Pattern

All Go functions exported to JavaScript use the `goAsync()` wrapper, which
converts blocking Go operations into JavaScript Promises:

```go
func goAsync(fn func() (any, error)) js.Func {
    return js.FuncOf(func(this js.Value, args []js.Value) any {
        handler := js.FuncOf(func(_ js.Value, promiseArgs []js.Value) any {
            resolve, reject := promiseArgs[0], promiseArgs[1]
            go func() {
                result, err := fn()
                if err != nil { reject.Invoke(err.Error()) }
                else          { resolve.Invoke(result) }
            }()
            return nil
        })
        return js.Global().Get("Promise").New(handler)
    })
}
```

This is **critical** for WASM event-loop safety — without it, blocking on
WebGPU channels or other async operations deadlocks the Go scheduler.

## Troubleshooting

**"Go WASM runtime not found"** — Load `wasm_exec.js` before `mempipe.js`:
```html
<script src="wasm_exec.js"></script>
<script src="mempipe.js"></script>
```

**"SharedArrayBuffer is not defined"** — Add COOP/COEP headers to your server.

**Large WASM binary** — Use `make tinygo-wasm` for a smaller build, or
ensure `-ldflags="-s -w"` is set.

**Slow first load** — Use `WebAssembly.instantiateStreaming()` for progressive
loading (the JS wrapper handles this automatically when given a URL).
