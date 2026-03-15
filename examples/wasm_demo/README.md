# MemPipe WASM Demo

Interactive browser demo showcasing two features of MemPipe running on WebAssembly:

1. **Basic WASM Features** — arena allocation, zero-copy memory, tensor ops, operator benchmarks
2. **GPT-2 Inference** — hardware-accelerated MatMul (WebGPU when available, tiled WASM fallback) driving autoregressive text generation from a `.mpmodel` file

## Quick Start

```bash
# Build & serve
chmod +x build.sh
./build.sh --serve 8080
```

Then open http://localhost:8080 in a WebGPU-capable browser (Chrome 113+, Edge 113+).

## Manual Build

```bash
# 1. Compile WASM
cd /path/to/mempipe
GOOS=js GOARCH=wasm go build -o examples/wasm_demo/demo.wasm ./examples/wasm_demo/

# 2. Copy Go's WASM runtime
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" examples/wasm_demo/

# 3. Symlink the model file (optional — page has a file picker fallback)
ln -s ../../gpt2.mpmodel examples/wasm_demo/gpt2.mpmodel

# 4. Serve
cd examples/wasm_demo
python3 -m http.server 8080
```

## Files

| File | Description |
|------|-------------|
| `main.go` | Go WASM entry point — registers `mempipeDemo` JS global |
| `index.html` | Single-page app with all UI, CSS, and JS |
| `build.sh` | Build + optional serve script |
| `demo.wasm` | Compiled WASM binary (generated) |
| `wasm_exec.js` | Go WASM runtime support (copied from Go install) |
| `gpt2.mpmodel` | GPT-2 model weights (symlinked from project root) |

## Browser Requirements

- Any modern browser for Demo 1 (basic features)
- Chrome 113+ / Edge 113+ for WebGPU-accelerated MatMul
- Falls back to optimized tiled WASM MatMul on older browsers

## What's Measured

**Demo 1:**
- Arena allocation time (µs per tensor)
- Zero-copy WASM memory access
- Per-operator benchmark: iterations, total time, avg µs/op, throughput (GFLOPS), allocs/op = 0

**Demo 2:**
- Model parse time, engine init time, JS→WASM copy time
- MatMul benchmark at GPT-2 shapes (1×768×768, 1×768×3072, 1×768×50257) with GFLOPS
- Per-token generation time, total tokens/sec
- Step-by-step timing table for each generated token
