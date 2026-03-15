# Audio DSP Example

Real-time audio processing pipeline using MemPipe's zero-allocation regions.

## Pipeline

```
oscillator → low-pass filter → gain → output metering
```

Each cell processes 128-sample blocks (matching Web Audio's `AudioWorklet` quantum).
All state lives in arena-backed regions — zero allocations per frame.

## Run

```bash
go run ./examples/audio_dsp
```

## WASM Deployment

This pipeline is designed for `AudioWorkletProcessor` integration:

1. Build WASM: `GOOS=js GOARCH=wasm go build -o audio.wasm ./examples/audio_dsp`
2. Load in an AudioWorklet with `mempipe.js`
3. Use `SharedArrayBuffer` for zero-copy sample exchange

See [docs/WASM.md](../../docs/WASM.md) for full browser integration details.

## Key APIs Demonstrated

- `AddRegion[T]()` — typed arena-backed state
- `pipe.Cell()` — cells with explicit input/output dependencies
- `pipe.OnIteration()` — per-frame monitoring callback
- `RegionHandle[T].Get()` / `.Set()` — zero-alloc struct access
