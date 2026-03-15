# MemPipe Conditional Compilation Matrix

This document describes which packages compile on which targets and how
build tags control feature availability.

## Build Targets

| Target | Build Command | Build Tags | Platform() |
|--------|---------------|------------|------------|
| **Native** (Linux/macOS/Windows) | `go build ./...` | `!js && !embedded` | `"native"` |
| **WASM** (Browser/Node.js) | `GOOS=js GOARCH=wasm go build ./platform/wasm/` | `js` | `"wasm"` |
| **Embedded** (TinyGo/bare-metal) | `tinygo build -tags embedded ...` | `embedded` | `"embedded"` |

## Package Compilation Matrix

| Package | Native | WASM | Embedded | Notes |
|---------|--------|------|----------|-------|
| `runtime/arena.go` | ✅ | ✅ | ✅ | Zero-alloc arena — core, compiles everywhere |
| `runtime/region.go` | ✅ | ✅ | ✅ | Typed memory access via unsafe.Pointer |
| `runtime/layout.go` | ✅ | ✅ | ⚠️ | Uses `reflect` — TinyGo has limited reflect support |
| `runtime/typed_region.go` | ✅ | ✅ | ❌ | Uses `reflect.TypeOf` — not available in TinyGo |
| `runtime/scheduler.go` | ✅ | ✅ | ⚠️ | Uses `sync.Mutex` and goroutines |
| `runtime/pipe.go` | ✅ | ✅ | ✅ | Build-tagged `rwMutex` — NopMutex on embedded |
| `runtime/clock.go` | ✅ | ✅ | ✅ | Build-tagged `rwMutex` — NopMutex on embedded |
| `runtime/logger.go` | ✅ | ✅ | ✅ | Build-tagged `rwMutex` — NopMutex on embedded |
| `inference/model.go` | ✅ | ✅ | ✅ | `LoadModel` needs `os.ReadFile` (native only); `LoadModelFromBytes` works everywhere |
| `inference/tensor.go` | ✅ | ✅ | ✅ | Arena-backed, unsafe.Pointer — fully portable |
| `inference/operator.go` | ✅ | ✅ | ✅ | Build-tagged `rwMutex` — NopMutex on embedded |
| `inference/engine.go` | ✅ | ✅ | ✅ | Zero-alloc hot path — fully portable |
| `inference/matmul_generic.go` | ⚠️ | ⚠️ | ✅ | Fallback i-p-j MatMul (used when no SIMD/WebGPU) |
| `inference/matmul_simd.go` | ✅ | ❌ | ❌ | BCE-optimized tiled MatMul for amd64/arm64 |
| `inference/matmul_wasm.go` | ❌ | ✅ | ❌ | WebGPU compute-shader accelerated MatMul |
| `inference/quantize.go` | ✅ | ✅ | ✅ | Pure math, no platform deps |
| `module/module.go` | ✅ | ✅ | ✅ | Build-tagged `rwMutex` — NopMutex on embedded |
| `module/math/` | ✅ | ✅ | ✅ | Pure computation |
| `module/time/` | ✅ | ✅ | ⚠️ | Uses `time` package (limited in TinyGo) |
| `module/io/` | ✅ | ✅ | ✅ | Pure memory-pipe I/O — no `os` dependency |
| `module/http/` | ✅ | ❌ | ❌ | Uses `net/http` — not available in WASM/embedded |
| `module/sys/` | ✅ | ❌ | ❌ | Uses `os` and `os/signal` |
| `module/audio/` | ✅ | ✅ | ✅ | Pure computation on arena memory |
| `module/wasm_http/` | ❌ | ✅ | ❌ | Uses `syscall/js` for fetch API |
| `module/embedded_http/` | ❌ | ❌ | ✅ | Stub / no-op |
| `platform/wasm/` | ❌ | ✅ | ❌ | WASM entry point, uses `syscall/js` |
| `platform/embedded/` | ✅ | ✅ | ✅ | No-op stubs, compiles everywhere |
| `build/` | ✅ | ✅ | ✅ | Build-tagged config files |
| `builder/` | ✅ | ✅ | ⚠️ | Pipeline builder API |

**Legend**: ✅ = compiles and works, ⚠️ = compiles but limited functionality, ❌ = does not compile on this target

## Feature Detection API

```go
import "github.com/GoMemPipe/mempipe/build"

// Compile-time platform detection (no runtime cost)
build.Platform()       // → "native", "wasm", or "embedded"
build.HasHTTP()        // → networking available?
build.HasFilesystem()  // → file I/O available?
build.HasConcurrency() // → goroutines + sync available?
build.HasSignalHandling() // → OS signals available?
build.HasNetworkIO()   // → raw TCP/UDP available?
build.HasOSInteraction() // → exec, environ, etc.?

// Version
build.Version     // "0.1.0"
build.ProjectName // "MemPipe"
```

## Build Tag Organization

```
build/
├── config.go           # Shared: Version, ProjectName (no build tags)
├── config_native.go    # //go:build !js && !embedded
├── config_wasm.go      # //go:build js
└── config_embedded.go  # //go:build embedded
```

## WASM-Specific Notes

### LoadModel vs LoadModelFromBytes

```go
// Native — can load from filesystem
model, err := inference.LoadModel("path/to/model.mpmodel")

// WASM — must load from bytes (no filesystem)
model, err := inference.LoadModelFromBytes(modelBytes)
```

### Zero-Copy Memory Access

The WASM entry point exposes tensor pointers into the WASM linear memory.
JavaScript can create `Float32Array` views directly over these pointers
for zero-copy data sharing:

```javascript
const ptr = mempipe.getInputPtr(engine, 0);
const shape = mempipe.getInputShape(engine, 0);
const count = shape.reduce((a, b) => a * b, 1);
const view = new Float32Array(wasmMemory.buffer, ptr, count);
// Write directly — no copy!
view.set(myInputData);
mempipe.inferZeroCopy(engine);
```

## Embedded/TinyGo Notes

### Known Limitations

1. **reflect**: TinyGo has limited `reflect` support. `TypedRegion` and
   `LayoutFromStruct` will not work. Use manual `LayoutTable` construction:

   ```go
   layout := &runtime.RegionLayout{
       Name: "sensor",
       Fields: []runtime.FieldLayout{
           {Name: "temp", Type: runtime.TypeF32, Offset: 0, Size: 4},
           {Name: "humidity", Type: runtime.TypeF32, Offset: 4, Size: 4},
       },
       TotalSize: 8,
   }
   ```

2. **sync.RWMutex**: All packages that previously used `sync.RWMutex`
   now use a build-tagged `rwMutex` type that compiles to a no-op on
   `embedded` targets, eliminating sync overhead and ISR deadlock risk.
   Each package has `rwmutex.go` / `rwmutex_embedded.go` build-tagged files.

3. **os.ReadFile**: Not available. Use `inference.LoadModelFromBytes`.

4. **net/http**: Not available. Use `platform/embedded` stubs.

### Recommended Build Command

```bash
# Arduino Nano 33
tinygo build -target=arduino-nano33 -tags embedded -o firmware.bin ./cmd/embedded/

# Generic ARM Cortex-M4
tinygo build -target=cortex-m4 -tags embedded -o firmware.elf ./cmd/embedded/

# WASI (embedded-like WASM)
tinygo build -target=wasi -tags embedded -o mempipe.wasm ./platform/wasm/
```

## Hardware-Accelerated MatMul Dispatch

The `matMulF32` kernel — the hot-path function called by `matMulOp`,
`denseOp`, and `batchedMatMulOp` — is split across three build-tagged
files under `inference/`:

| File | Build Tag | Strategy |
|------|-----------|----------|
| `matmul_generic.go` | `!(js && wasm) && !((linux\|\|darwin) && (amd64\|\|arm64))` | Portable i-p-j loop (fallback) |
| `matmul_wasm.go` | `js && wasm` | WebGPU compute shader via `syscall/js` |
| `matmul_simd.go` | `(linux \|\| darwin) && (amd64 \|\| arm64)` | BCE-optimized blocked 4×4 micro-kernel |

Each file also defines a `matMulAccel` struct embedded into `matMulOp`.
This struct is empty on the generic path, holds WebGPU pipeline state
on WASM, and holds a packing scratch buffer on the SIMD path.

### Initializable Interface

Operators that need one-time setup implement the optional interface:

```go
type Initializable interface {
    Init(arena *InferenceArena) error
}
```

The `Engine` calls `Init(arena)` once during `compile()` for any
operator that satisfies `Initializable`. This keeps the `Operator`
interface unchanged and is fully backward-compatible.

### Zero-GC Guarantee

All three `matMulF32` implementations maintain 0 B/op and 0 allocs/op:

- **Generic**: No allocations — direct slice arithmetic.
- **SIMD**: Packing buffer pre-allocated from arena in `Init()`.
- **WASM**: `js.Value` temporaries are stack-local; `js.FuncOf`
  callbacks and GPU pipeline objects are created once in `Init()`.

### WebGPU Event-Loop Safety (`goAsync` Pattern)

Go WASM runs on a single-threaded event loop. Blocking operations (like
waiting for a WebGPU compute result) deadlock the `handleEvent` goroutine.
The solution is the `goAsync()` wrapper used in the WASM demo and matmul:

```go
// goAsync wraps a blocking Go callback as a JS Promise.
// The goroutine spawned inside can safely block on channels.
func goAsync(fn func() (any, error)) js.Func {
    return js.FuncOf(func(this js.Value, args []js.Value) any {
        resolve := args[0]
        reject := args[1]
        go func() {
            result, err := fn()
            if err != nil {
                reject.Invoke(err.Error())
            } else {
                resolve.Invoke(result)
            }
        }()
        return nil
    })
}
```

This pattern is used by:
- `matmul_wasm.go` — GPU dispatch blocks on `<-doneChan` in a goroutine
- `examples/wasm_demo/main.go` — all JS-exported functions use `goAsync`
- `platform/wasm/main.go` — inference API exports

### Operator Compilation Matrix

All 33 operators compile on all three platforms. Key differences:

| Operator Feature | Native | WASM | Embedded |
|-----------------|--------|------|----------|
| MatMul kernel | SIMD 4×4 micro-kernel | WebGPU compute shader | Generic i-p-j loop |
| Conv2D im2col scratch | From arena | From arena | From arena |
| Operator registry locking | `sync.RWMutex` | `sync.RWMutex` | No-op (single-threaded) |
| `LoadModel(path)` | ✅ filesystem | ❌ use `LoadModelFromBytes` | ❌ use `LoadModelFromBytes` |
| WebGPU init | N/A | `EnsureWebGPU()` + `IsWebGPUReady()` | N/A |
