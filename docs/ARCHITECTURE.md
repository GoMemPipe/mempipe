# MemPipe Architecture

## Overview

MemPipe is a zero-allocation pipeline library.  All working memory is
pre-allocated in a single `[]byte` slab (the **arena**) at build time.
Pipeline stages (**cells**) read and write **regions** — typed views into
the arena — using raw pointer arithmetic.  The **scheduler** runs cells
in topological order, sequentially or in parallel.

```
┌─────────────────── Pipeline ───────────────────────┐
│                                                     │
│   ┌─────────────────────────────────────────────┐  │
│   │              RegionArena                     │  │
│   │  ┌────────┐┌────────┐┌────────┐┌────────┐  │  │
│   │  │  Rgn 0 ││  Rgn 1 ││  Rgn 2 ││  Rgn 3 │  │  │
│   │  │ 64-byte││        ││        ││        │  │  │
│   │  │aligned ││        ││        ││        │  │  │
│   │  └────────┘└────────┘└────────┘└────────┘  │  │
│   │  ◄──── single make([]byte, totalSize) ────► │  │
│   └─────────────────────────────────────────────┘  │
│                                                     │
│   ┌──────┐   ┌──────┐   ┌──────┐                  │
│   │Cell A│ → │Cell B│ → │Cell C│                  │
│   └──┬───┘   └──┬───┘   └──┬───┘                  │
│      │           │           │                      │
│    read/       read/       read/                    │
│    write       write       write                    │
│    regions     regions     regions                  │
│                                                     │
│   ┌──────────────────────────────────┐             │
│   │         Scheduler                 │             │
│   │  topological sort → execute loop │             │
│   └──────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

## Memory Model

### The Arena

```go
// One allocation for the entire pipeline
arena := &RegionArena{
    memory: make([]byte, totalSize), // THE allocation
}
```

The `RegionArena` is a single contiguous byte slice.  Every region is
placed at a **64-byte aligned** offset for cache-line and SIMD friendliness.
After construction, no further heap allocations occur on the hot path.

### Regions

A `Region` is a **typed view** into arena memory:

```go
type Region struct {
    name   string
    base   uintptr       // pointer into arena.memory
    size   int64
    mode   RegionMode    // stream, ring, slab, windowed, append
    layout *RegionLayout // field names/types/offsets
}
```

Every field accessor (`.U32("counter")`, `.F32("temp")`, `.SetF32("temp", 42.0)`)
is a direct `unsafe.Pointer` dereference — **zero allocations, zero interface boxing**.

### RegionLayout

Layouts are computed once at build time from struct tags:

```go
type SensorData struct {
    Temp     float32 `mempipe:"field:temp"`
    Humidity float32 `mempipe:"field:humidity"`
    Count    uint32  `mempipe:"field:count"`
    Active   bool    `mempipe:"field:active"`
}

layout, _ := runtime.LayoutFromStruct("sensor", SensorData{})
// layout.Fields:
//   {Name:"temp",     Type:f32,  Offset:0,  Size:4}
//   {Name:"humidity", Type:f32,  Offset:4,  Size:4}
//   {Name:"count",    Type:u32,  Offset:8,  Size:4}
//   {Name:"active",   Type:bool, Offset:12, Size:1}
// layout.Size: 16 (padded to alignment)
```

Alternatively, layouts can be specified imperatively:

```go
pipe.AddFieldRegion("data", map[string]string{
    "x": "f64",
    "y": "f64",
})
```

### TypedRegion[T]

`TypedRegion[T]` provides generic, type-safe access to a region's data.
It maps the struct T's fields to region memory using `mempipe` struct tags
and copies data between Go values and arena memory via `unsafe.Pointer`:

```go
tr, _ := runtime.NewTypedRegion[SensorData](arena, "sensor")
v := tr.Get()    // reads SensorData from arena — zero alloc
v.Temp = 22.5
tr.Set(v)        // writes SensorData to arena — zero alloc
```

At the pipeline level, `RegionHandle[T]` wraps this:

```go
sensor := mempipe.AddRegion[SensorData](pipe, "sensor")
// after pipe.Run():
v := sensor.Get()
sensor.Set(v)
```

## Pipeline Lifecycle

### 1. Define

```go
pipe := mempipe.NewPipeline(mempipe.WithWorkers(4))
sensor := mempipe.AddRegion[SensorData](pipe, "sensor")
output := mempipe.AddRegion[Output](pipe, "output")
```

### 2. Register Cells

```go
pipe.Cell("process", func() {
    s := sensor.Get()
    s.Temp *= 1.01
    sensor.Set(s)
}, []string{"sensor"}, []string{"output"})
```

Cells declare their input and output regions for dependency ordering.
`SimpleCell` omits declarations and runs in registration order.

### 3. Validate & Build

On the first call to `Run()`:

1. **Validate** — check for duplicate names, unknown region references, cycles
2. **Compute layouts** — each region's field offsets from struct tags
3. **Allocate arena** — single `make([]byte, totalSize)` with 64-byte alignment
4. **Create regions** — each region gets a base pointer into the arena
5. **Wire scheduler** — add cells, set scheduling policy

### 4. Execute

```go
pipe.Run(1000)        // fixed iterations
pipe.RunContinuous(ctx) // until context cancelled
```

Each iteration:
1. Scheduler topologically sorts cells (Kahn's algorithm)
2. Cells execute in order (sequential) or concurrently (parallel policy)
3. Optional `OnIteration` callback fires between iterations

## Scheduling

```go
// Sequential (default) — cells run one at a time in topo order
pipe := mempipe.NewPipeline()

// Parallel — independent cells run concurrently
pipe := mempipe.NewPipeline(mempipe.WithWorkers(4))
```

The scheduler supports:
- **ScheduleSequential** — cells execute one at a time
- **ScheduleParallel** — cells with no data dependencies execute concurrently
  using a worker pool with `sync.WaitGroup`

Topological ordering uses Kahn's algorithm on the cell→region dependency graph.
If no dependencies are declared (all `SimpleCell`), cells run in registration order.

## Module System

Modules are registered via `module.Registry`:

```go
type Module interface {
    Name() string
    Init() error
}

// Optional lifecycle interfaces:
type Ticker interface { Tick(tickCount uint64) }
type Shutdowner interface { Shutdown() error }
```

Built-in modules: `math`, `time`, `io`, `sys`, `audio`, `http`, `wasm_http`, `embedded_http`.

Modules expose typed Go methods — no `interface{}` boxing:

```go
type MathModule struct{}
func (m *MathModule) Sin(x float64) float64 { return math.Sin(x) }
```

## Inference Engine

The `inference` package provides a separate arena-backed neural network
execution engine. See [INFERENCE.md](INFERENCE.md) for the full operator
reference and API guide.

```go
model, _ := inference.LoadModelFromBytes(data)
engine, _ := inference.NewEngine(model)
outputs, _ := engine.InferTensor() // zero alloc
```

### Inference Arena vs Pipeline Arena

MemPipe has **two independent arena systems**, each optimized for its domain:

| | Pipeline Arena (`runtime.RegionArena`) | Inference Arena (`inference.InferenceArena`) |
|---|---|---|
| **Purpose** | Named typed regions for pipeline data flow | Neural network weights, activations, I/O tensors |
| **Allocation** | `make([]byte, totalSize)` with region offsets | `make([]byte, totalSize)` with bump allocator |
| **Access pattern** | Named field access: `region.F32("temp")` | Tensor views: `tensor.AtF32(row, col)` |
| **Alignment** | 64-byte cache-line per region | 64-byte per tensor allocation |
| **Growth** | `arena.Grow()` for offline resize | Fixed at compile time (shape inference) |
| **Lifecycle** | Defined → Built → Run (many iterations) | Compile → Infer (many forward passes) |

Both share the core principle: **one allocation, zero GC on the hot path**.

### Compiled Execution Model

The engine compiles the model graph into a flat list of `compiledOp` structs,
each holding pre-resolved pointers to input and output tensors:

```go
type compiledOp struct {
    op      Operator   // the operator implementation
    inputs  []*Tensor  // pre-resolved input tensor pointers
    outputs []*Tensor  // pre-resolved output tensor pointers
}
```

During `InferTensor()`, the engine simply iterates this list — no map lookups,
no interface dispatch overhead beyond the `Execute` method call.

### Hardware-Accelerated MatMul

The `matMulF32` kernel is split across three build-tagged implementations:

```
inference/
├── matmul_generic.go   # Portable i-p-j loop (fallback)
├── matmul_simd.go      # BCE-optimized 4×4 micro-kernel (amd64/arm64)
└── matmul_wasm.go      # WebGPU compute shader (browser WASM)
```

Each file defines a `matMulAccel` struct embedded into `matMulOp`. The `Engine`
calls `Init(arena)` on operators implementing the `Initializable` interface,
allowing the SIMD path to pre-allocate packing buffers and the WASM path to
compile GPU shaders — all before the first inference call.

See [CONDITIONAL_COMPILATION.md](CONDITIONAL_COMPILATION.md) for the full build
tag matrix.

## Memory-Native I/O

MemPipe replaces OS-level I/O with deterministic, memory-backed alternatives.

### MemoryPipe (Ring Buffer)

`runtime.MemoryPipe` is a thread-safe ring buffer that replaces OS stdout/stderr:

```go
type MemoryPipe struct {
    name   string
    buf    []byte     // ring buffer (default 64KB)
    head   int
    tail   int
    size   int
    lines  []string   // line storage for ReadLines()
}
```

**Methods**: `Write`, `WriteBytes`, `Writeln`, `Read`, `ReadLines`, `ReadLastN`,
`Clear`, `Size`, `LineCount`.

**Why**: Deterministic output capture for testing and replay. No OS syscalls on
the hot path. Works identically on native, WASM, and embedded targets.

### PipeManager

`runtime.PipeManager` manages named memory pipes:

```go
pm := runtime.NewPipeManager()
pm.CreatePipe("events", 4096)
pipe, _ := pm.GetPipe("events")
pipe.Writeln("sensor triggered")
lines := pipe.ReadLines()
```

**Methods**: `CreatePipe`, `GetPipe`, `GetOrCreatePipe`, `DeletePipe`,
`ListPipes`, `DumpAll`, `ClearAll`.

### MemoryLogger

`runtime.MemoryLogger` provides structured, tick-indexed logging backed by
the `RuntimeClock`:

```go
logger := runtime.NewMemoryLogger(clock, 10000) // max 10K entries
logger.SetMinLevel(runtime.LogInfo)

logger.Info("pipeline started")
logger.Warn("high latency detected")
logger.Error("region overflow")

// Query by tick range
entries := logger.GetEntriesSince(lastTick)
recent := logger.GetLastN(50)
formatted := logger.FormatAll()
```

Each entry stores: `{Tick uint64, Level LogLevel, Message string}`.

**Log levels**: `Debug`, `Info`, `Warn`, `Error`.

### DualLogger

`runtime.DualLogger` wraps `MemoryLogger` plus optional OS `fmt.Printf` output,
useful during development/migration:

```go
dual := runtime.NewDualLogger(memLogger, true) // true = also print to OS
dual.Info("visible in both memory log and terminal")
```

## RuntimeClock

`runtime.RuntimeClock` is a deterministic, monotonic tick counter that replaces
OS time. All operations use `sync/atomic` for lock-free thread safety.

```go
clock := runtime.NewRuntimeClock(1000) // 1000 ticks per ms
clock.Tick()                           // atomic increment
clock.TickBy(5)                        // atomic increment by 5

now := clock.Now()                     // current tick
ms  := clock.NowMs()                   // current time in ms
elapsed := clock.Since(startTick)      // ticks since startTick
elapsedMs := clock.SinceMs(startTick)  // ms since startTick

clock.Reset()                          // zero for replay
clock.SetTicks(42)                     // set exact value for testing
```

**Why**: Fully replayable — same inputs produce same tick sequence. No OS
time dependency. Essential for deterministic pipeline execution.

## Zero-GC Design Rules

1. **One allocation per arena** — `make([]byte, N)` once
2. **No `interface{}` on hot paths** — use concrete types or generics
3. **No `make`/`new`/`append` on hot paths** — all memory pre-allocated
4. **No `reflect` on hot paths** — struct tags parsed at build time only
5. **`//mem:hot` / `//mem:nogc` annotations** — document allocation-free code
6. **Benchmark enforcement** — all `//mem:nogc` functions must show `0 allocs/op`

## Platform Support

| Platform | API | Notes |
|----------|-----|-------|
| Native | Full | All features, SIMD-accelerated MatMul |
| WASM | `LoadModelFromBytes`, `InferTensor`, zero-copy JS views | No filesystem, WebGPU MatMul |
| Embedded | Core arena + inference | Limited `reflect`, no HTTP, NopMutex |

See [CONDITIONAL_COMPILATION.md](CONDITIONAL_COMPILATION.md) for the full matrix.

## Further Reading

| Document | Description |
|----------|-------------|
| [PIPELINE_API.md](PIPELINE_API.md) | Top-level pipeline API (`NewPipeline`, `AddRegion[T]`, cells, scheduling) |
| [BUILDER_API.md](BUILDER_API.md) | Builder package API (`builder.NewPipeline`, `Context`, typed accessors) |
| [INFERENCE.md](INFERENCE.md) | Inference engine — all 33 operators, tensors, quantization, benchmarks |
| [MODULES.md](MODULES.md) | Module system — audio, HTTP, I/O, math, sys, time modules |
| [TOOLS.md](TOOLS.md) | memlint static analyzer and mempipe-convert model converter |
| [EXAMPLES.md](EXAMPLES.md) | Guide to all 7 example applications |
| [MPMODEL_FORMAT.md](MPMODEL_FORMAT.md) | `.mpmodel` binary format byte-level specification |
| [WASM.md](WASM.md) | WASM deployment — building, JS integration, WebGPU, Web Audio |
| [CONDITIONAL_COMPILATION.md](CONDITIONAL_COMPILATION.md) | Build tags, platform matrix, feature detection |
