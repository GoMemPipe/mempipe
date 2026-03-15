# Pipeline API Reference

The top-level `mempipe` package provides the primary user-facing API for building
zero-allocation data pipelines. It wraps the lower-level `runtime` package with
a type-safe, generics-based interface.

## Overview

```go
import "github.com/GoMemPipe/mempipe"

// 1. Create a pipeline
pipe := mempipe.NewPipeline(mempipe.WithWorkers(4))

// 2. Define typed regions (arena-backed memory)
sensor := mempipe.AddRegion[SensorData](pipe, "sensor")
output := mempipe.AddRegion[OutputData](pipe, "output")

// 3. Register cells (processing stages)
pipe.Cell("process", func() {
    s := sensor.Get()    // zero-alloc read from arena
    s.Temp *= 1.01
    sensor.Set(s)        // zero-alloc write to arena
}, []string{"sensor"}, []string{"output"})

// 4. Run
pipe.Run(1000)           // 1000 deterministic iterations
// or
pipe.RunContinuous(ctx)  // until context cancellation
```

## Pipeline Construction

### NewPipeline

```go
func NewPipeline(opts ...Option) *Pipeline
```

Creates a new pipeline with the given options. Defaults to 1 worker
(sequential execution).

```go
pipe := mempipe.NewPipeline()
pipe := mempipe.NewPipeline(mempipe.WithWorkers(4))
pipe := mempipe.NewPipeline(
    mempipe.WithWorkers(8),
    mempipe.WithArenaSizeHint(1<<20), // 1MB hint
)
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `WithWorkers(n int)` | Number of parallel workers (min 1). When >1, enables `ScheduleParallel`. | `1` |
| `WithArenaSizeHint(bytes int64)` | Hint for total arena size pre-allocation | Auto-computed |

## Region Registration

### AddRegion[T] (Generic)

```go
func AddRegion[T any](pipe *Pipeline, name string, opts ...RegionOption) *RegionHandle[T]
```

Registers a typed region backed by arena memory. The struct `T` must have
`mempipe` struct tags defining the field layout.

```go
type SensorData struct {
    Temperature float32 `mempipe:"field:temperature"`
    Humidity    float32 `mempipe:"field:humidity"`
    Count       uint32  `mempipe:"field:count"`
    Active      bool    `mempipe:"field:active"`
}

sensor := mempipe.AddRegion[SensorData](pipe, "sensor")
sensor := mempipe.AddRegion[SensorData](pipe, "sensor",
    mempipe.WithRegionMode("ring"),
    mempipe.WithRegionSize(4096),
)
```

### Struct Tag Format

The `mempipe` struct tag controls how fields are mapped to arena memory:

```
mempipe:"field:<name>[,type:<type>][,offset:<n>][,size:<n>]"
```

| Tag Attribute | Required | Description | Example |
|---------------|----------|-------------|---------|
| `field:<name>` | Yes | Field name in the region layout | `field:temperature` |
| `type:<type>` | No | Explicit field type (auto-inferred from Go type if omitted) | `type:f32` |
| `offset:<n>` | No | Explicit byte offset (auto-computed if omitted) | `offset:8` |
| `size:<n>` | No | Explicit field size in bytes (auto-computed if omitted) | `size:4` |

**Supported types** (inferred from Go types):

| Go Type | Region Type | Size |
|---------|-------------|------|
| `uint8` | `u8` | 1 |
| `uint16` | `u16` | 2 |
| `uint32` | `u32` | 4 |
| `uint64` | `u64` | 8 |
| `int8` | `i8` | 1 |
| `int16` | `i16` | 2 |
| `int32` | `i32` | 4 |
| `int64` | `i64` | 8 |
| `float32` | `f32` | 4 |
| `float64` | `f64` | 8 |
| `bool` | `bool` | 1 |

### AddFieldRegion (Non-Generic)

```go
func (p *Pipeline) AddFieldRegion(name string, fields map[string]string, opts ...RegionOption) *Pipeline
```

Registers a region using an explicit field-name → type-string map. Useful when
the struct layout isn't known at compile time.

```go
pipe.AddFieldRegion("data", map[string]string{
    "x": "f64",
    "y": "f64",
    "z": "f64",
})
```

### Region Options

| Option | Description | Default |
|--------|-------------|---------|
| `WithRegionMode(mode string)` | Region mode: `"stream"`, `"ring"`, `"slab"`, `"windowed"`, `"append"` | `"stream"` |
| `WithRegionSize(bytes int64)` | Explicit region size in bytes | Auto-computed from layout |

### Region Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `stream` | Default — linear read/write | General-purpose data |
| `ring` | Ring buffer with wrapping | Time-series, audio buffers |
| `slab` | Fixed-size object pool | Entity management |
| `windowed` | Sliding window over data | Moving averages |
| `append` | Append-only log | Event logging |

## RegionHandle[T]

`RegionHandle[T]` provides type-safe, zero-allocation access to arena memory.

### Get / Set

```go
func (h *RegionHandle[T]) Get() T
func (h *RegionHandle[T]) Set(v T)
```

Read and write the entire struct from/to arena memory. Both operations are
**zero-allocation** — they use `unsafe.Pointer` arithmetic to copy field-by-field
between the Go value and arena memory.

```go
sensor := mempipe.AddRegion[SensorData](pipe, "sensor")

// After pipeline is built (via Run or explicit build):
v := sensor.Get()         // zero-alloc read from arena
v.Temperature = 22.5
v.Count++
sensor.Set(v)             // zero-alloc write to arena
```

**Lazy resolution**: The first call to `Get()` or `Set()` resolves the underlying
`runtime.TypedRegion[T]` from the arena. Panics if the pipeline hasn't been built yet.

### Metadata

```go
func (h *RegionHandle[T]) Name() string           // region name
func (h *RegionHandle[T]) Region() *runtime.Region // underlying region (for field-level access)
```

### Field-Level Access (via Region)

For fine-grained access to individual fields without reading the entire struct:

```go
r := sensor.Region()
temp := r.F32("temperature")           // read float32 field
r.SetF32("temperature", 22.5)          // write float32 field
count := r.U32("count")                // read uint32 field
r.SetBool("active", true)              // write bool field

// Vector access (for vecf32 fields)
ptr, len := r.VecF32Ptr("samples")     // zero-alloc pointer + length
r.VecF32Write("samples", data)         // write []float32 into arena
samples := r.VecF32Read("samples")     // allocating read (returns new slice)
```

## Cell Registration

### Cell (with Dependencies)

```go
func (p *Pipeline) Cell(name string, fn CellFunc, inputs []string, outputs []string) *Pipeline
```

Registers a cell with explicit input/output region dependency declarations.
Dependencies determine topological execution order and enable parallel scheduling.

```go
pipe.Cell("physics", func() {
    pos := position.Get()
    vel := velocity.Get()
    pos.X += vel.DX * dt
    pos.Y += vel.DY * dt
    position.Set(pos)
}, []string{"velocity"}, []string{"position"})
```

- `inputs`: Region names this cell reads from
- `outputs`: Region names this cell writes to
- Returns `*Pipeline` for method chaining

### SimpleCell (no Dependencies)

```go
func (p *Pipeline) SimpleCell(name string, fn CellFunc) *Pipeline
```

Registers a cell with no declared dependencies. SimpleCells run in
registration order.

```go
pipe.SimpleCell("log", func() {
    fmt.Printf("tick %d: temp=%.1f\n", tick, sensor.Get().Temperature)
})
```

### CellFunc

```go
type CellFunc func()
```

A cell is simply a `func()` closure that captures region handles. No
arguments, no return values — all data flows through arena regions.

## Iteration Callback

```go
func (p *Pipeline) OnIteration(fn func()) *Pipeline
```

Registers a callback that fires once per iteration, after all cells have
executed. Useful for logging, metrics, or progress reporting.

```go
pipe.OnIteration(func() {
    if tick%100 == 0 {
        v := sensor.Get()
        fmt.Printf("[%d] temp=%.2f\n", tick, v.Temperature)
    }
})
```

## Validation

```go
func (p *Pipeline) Validate() error
```

Checks the pipeline for configuration errors:
- Duplicate region names
- Duplicate cell names
- Cells referencing unknown input/output region names

Called automatically by `Run()` / `RunContinuous()`, but can be called
explicitly for early validation.

```go
if err := pipe.Validate(); err != nil {
    log.Fatalf("pipeline config error: %v", err)
}
```

## Execution

### Run (Fixed Iterations)

```go
func (p *Pipeline) Run(iterations int) error
```

Builds the pipeline (if not already built) and executes the given number
of iterations. Each iteration:

1. Scheduler topologically sorts cells (Kahn's algorithm)
2. Cells execute in order (sequential) or concurrently (parallel)
3. `OnIteration` callback fires (if registered)

```go
pipe.Run(1_000_000) // 1M deterministic ticks
```

### RunContinuous (Until Cancelled)

```go
func (p *Pipeline) RunContinuous(ctx context.Context) error
```

Builds the pipeline and runs indefinitely until the context is cancelled.
Returns the context's error (typically `context.DeadlineExceeded` or
`context.Canceled`).

```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()
err := pipe.RunContinuous(ctx)
// err == context.DeadlineExceeded after 10s
```

## Post-Run Inspection

After `Run()` or `RunContinuous()` returns:

```go
// Access the underlying arena
arena := pipe.Arena()

// Access the deterministic clock
clock := pipe.Clock()
fmt.Printf("Total ticks: %d\n", clock.Now())

// Access captured stdout/stderr (memory-native I/O)
stdout := pipe.Stdout()
stderr := pipe.Stderr()
```

## Two-Phase Lifecycle

The pipeline has a strict two-phase lifecycle:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Definition     │     │      Build       │     │    Execution     │
│                  │────►│                  │────►│                  │
│ AddRegion[T]()   │     │ Compute layouts  │     │ Run(n) or        │
│ Cell()           │     │ Allocate arena   │     │ RunContinuous()  │
│ SimpleCell()     │     │ Create regions   │     │                  │
│ OnIteration()    │     │ Wire scheduler   │     │ Get()/Set() ok   │
│ Validate()       │     │                  │     │ Clock/Arena ok   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
     mutable                 THE allocation           zero-alloc
```

1. **Definition phase**: Register regions, cells, callbacks. Mutable.
2. **Build phase**: Triggered lazily on first `Run()` / `RunContinuous()`.
   Computes layouts, allocates the single arena `make([]byte, totalSize)`,
   creates `Region` objects, wires the scheduler. This is THE allocation.
3. **Execution phase**: Iterator loop. `Get()` / `Set()` are now valid.
   Zero heap allocations on the hot path.

**Important**: Calling `Get()` or `Set()` before the pipeline is built will panic.

## Complete Example

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/GoMemPipe/mempipe"
)

type Physics struct {
    X  float64 `mempipe:"field:x"`
    Y  float64 `mempipe:"field:y"`
    VX float64 `mempipe:"field:vx"`
    VY float64 `mempipe:"field:vy"`
}

type Metrics struct {
    Speed    float64 `mempipe:"field:speed"`
    Distance float64 `mempipe:"field:distance"`
}

func main() {
    pipe := mempipe.NewPipeline(mempipe.WithWorkers(2))

    phys := mempipe.AddRegion[Physics](pipe, "physics")
    metrics := mempipe.AddRegion[Metrics](pipe, "metrics")

    // Initialize
    pipe.SimpleCell("init_once", func() {
        p := phys.Get()
        if p.VX == 0 && p.VY == 0 {
            p.VX, p.VY = 1.0, 0.5
            phys.Set(p)
        }
    })

    // Physics update
    pipe.Cell("integrate", func() {
        p := phys.Get()
        p.X += p.VX * 0.016 // ~60 Hz
        p.Y += p.VY * 0.016
        phys.Set(p)
    }, []string{"physics"}, []string{"physics"})

    // Compute metrics
    pipe.Cell("metrics", func() {
        p := phys.Get()
        m := metrics.Get()
        m.Speed = math.Sqrt(p.VX*p.VX + p.VY*p.VY)
        m.Distance = math.Sqrt(p.X*p.X + p.Y*p.Y)
        metrics.Set(m)
    }, []string{"physics"}, []string{"metrics"})

    // Periodic logging
    tick := 0
    pipe.OnIteration(func() {
        tick++
        if tick%1000 == 0 {
            m := metrics.Get()
            fmt.Printf("[%d] speed=%.2f dist=%.2f\n", tick, m.Speed, m.Distance)
        }
    })

    // Run for 10 seconds
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    pipe.RunContinuous(ctx)
}
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — Memory model, arena design, scheduling internals
- [BUILDER_API.md](BUILDER_API.md) — Alternative builder API with `Context` struct
- [MODULES.md](MODULES.md) — Built-in modules (audio, HTTP, math, time, etc.)
- [EXAMPLES.md](EXAMPLES.md) — Runnable example applications
