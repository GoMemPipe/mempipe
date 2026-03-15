# Builder API Reference

The `builder` package provides a high-level, imperative API for defining and
running MemPipe pipelines. It offers a `Context` struct with typed field
accessors, making it ideal for dynamic pipelines where the region layout
is defined at runtime rather than via struct tags.

## Overview

```go
import "github.com/GoMemPipe/mempipe/builder"

pipe := builder.NewPipeline()

pipe.Region("sensor", builder.Fields{
    "temperature": builder.Float64,
    "humidity":    builder.Float64,
    "count":       builder.Uint32,
})

pipe.Region("output", builder.Fields{
    "avg_temp": builder.Float64,
})

pipe.Cell("process", func(ctx *builder.Context) {
    temp := ctx.F64("sensor", "temperature")
    count := ctx.U32("sensor", "count")
    ctx.SetF64("sensor", "temperature", temp+0.1)
    ctx.SetU32("sensor", "count", count+1)
    ctx.SetF64("output", "avg_temp", temp/float64(count+1))
})

pipe.Continuous(1000)
pipe.Run()
```

## Pipeline Construction

### NewPipeline

```go
func NewPipeline() *Pipeline
```

Creates a new builder pipeline.

### Region

```go
func (p *Pipeline) Region(name string, fields Fields) *Pipeline
```

Defines a named region with explicit field types. The `Fields` type is
`map[string]FieldType`.

```go
pipe.Region("market_data", builder.Fields{
    "price":    builder.Float64,
    "volume":   builder.Uint64,
    "bid":      builder.Float32,
    "ask":      builder.Float32,
    "sequence": builder.Uint32,
    "active":   builder.Bool,
})
```

Returns `*Pipeline` for method chaining.

### Field Types

| Constant | Go Type | Arena Type | Size |
|----------|---------|------------|------|
| `builder.Uint8` | `uint8` | `u8` | 1B |
| `builder.Uint16` | `uint16` | `u16` | 2B |
| `builder.Uint32` | `uint32` | `u32` | 4B |
| `builder.Uint64` | `uint64` | `u64` | 8B |
| `builder.Int8` | `int8` | `i8` | 1B |
| `builder.Int16` | `int16` | `i16` | 2B |
| `builder.Int32` | `int32` | `i32` | 4B |
| `builder.Int64` | `int64` | `i64` | 8B |
| `builder.Float32` | `float32` | `f32` | 4B |
| `builder.Float64` | `float64` | `f64` | 8B |
| `builder.Bool` | `bool` | `bool` | 1B |

### Cell

```go
func (p *Pipeline) Cell(name string, fn func(ctx *Context)) *Pipeline
```

Registers a processing cell. The function receives a `*Context` that provides
typed accessors for reading and writing region fields.

```go
pipe.Cell("compute", func(ctx *builder.Context) {
    x := ctx.F64("input", "x")
    y := ctx.F64("input", "y")
    ctx.SetF64("output", "sum", x+y)
    ctx.SetF64("output", "product", x*y)
})
```

### Continuous

```go
func (p *Pipeline) Continuous(n int) *Pipeline
```

Sets the number of iterations to run.

```go
pipe.Continuous(10000) // 10K iterations
```

### Run

```go
func (p *Pipeline) Run() error
```

Builds the pipeline (if not already built) and executes all iterations.
Internally:
1. Collects region specs and computes layouts
2. Creates a single `runtime.RegionArena` (one allocation)
3. Creates a `runtime.Scheduler`
4. Loops through iterations, calling each cell with a `Context`

### Stdout / Stderr

```go
func (p *Pipeline) Stdout() string
func (p *Pipeline) Stderr() string
```

Returns the captured memory-native stdout/stderr output after execution.

## Context

The `Context` struct is passed to every cell function on each iteration. It
provides **zero-allocation typed accessors** for reading and writing region fields.

### Typed Readers

All readers are annotated `//mem:hot //mem:nogc` — zero allocations.

```go
// Unsigned integers
val := ctx.U8(region, field)   // uint8
val := ctx.U16(region, field)  // uint16
val := ctx.U32(region, field)  // uint32
val := ctx.U64(region, field)  // uint64

// Signed integers
val := ctx.I8(region, field)   // int8
val := ctx.I16(region, field)  // int16
val := ctx.I32(region, field)  // int32
val := ctx.I64(region, field)  // int64

// Floating point
val := ctx.F32(region, field)  // float32
val := ctx.F64(region, field)  // float64

// Boolean
val := ctx.Bool(region, field) // bool
```

### Typed Writers

All writers are annotated `//mem:hot //mem:nogc` — zero allocations.

```go
ctx.SetU8(region, field, val)
ctx.SetU16(region, field, val)
ctx.SetU32(region, field, val)
ctx.SetU64(region, field, val)
ctx.SetI8(region, field, val)
ctx.SetI16(region, field, val)
ctx.SetI32(region, field, val)
ctx.SetI64(region, field, val)
ctx.SetF32(region, field, val)
ctx.SetF64(region, field, val)
ctx.SetBool(region, field, val)
```

### Iteration Counter

```go
iter := ctx.Iteration() // current iteration number (0-based)
```

### Local Variables

The context provides per-cell local variable storage that persists across
iterations but is scoped to the current cell execution:

```go
ctx.SetVar("running_sum", currentSum)
sum, ok := ctx.GetVar("running_sum")
if ok {
    total := sum.(float64) + newValue
}
```

**Note**: The local variable map is cleared (without reallocating) between
iterations to prevent memory leaks. Values persist only within a single
iteration's cell execution.

### Output

```go
ctx.Println("processing tick", ctx.Iteration())
ctx.Printf("temp=%.2f humidity=%.2f\n", temp, humidity)
```

Both write to memory-native stdout via the global `IOModule`. Output is
captured and retrievable via `pipe.Stdout()`.

## Pipeline vs Builder API

The `mempipe` (pipeline) and `builder` packages serve the same purpose but
with different trade-offs:

| Feature | `mempipe` (Pipeline API) | `builder` (Builder API) |
|---------|--------------------------|-------------------------|
| **Region definition** | Go struct with `mempipe` tags | `Fields` map at runtime |
| **Type safety** | Compile-time via generics `[T]` | Runtime via `Context` accessors |
| **Data access** | `handle.Get()` / `handle.Set()` (whole struct) | `ctx.F64("name", "field")` (per-field) |
| **Dependencies** | Explicit `Cell(name, fn, inputs, outputs)` | Implicit (registration order) |
| **Scheduling** | Topological with parallel option | Sequential |
| **Best for** | Static pipelines with known struct layouts | Dynamic pipelines, prototyping, scripting |
| **Overhead** | Slightly lower (resolved at build time) | Slightly higher (map lookup per field access) |

### When to Use Which

**Use the Pipeline API** when:
- Your region structs are known at compile time
- You need parallel cell scheduling
- You want compile-time type safety
- Performance is critical

**Use the Builder API** when:
- Region layouts are determined at runtime (e.g., from config files)
- You're prototyping or experimenting
- You want simpler per-field access without struct boilerplate
- You're building a scripting or DSL layer on top of MemPipe

## Complete Example

```go
package main

import (
    "fmt"
    "github.com/GoMemPipe/mempipe/builder"
)

func main() {
    pipe := builder.NewPipeline()

    pipe.Region("input", builder.Fields{
        "a": builder.Float64,
        "b": builder.Float64,
    })

    pipe.Region("output", builder.Fields{
        "sum":     builder.Float64,
        "product": builder.Float64,
        "count":   builder.Uint64,
    })

    // Initialize inputs
    pipe.Cell("init", func(ctx *builder.Context) {
        iter := ctx.Iteration()
        ctx.SetF64("input", "a", float64(iter)*1.5)
        ctx.SetF64("input", "b", float64(iter)*0.7+1.0)
    })

    // Compute outputs
    pipe.Cell("compute", func(ctx *builder.Context) {
        a := ctx.F64("input", "a")
        b := ctx.F64("input", "b")
        count := ctx.U64("output", "count")

        ctx.SetF64("output", "sum", a+b)
        ctx.SetF64("output", "product", a*b)
        ctx.SetU64("output", "count", count+1)

        if ctx.Iteration()%100 == 0 {
            ctx.Printf("[%d] a=%.2f b=%.2f sum=%.2f\n",
                ctx.Iteration(), a, b, a+b)
        }
    })

    pipe.Continuous(1000)

    if err := pipe.Run(); err != nil {
        fmt.Printf("error: %v\n", err)
    }

    fmt.Println(pipe.Stdout())
}
```

## Benchmarks

Builder API benchmarks demonstrate zero-allocation steady-state performance:

```
BenchmarkBuilderContextU32Read          0 allocs/op
BenchmarkBuilderContextU32Write         0 allocs/op
BenchmarkBuilderContextF64Read          0 allocs/op
BenchmarkBuilderContextF64Write         0 allocs/op
BenchmarkBuilderContextBoolReadWrite    0 allocs/op
BenchmarkBuilderRealisticWorkload       0 allocs/op   (2 reads + compute + 2 writes)
BenchmarkBuilderMultiFieldAccess        0 allocs/op   (4 U32 fields)

BenchmarkMemFlowSingleIteration         0 B/op  0 allocs/op  (3-stage Hebbian pipeline)
BenchmarkMemFlowComputeOnly            ~68 ns/op              (pure math baseline)
```

## See Also

- [PIPELINE_API.md](PIPELINE_API.md) — Generic pipeline API with struct tags
- [ARCHITECTURE.md](ARCHITECTURE.md) — Memory model and runtime internals
- [MODULES.md](MODULES.md) — Built-in modules accessible from cells
