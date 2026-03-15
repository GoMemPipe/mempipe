# Module System

MemPipe includes a modular subsystem for extending pipeline capabilities with
reusable, platform-aware components. Modules register themselves into a global
registry and can participate in the pipeline lifecycle.

## Module Interface

```go
// Every module must implement this minimal interface
type Module interface {
    Name() string
    Init() error
}

// Optional: called every scheduler tick
type Ticker interface {
    Tick(tickCount uint64)
}

// Optional: called during graceful shutdown
type Shutdowner interface {
    Shutdown() error
}
```

## Registry

The `module.Registry` is a thread-safe container for modules, protected by a
platform-adaptive `rwMutex` (real `sync.RWMutex` on native/WASM, no-op on embedded).

```go
import "github.com/GoMemPipe/mempipe/module"

// Create a custom registry
registry := module.NewRegistry()
registry.Register(myModule)
mod, ok := registry.Get("my_module")
names := registry.List()
tickers := registry.Tickers()  // all modules implementing Ticker
registry.Shutdown()             // calls Shutdown() on all Shutdowner modules
```

### Global Registry

A package-level singleton is available for convenience. All built-in modules
auto-register via `init()`:

```go
module.Register(myModule)              // register into global
mod, ok := module.Get("my_module")     // lookup from global
names := module.List()                 // all registered names
reg := module.GetGlobalRegistry()      // access the registry itself
```

### Generic Lookup

Type-safe module retrieval using Go 1.18+ generics:

```go
// Panics if not found or wrong type
audio := module.MustGet[*audio.AudioModule]("audio")

// Returns (value, ok)
audio, ok := module.Lookup[*audio.AudioModule]("audio")
```

## Built-in Modules

### Audio Module (`module/audio`)

Zero-allocation audio generation and DSP, operating directly on arena memory
via `unsafe.Pointer` arithmetic.

```go
import "github.com/GoMemPipe/mempipe/module/audio"

// Default 44100 Hz module auto-registered as "audio"
mod := audio.NewAudioModule(48000, 12345)  // sample rate, PRNG seed

// Attach to a vecf32 region field
mod.Attach(region, "samples")

// Signal generation (zero-alloc, writes directly to arena)
mod.SetFrequency(440.0)    // A4
mod.GenSine(256)            // 256 sine samples
mod.GenNoise(256, 0.5)     // white noise, amplitude 0.5
mod.GenSilence(256)         // zero-fill

// Zero-copy access
ptr, length := mod.PullFramePtr()  // unsafe.Pointer + count
mod.ReadSamples(dst)               // copy from arena → []float32
mod.WriteSamples(src)              // copy from []float32 → arena
```

#### DSP Filters

All filters operate **in-place** on arena memory — zero allocations:

```go
// First-order IIR low-pass filter
mod.LowPassFilter(count, cutoffHz)

// First-order IIR high-pass filter
mod.HighPassFilter(count, cutoffHz)

// Scalar gain (multiply all samples by factor)
mod.Gain(count, 0.8)

// Weighted mix from another region: dst = dst*(1-w) + src*w
mod.Mix(count, srcRegion, "src_field", 0.5)
```

#### State Management

```go
mod.ResetPhase()           // reset oscillator phase to 0
mod.ResetPRNG(newSeed)     // reset noise generator seed
phase := mod.GetPhase()    // current oscillator phase
state := mod.GetPRNGState() // current PRNG state
```

#### Deterministic Noise

The noise generator uses a **xorshift64** PRNG seeded at construction time.
Same seed → same output, guaranteed. Ideal for reproducible audio tests.

---

### HTTP Module (`module/http`)

Platform-aware HTTP client with three implementations selected by build tags:

| Platform | Implementation | Features |
|----------|---------------|----------|
| **Native** (`!wasm && !embedded`) | `net/http.Client` | Full HTTP, 30s default timeout |
| **WASM** (`wasm`) | JavaScript `fetch` API via `syscall/js` | GET/POST with Promise-based blocking |
| **Embedded** (`embedded`) | No-op stub | All methods return `ErrUnavailable` |

```go
import mhttp "github.com/GoMemPipe/mempipe/module/http"

// Auto-registered as "http" — correct impl selected by build tag
resp := mhttp.Get("https://api.example.com/data")
if resp.Error != nil {
    // handle error
}
fmt.Println(resp.StatusCode, resp.Body)

resp = mhttp.Post("https://api.example.com/data", jsonBody, "application/json")

// Configure timeout
mhttp.SetTimeout(5000) // 5 seconds

// Health check
ok := mhttp.HealthCheck("https://api.example.com/health")
```

#### Response Type

```go
type Response struct {
    StatusCode   int
    Body         string
    Headers      map[string]string
    ResponseTime float64  // milliseconds
    Error        error
}
```

---

### I/O Module (`module/io`)

Pure memory-based I/O — all output goes to `runtime.MemoryPipe` ring buffers,
never to OS stdout/stderr. This ensures deterministic, replayable output on
all platforms.

```go
import mio "github.com/GoMemPipe/mempipe/module/io"

// Auto-registered as "io" with 64KB stdout/stderr pipes
io := mio.GetGlobalIOModule()

// Write to memory-native stdout
io.Write("hello")
io.Writeln("world")
io.Println("formatted", "output")
io.Printf("temp=%.2f\n", 22.5)

// Write to memory-native stderr
io.Eprint("warning: ")
io.Eprintln("high latency")

// Read captured output
stdout := io.ReadStdout()
stderr := io.ReadStderr()

// Named pipes for custom channels
io.CreatePipe("events", 4096)
io.WritePipe("events", "sensor triggered")
lines := io.ReadLines("events")
io.ClosePipe("events")

// Replace pipe manager (e.g., connect to scheduler's manager)
io.SetPipeManager(scheduler.PipeManager())
```

---

### Math Module (`module/math`)

Zero-allocation math functions — thin wrappers around the Go `math` standard
library plus extra utilities.

```go
import mmath "github.com/GoMemPipe/mempipe/module/math"

// Trigonometric
mmath.Sin(x)
mmath.Cos(x)
mmath.Tan(x)

// Exponential / logarithmic
mmath.Exp(x)
mmath.Log(x)
mmath.Log10(x)
mmath.Pow(base, exp)
mmath.Sqrt(x)

// Rounding
mmath.Floor(x)
mmath.Ceil(x)
mmath.Round(x)

// Comparison
mmath.Abs(x)
mmath.Min(a, b)
mmath.Max(a, b)
mmath.Clamp(x, lo, hi)  // constrain x to [lo, hi]

// Integer utilities
mmath.Factorial(n)       // int64 factorial
mmath.GCD(a, b)          // Euclidean GCD
mmath.LCM(a, b)          // least common multiple via GCD
```

All functions take and return concrete types (`float64` or `int64`) — no
`interface{}` boxing.

---

### Sys Module (`module/sys`)

System information from the MemPipe runtime — **not** from the host OS.

```go
import "github.com/GoMemPipe/mempipe/module/sys"

mod := sys.GetGlobalSysModule()

mod.Version()              // "1.0.0-dev"
mod.Arch()                 // "bare"
mod.Ticks()                // current clock ticks
mod.UptimeMs()             // ms since module init
mod.Info()                 // human-readable summary string

// Arena stats (for diagnostics)
stats := mod.ArenaStats(arena)
fmt.Printf("Total: %d, Used: %d, Fragmentation: %.2f%%\n",
    stats.TotalSize, stats.UsedSize, stats.FragmentationRatio*100)

// Set clock reference (called during runtime init)
mod.SetClock(scheduler.Clock())
```

---

### Time Module (`module/time`)

Deterministic tick-based time — fully OS-independent and reproducible. Same
ticks-per-ms rate + same inputs = same time values on every platform.

```go
import mtime "github.com/GoMemPipe/mempipe/module/time"

mod := mtime.GetGlobalTimeModule()

now := mod.Now()                // current tick count
ms := mod.Ms()                  // ticks → milliseconds
elapsed := mod.Since(startTick) // ticks since startTick
uptime := mod.Elapsed()         // ticks since module init

mod.Sleep(100)                  // no-op placeholder (future: cooperative yield)

// Set clock reference (called during runtime init)
mod.SetClock(scheduler.Clock())
```

## Module Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│  init()  │────►│Register()│────►│    Init()     │────►│  Tick()  │
│          │     │          │     │  (one-time)   │     │ (per tick)│
│ auto via │     │ adds to  │     │ returns error │     │ optional  │
│ import   │     │ registry │     │ if setup fails│     │           │
└──────────┘     └──────────┘     └──────────────┘     └──────────┘
                                                             │
                                                             ▼
                                                       ┌──────────┐
                                                       │Shutdown()│
                                                       │ optional  │
                                                       └──────────┘
```

1. **`init()`**: Built-in modules register themselves at import time
2. **`Register(m)`**: Adds module to registry, immediately calls `m.Init()`
3. **`Tick(n)`**: Called by the scheduler each iteration (if module implements `Ticker`)
4. **`Shutdown()`**: Called during graceful teardown (if module implements `Shutdowner`)

## Deferred Clock Injection

The `time` and `sys` modules start with a `nil` clock reference (since they
register in `init()` before the scheduler exists). The clock is injected
later during runtime initialization:

```go
scheduler := runtime.NewScheduler()
sys.GetGlobalSysModule().SetClock(scheduler.Clock())
mtime.GetGlobalTimeModule().SetClock(scheduler.Clock())
```

This pattern allows modules to be imported and registered before the pipeline
is constructed, while still accessing the scheduler's deterministic clock.

## Platform Compilation Matrix

| Module | Native | WASM | Embedded |
|--------|--------|------|----------|
| `module/audio` | ✅ | ✅ | ✅ |
| `module/http` (native) | ✅ | ❌ | ❌ |
| `module/http` (wasm) | ❌ | ✅ | ❌ |
| `module/http` (embedded) | ❌ | ❌ | ✅ |
| `module/io` | ✅ | ✅ | ✅ |
| `module/math` | ✅ | ✅ | ✅ |
| `module/sys` | ✅ | ✅ | ✅ |
| `module/time` | ✅ | ✅ | ✅ |

## Writing Custom Modules

```go
package mymodule

import "github.com/GoMemPipe/mempipe/module"

type MyModule struct {
    count uint64
}

func (m *MyModule) Name() string  { return "my_module" }
func (m *MyModule) Init() error   { return nil }
func (m *MyModule) Tick(n uint64) { m.count++ }
func (m *MyModule) Shutdown() error {
    fmt.Printf("processed %d ticks\n", m.count)
    return nil
}

func init() {
    module.Register(&MyModule{})
}
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — Module system design and registry internals
- [PIPELINE_API.md](PIPELINE_API.md) — Using modules within pipeline cells
- [CONDITIONAL_COMPILATION.md](CONDITIONAL_COMPILATION.md) — Platform-specific module variants
