# HFT Pipeline Example

High-frequency trading signal pipeline using MemPipe's zero-allocation engine.

## Pipeline

```
market_data → signal_compute → order_gen
```

- **market_data**: Simulates a price feed with geometric Brownian motion
- **signal_compute**: SMA, EMA, rolling σ, z-score, momentum via ring buffer
- **order_gen**: Mean-reversion strategy — buy when z < -1.5, sell when z > +1.5

## Run

```bash
go run ./examples/hft_pipeline
```

## Key APIs Demonstrated

- `AddRegion[T]()` — market data, signals, orders, P&L as typed regions
- `pipe.Cell()` — dependent cells with topological ordering
- Ring-buffer pattern for windowed time-series computation
- Zero-allocation per-tick processing
