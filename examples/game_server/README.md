# Game Server Example

Tick-based 2D game simulation using MemPipe's pipeline and typed regions.

## Architecture

```
input_processor → physics → collision → state_sync
```

- **input_processor**: Simulates player commands (sinusoidal acceleration)
- **physics**: Euler integration, wall bouncing, drag
- **collision**: O(n²) circle collision + elastic response
- **state_sync**: Aggregate metrics (alive count, avg speed, collision totals)

All entity state and game metrics live in arena-backed typed regions.

## Run

```bash
go run ./examples/game_server
```

## Key APIs Demonstrated

- `AddRegion[T]()` with game state structs
- `pipe.Cell()` with explicit dependency ordering
- `pipe.OnIteration()` for tick-based monitoring
- Multi-cell pipeline with topological execution order
