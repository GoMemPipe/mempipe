# AI Inference Example

Zero-allocation MNIST MLP inference using the MemPipe inference engine.

## What It Does

1. Programmatically builds a 784→128→10 MLP model (MNIST digit classifier)
2. Serializes to `.mpmodel` format and loads it back
3. Runs byte-based inference (copies input/output)
4. Runs **zero-copy tensor inference** (0 allocations, direct arena access)
5. Benchmarks throughput (100K inferences)

## Run

```bash
go run ./examples/ai_inference
```

## Key APIs Demonstrated

- `inference.SerializeModel(model)` — create .mpmodel bytes
- `inference.LoadModelFromBytes(data)` — parse model
- `inference.NewEngine(model)` — compile engine, single arena allocation
- `engine.Infer(bytes)` — byte-based I/O
- `engine.InferTensor()` — zero-copy tensor I/O (hot path, 0 allocs)
- `engine.InputTensors()[0].Float32s()` — direct arena memory access
