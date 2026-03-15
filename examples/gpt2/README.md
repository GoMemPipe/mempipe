# GPT-2 Inference with MemPipe

Run a real GPT-2 model with zero heap allocations on the hot path.

## Prerequisites

Convert a GPT-2 ONNX model to `.mpmodel`:

```bash
# Option A: Use the zoo script (exports + converts automatically)
pip install transformers torch onnx
pip install -e tools/mempipe-convert
python tools/mempipe-convert/zoo/convert_gpt2.py --output gpt2.mpmodel

# Option B: Convert a pre-exported ONNX file
mempipe-convert onnx --transformer gpt2.onnx -o gpt2.mpmodel
```

## Run

```bash
# Greedy generation with a text prompt (byte-level encoding)
go run ./examples/gpt2 -model gpt2.mpmodel -prompt "The quick brown fox" -n 50

# With pre-tokenized BPE token IDs (recommended for accuracy)
go run ./examples/gpt2 -model gpt2.mpmodel -tokens 464,2068,7586 -n 50

# Temperature + top-k sampling
go run ./examples/gpt2 -model gpt2.mpmodel -prompt "Once upon a" -n 100 -temp 0.8 -topk 40

# Deterministic generation with a fixed seed
go run ./examples/gpt2 -model gpt2.mpmodel -tokens 15496,995 -n 32 -temp 0 -seed 42 -v
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | `gpt2.mpmodel` | Path to converted `.mpmodel` file |
| `-prompt` | | Text prompt (byte-level encoding) |
| `-tokens` | | Comma-separated BPE token IDs |
| `-n` | `32` | Number of tokens to generate |
| `-temp` | `1.0` | Sampling temperature (0 = greedy argmax) |
| `-topk` | `0` | Top-k sampling (0 = disabled) |
| `-seed` | `0` | Random seed (0 = time-based) |
| `-v` | `false` | Verbose per-step timing |

## Tokenization

GPT-2 uses byte-pair encoding (BPE). The `-prompt` flag uses a simple byte-level
fallback (each UTF-8 byte becomes a token ID), which works for basic ASCII but
won't match GPT-2's actual vocabulary. For accurate results:

1. Tokenize your prompt with [tiktoken](https://github.com/openai/tiktoken) or
   the Hugging Face tokenizer
2. Pass the resulting IDs via `-tokens`

```python
# Example: get GPT-2 token IDs with tiktoken
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("The quick brown fox")
print(",".join(str(i) for i in ids))  # 464,2068,7586,21831
```

## Architecture

The inference engine:

1. Loads the `.mpmodel` (header → metadata → graph → weights) via `inference.LoadModel()`
2. Allocates a single contiguous arena for all weights + activations
3. Compiles the operator graph (Gather, LayerNorm, MatMul, GELU, etc.)
4. Runs autoregressive generation with **zero heap allocations** per step

All tensor data lives in pre-allocated arena memory. The `InferTensor()` call
executes the full forward pass without touching the GC.
