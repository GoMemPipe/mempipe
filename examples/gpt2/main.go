// Command gpt2 runs GPT-2 inference on a converted .mpmodel file with zero
// heap allocations on the hot path.
//
// Convert a GPT-2 ONNX model first:
//
//	mempipe-convert onnx --transformer gpt2.onnx -o gpt2.mpmodel
//
// Then run inference:
//
//	go run ./examples/gpt2 -model gpt2.mpmodel -prompt "The quick brown fox"
//	go run ./examples/gpt2 -model gpt2.mpmodel -tokens 464,2068,7586
//	go run ./examples/gpt2 -model gpt2.mpmodel -prompt "Hello world" -n 50 -temp 0.8
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GoMemPipe/mempipe/inference"
)

func main() {
	// ── CLI flags ──────────────────────────────────────────────────────
	modelPath := flag.String("model", "gpt2.mpmodel", "Path to converted .mpmodel file")
	prompt := flag.String("prompt", "", "Text prompt (byte-level encoding; for proper BPE use -tokens)")
	tokenStr := flag.String("tokens", "", "Comma-separated token IDs (e.g. 15496,995)")
	maxTokens := flag.Int("n", 32, "Number of tokens to generate")
	temperature := flag.Float64("temp", 1.0, "Sampling temperature (0 = greedy argmax)")
	topK := flag.Int("topk", 0, "Top-k sampling (0 = disabled, use argmax or full softmax)")
	seed := flag.Int64("seed", 0, "Random seed (0 = time-based)")
	verbose := flag.Bool("v", false, "Verbose: print per-step timing and token IDs")
	flag.Parse()

	if *prompt == "" && *tokenStr == "" {
		fmt.Fprintln(os.Stderr, "error: provide -prompt or -tokens")
		flag.Usage()
		os.Exit(1)
	}

	// ── Step 1: Load converted GPT-2 model ────────────────────────────
	fmt.Fprintf(os.Stderr, "Loading model: %s\n", *modelPath)
	model, err := inference.LoadModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load model: %v\n", err)
		os.Exit(1)
	}

	inShape := model.Metadata.InputShapes[0]
	outShape := model.Metadata.OutputShapes[0]
	seqLen := inShape.Dims[len(inShape.Dims)-1] // last dim = sequence length
	vocabSize := outShape.Dims[len(outShape.Dims)-1]

	fmt.Fprintf(os.Stderr, "Model:     %s\n", model.Metadata.Name)
	fmt.Fprintf(os.Stderr, "SeqLen:    %d\n", seqLen)
	fmt.Fprintf(os.Stderr, "Vocab:     %d\n", vocabSize)
	fmt.Fprintf(os.Stderr, "Weights:   %.1f MB\n", float64(model.WeightsSize())/(1024*1024))
	fmt.Fprintf(os.Stderr, "Graph ops: %d\n", len(model.Graph))

	// ── Step 2: Create zero-alloc engine ──────────────────────────────
	engine, err := inference.NewEngine(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine init failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Arena:     %.1f MB used / %.1f MB total\n",
		float64(engine.ArenaUsed())/(1024*1024),
		float64(engine.ArenaTotal())/(1024*1024))
	fmt.Fprintln(os.Stderr)

	// ── Step 3: Prepare input tokens ──────────────────────────────────
	var tokens []int32
	if *tokenStr != "" {
		// Parse comma-separated token IDs
		for _, s := range strings.Split(*tokenStr, ",") {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			id, err := strconv.ParseInt(s, 10, 32)
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid token ID %q: %v\n", s, err)
				os.Exit(1)
			}
			tokens = append(tokens, int32(id))
		}
	} else {
		// Byte-level encoding: each byte of the UTF-8 string becomes a token ID.
		// NOTE: GPT-2 uses BPE tokenization. For accurate results, pre-tokenize
		// with tiktoken or the HF tokenizer and pass IDs via -tokens.
		for _, b := range []byte(*prompt) {
			tokens = append(tokens, int32(b))
		}
		fmt.Fprintf(os.Stderr, "NOTE: Using byte-level encoding (%d tokens).\n", len(tokens))
		fmt.Fprintf(os.Stderr, "      For proper BPE tokens, use -tokens with pre-tokenized IDs.\n")
		fmt.Fprintln(os.Stderr)
	}

	if len(tokens) > seqLen {
		fmt.Fprintf(os.Stderr, "warning: prompt (%d tokens) exceeds seq_len (%d), truncating\n",
			len(tokens), seqLen)
		tokens = tokens[len(tokens)-seqLen:]
	}

	// ── Step 4: Autoregressive generation (zero-alloc hot path) ──────
	rng := rand.New(rand.NewSource(*seed))
	if *seed == 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	inputTensor := engine.InputTensors()[0]
	inputSlice := inputTensor.Int32s()
	generated := make([]int32, 0, *maxTokens)

	fmt.Fprintf(os.Stderr, "Generating %d tokens...\n", *maxTokens)
	totalStart := time.Now()

	for step := 0; step < *maxTokens; step++ {
		// Pad input: right-align the context window
		contextLen := len(tokens)
		if contextLen > seqLen {
			// Sliding window: keep last seqLen tokens
			tokens = tokens[contextLen-seqLen:]
			contextLen = seqLen
		}

		// Zero-fill then copy tokens into input tensor
		for i := range inputSlice {
			inputSlice[i] = 0
		}
		// Left-align tokens in the input
		copy(inputSlice[:contextLen], tokens)

		stepStart := time.Now()

		// ── HOT PATH: zero-alloc inference ──
		outputs, err := engine.InferTensor()
		if err != nil {
			fmt.Fprintf(os.Stderr, "\ninference error at step %d: %v\n", step, err)
			os.Exit(1)
		}

		stepElapsed := time.Since(stepStart)

		// Extract logits at the last filled position
		logits := outputs[0].Float32s()
		// logits shape: [1, seqLen, vocabSize] or [seqLen, vocabSize]
		lastPos := contextLen - 1
		rowStart := lastPos * vocabSize
		rowEnd := rowStart + vocabSize
		posLogits := logits[rowStart:rowEnd]

		// Select next token
		var nextToken int32
		if *temperature == 0 {
			nextToken = int32(argmax(posLogits))
		} else {
			nextToken = int32(sampleTopK(posLogits, float32(*temperature), *topK, rng))
		}

		if *verbose {
			fmt.Fprintf(os.Stderr, "  step %3d: token=%5d  (%.2f ms)\n",
				step, nextToken, float64(stepElapsed.Microseconds())/1000.0)
		}

		tokens = append(tokens, nextToken)
		generated = append(generated, nextToken)
	}

	totalElapsed := time.Since(totalStart)
	tokPerSec := float64(*maxTokens) / totalElapsed.Seconds()

	fmt.Fprintln(os.Stderr)
	fmt.Fprintf(os.Stderr, "Generated %d tokens in %.2f s (%.1f tok/s)\n",
		*maxTokens, totalElapsed.Seconds(), tokPerSec)
	fmt.Fprintln(os.Stderr)

	// ── Step 5: Output ────────────────────────────────────────────────
	// Print generated token IDs to stderr
	fmt.Fprintf(os.Stderr, "Token IDs: ")
	for i, t := range generated {
		if i > 0 {
			fmt.Fprint(os.Stderr, ",")
		}
		fmt.Fprintf(os.Stderr, "%d", t)
	}
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr)

	// Print byte-decoded text to stdout (best-effort for byte-level tokens)
	fmt.Fprintf(os.Stderr, "Output (byte-decoded):\n")
	var buf []byte
	for _, t := range generated {
		if t >= 0 && t < 256 {
			buf = append(buf, byte(t))
		} else {
			// Non-byte token: print placeholder
			buf = append(buf, '<')
			buf = append(buf, []byte(strconv.Itoa(int(t)))...)
			buf = append(buf, '>')
		}
	}
	fmt.Println(string(buf))
}

// ────────────────────────────────────────────────────────────────────────────
// Sampling helpers
// ────────────────────────────────────────────────────────────────────────────

// argmax returns the index of the largest value.
func argmax(values []float32) int {
	best := 0
	bestVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestVal {
			bestVal = values[i]
			best = i
		}
	}
	return best
}

// sampleTopK applies temperature scaling and optional top-k filtering,
// then samples from the resulting probability distribution.
func sampleTopK(logits []float32, temp float32, k int, rng *rand.Rand) int {
	n := len(logits)

	// Apply temperature
	if temp != 1.0 {
		invT := 1.0 / temp
		for i := range logits {
			logits[i] *= invT
		}
	}

	// Stable softmax
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i := range logits {
		logits[i] = float32(math.Exp(float64(logits[i] - maxVal)))
		sum += logits[i]
	}
	invSum := 1.0 / sum
	for i := range logits {
		logits[i] *= invSum
	}

	// Top-k filtering: zero out all but the top-k probabilities
	if k > 0 && k < n {
		// Find the k-th largest value via partial selection
		threshold := kthLargest(logits, k)
		for i := range logits {
			if logits[i] < threshold {
				logits[i] = 0
			}
		}
		// Renormalize
		sum = 0
		for _, v := range logits {
			sum += v
		}
		invSum = 1.0 / sum
		for i := range logits {
			logits[i] *= invSum
		}
	}

	// Sample from categorical distribution
	r := rng.Float32()
	var cumulative float32
	for i, p := range logits {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return n - 1
}

// kthLargest returns the k-th largest value in a slice using a simple
// partial sort. For vocab sizes ≤ 50K this is fast enough.
func kthLargest(values []float32, k int) float32 {
	// Collect top-k in a min-heap-like approach (simple slice for small k)
	topK := make([]float32, 0, k)
	minIdx := 0
	for _, v := range values {
		if len(topK) < k {
			topK = append(topK, v)
			if v < topK[minIdx] {
				minIdx = len(topK) - 1
			}
		} else if v > topK[minIdx] {
			topK[minIdx] = v
			// Find new min
			minIdx = 0
			for j := 1; j < len(topK); j++ {
				if topK[j] < topK[minIdx] {
					minIdx = j
				}
			}
		}
	}
	return topK[minIdx]
}

// tokensToBytes converts int32 token IDs to little-endian bytes.
func tokensToBytes(tokens []int32) []byte {
	buf := make([]byte, len(tokens)*4)
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(t))
	}
	return buf
}
