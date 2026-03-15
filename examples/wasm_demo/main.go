//go:build js && wasm

// Command wasm_demo provides an interactive browser-based demonstration
// of MemPipe WASM capabilities:
//
//	Demo 1: Basic MemPipe Features — arena, tensors, operators, benchmarks
//	Demo 2: GPT-2 Inference — loads gpt2.mpmodel, runs autoregressive generation
//
// Build:
//
//	GOOS=js GOARCH=wasm go build -o demo.wasm ./examples/wasm_demo/
//
// Serve:
//
//	python3 -m http.server 8080   (from this directory after build)
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"syscall/js"
	"time"
	"unsafe"

	"github.com/GoMemPipe/mempipe/build"
	"github.com/GoMemPipe/mempipe/inference"
)

// ────────────────────────────────────────────────────────────────────────────
// Globals
// ────────────────────────────────────────────────────────────────────────────

var (
	gptModel  *inference.Model
	gptEngine *inference.Engine
)

// ════════════════════════════════════════════════════════════════════════════
// Demo 1 — Basic MemPipe Features
// ════════════════════════════════════════════════════════════════════════════

// jsDemo1Info returns basic MemPipe build information.
func jsDemo1Info(_ js.Value, _ []js.Value) interface{} {
	result := js.Global().Get("Object").New()
	result.Set("version", build.Version)
	result.Set("platform", build.Platform())
	result.Set("hasHTTP", build.HasHTTP())
	result.Set("hasFilesystem", build.HasFilesystem())
	result.Set("hasConcurrency", build.HasConcurrency())
	result.Set("isWebGPUReady", inference.IsWebGPUReady())
	return result
}

// jsDemo1Arena demonstrates arena allocation and returns metrics.
func jsDemo1Arena(_ js.Value, args []js.Value) interface{} {
	arenaSize := 1024 * 1024 // 1 MB default
	if len(args) > 0 && !args[0].IsUndefined() {
		arenaSize = args[0].Int()
	}

	result := js.Global().Get("Object").New()
	start := time.Now()

	arena := inference.NewInferenceArena(arenaSize)

	allocDur := time.Since(start)
	result.Set("arenaSizeBytes", arena.TotalBytes())
	result.Set("allocTimeUs", allocDur.Microseconds())

	// Allocate a few tensors and report
	tensors := js.Global().Get("Array").New()
	shapes := [][2]int{{64, 64}, {128, 32}, {256, 256}, {1, 512}}
	for _, s := range shapes {
		tStart := time.Now()
		t, err := arena.AllocTensor(fmt.Sprintf("t_%dx%d", s[0], s[1]), []int{s[0], s[1]}, inference.Float32)
		tDur := time.Since(tStart)
		entry := js.Global().Get("Object").New()
		entry.Set("shape", fmt.Sprintf("[%d, %d]", s[0], s[1]))
		entry.Set("bytes", s[0]*s[1]*4)
		entry.Set("allocTimeUs", tDur.Microseconds())
		if err != nil {
			entry.Set("error", err.Error())
		} else {
			entry.Set("ptr", int(uintptr(t.DataPtr())))
			entry.Set("error", js.Null())
		}
		tensors.Call("push", entry)
	}
	result.Set("tensors", tensors)
	result.Set("usedBytes", arena.UsedBytes())
	result.Set("totalBytes", arena.TotalBytes())
	return result
}

// jsDemo1Operators runs a set of operators and benchmarks them.
func jsDemo1Operators(_ js.Value, _ []js.Value) interface{} {
	results := js.Global().Get("Array").New()

	// MatMul benchmark at various sizes
	sizes := [][3]int{{4, 4, 4}, {16, 16, 16}, {64, 64, 64}, {128, 128, 128}, {256, 256, 256}}
	for _, s := range sizes {
		m, k, n := s[0], s[1], s[2]
		entry := benchmarkMatMulJS(m, k, n)
		results.Call("push", entry)
	}

	// Activation benchmarks
	results.Call("push", benchmarkActivationJS("ReLU", inference.OpReLU, 4096))
	results.Call("push", benchmarkActivationJS("Sigmoid", inference.OpSigmoid, 4096))
	results.Call("push", benchmarkActivationJS("Softmax", inference.OpSoftmax, 1024))

	return results
}

func benchmarkMatMulJS(m, k, n int) js.Value {
	arenaSize := (m*k + k*n + m*n) * 4 * 2
	arena := inference.NewInferenceArena(arenaSize)
	a, _ := arena.AllocTensor("A", []int{m, k}, inference.Float32)
	b, _ := arena.AllocTensor("B", []int{k, n}, inference.Float32)
	c, _ := arena.AllocTensor("C", []int{m, n}, inference.Float32)

	// Fill with small values
	for i := range a.Float32s() {
		a.Float32s()[i] = 0.01
	}
	for i := range b.Float32s() {
		b.Float32s()[i] = 0.01
	}

	op, _ := inference.GetOperator(inference.OpMatMul)

	// Warmup
	op.Execute([]*inference.Tensor{a, b}, []*inference.Tensor{c})

	// Benchmark: run several iterations
	iters := 100
	if m >= 128 {
		iters = 10
	}
	if m >= 256 {
		iters = 5
	}

	start := time.Now()
	for i := 0; i < iters; i++ {
		op.Execute([]*inference.Tensor{a, b}, []*inference.Tensor{c})
	}
	elapsed := time.Since(start)

	// Verify a value
	sample := c.Float32s()[0]

	entry := js.Global().Get("Object").New()
	entry.Set("op", fmt.Sprintf("MatMul %dx%dx%d", m, k, n))
	entry.Set("iters", iters)
	entry.Set("totalMs", float64(elapsed.Microseconds())/1000.0)
	entry.Set("avgUs", float64(elapsed.Microseconds())/float64(iters))
	entry.Set("flops", float64(2*m*k*n*iters)/elapsed.Seconds())
	entry.Set("sampleOutput", sample)
	entry.Set("allocsPerOp", 0)
	return entry
}

func benchmarkActivationJS(name string, opType inference.OpType, n int) js.Value {
	arenaSize := n * 4 * 3
	arena := inference.NewInferenceArena(arenaSize)

	shape := []int{n}
	if name == "Softmax" {
		shape = []int{n / 10, 10}
	}

	in, _ := arena.AllocTensor("in", shape, inference.Float32)
	out, _ := arena.AllocTensor("out", shape, inference.Float32)

	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i%20) - 10.0
	}

	op, _ := inference.GetOperator(opType)
	op.Execute([]*inference.Tensor{in}, []*inference.Tensor{out})

	iters := 1000
	start := time.Now()
	for i := 0; i < iters; i++ {
		op.Execute([]*inference.Tensor{in}, []*inference.Tensor{out})
	}
	elapsed := time.Since(start)

	entry := js.Global().Get("Object").New()
	entry.Set("op", fmt.Sprintf("%s (%d elements)", name, n))
	entry.Set("iters", iters)
	entry.Set("totalMs", float64(elapsed.Microseconds())/1000.0)
	entry.Set("avgUs", float64(elapsed.Microseconds())/float64(iters))
	entry.Set("flops", float64(n*iters)/elapsed.Seconds())
	entry.Set("sampleOutput", out.Float32s()[0])
	entry.Set("allocsPerOp", 0)
	return entry
}

// jsDemo1ZeroCopy demonstrates zero-copy memory sharing.
func jsDemo1ZeroCopy(_ js.Value, _ []js.Value) interface{} {
	arena := inference.NewInferenceArena(256 * 4 * 3)
	t, _ := arena.AllocTensor("shared", []int{16, 16}, inference.Float32)

	result := js.Global().Get("Object").New()
	result.Set("ptr", int(uintptr(t.DataPtr())))
	result.Set("byteSize", t.ByteSize())
	result.Set("numElements", t.NumElements())
	result.Set("shape", fmt.Sprintf("%v", t.Shape()))
	result.Set("dtype", t.DType().String())

	// Write some data
	data := t.Float32s()
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	result.Set("first5", fmt.Sprintf("[%.1f, %.1f, %.1f, %.1f, %.1f]",
		data[0], data[1], data[2], data[3], data[4]))

	return result
}

// ════════════════════════════════════════════════════════════════════════════
// Demo 2 — GPT-2 Inference with Hardware-Accelerated MatMul
// ════════════════════════════════════════════════════════════════════════════

// jsLoadGPT2 loads the GPT-2 model from a Uint8Array.
func jsLoadGPT2(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("loadGPT2 requires a Uint8Array argument")
	}

	result := js.Global().Get("Object").New()

	arr := args[0]
	length := arr.Get("byteLength").Int()
	result.Set("fileSize", length)

	// Copy model bytes from JS
	start := time.Now()
	data := make([]byte, length)
	js.CopyBytesToGo(data, arr)
	copyDur := time.Since(start)
	result.Set("copyTimeMs", float64(copyDur.Microseconds())/1000.0)

	// Parse model
	parseStart := time.Now()
	model, err := inference.LoadModelFromBytes(data)
	if err != nil {
		result.Set("error", err.Error())
		return result
	}
	parseDur := time.Since(parseStart)
	gptModel = model

	result.Set("parseTimeMs", float64(parseDur.Microseconds())/1000.0)
	result.Set("modelName", model.Metadata.Name)
	result.Set("graphOps", len(model.Graph))
	result.Set("weightsMB", float64(model.WeightsSize())/(1024*1024))
	result.Set("tensorCount", len(model.TensorNames))

	inShape := model.Metadata.InputShapes[0]
	outShape := model.Metadata.OutputShapes[0]
	result.Set("inputShape", fmt.Sprintf("%v", inShape.Dims))
	result.Set("outputShape", fmt.Sprintf("%v", outShape.Dims))
	result.Set("seqLen", inShape.Dims[len(inShape.Dims)-1])
	result.Set("vocabSize", outShape.Dims[len(outShape.Dims)-1])

	// Create engine
	engineStart := time.Now()
	engine, err := inference.NewEngine(model)
	if err != nil {
		result.Set("error", "engine: "+err.Error())
		return result
	}
	engineDur := time.Since(engineStart)
	gptEngine = engine

	result.Set("engineTimeMs", float64(engineDur.Microseconds())/1000.0)
	result.Set("arenaUsedMB", float64(engine.ArenaUsed())/(1024*1024))
	result.Set("arenaTotalMB", float64(engine.ArenaTotal())/(1024*1024))
	result.Set("webgpuActive", inference.IsWebGPUReady())
	result.Set("error", js.Null())

	return result
}

// jsGPT2Generate runs autoregressive token generation.
// Args: tokenIDs (Int32Array or Array of ints), maxTokens (int), temperature (float)
// NOTE: Kept for reference; the worker.js loop calls generateStep instead.
func jsGPT2Generate(_ js.Value, args []js.Value) interface{} {
	if gptEngine == nil {
		return jsError("GPT-2 model not loaded — call loadGPT2 first")
	}
	if len(args) < 3 {
		return jsError("generate requires (tokenIDs, maxTokens, temperature)")
	}

	// Parse input tokens
	tokenArr := args[0]
	numTokens := tokenArr.Get("length").Int()
	tokens := make([]int32, numTokens)
	for i := 0; i < numTokens; i++ {
		tokens[i] = int32(tokenArr.Index(i).Int())
	}

	maxTokens := args[1].Int()
	temperature := float32(args[2].Float())

	model := gptModel
	engine := gptEngine

	inShape := model.Metadata.InputShapes[0]
	outShape := model.Metadata.OutputShapes[0]
	seqLen := inShape.Dims[len(inShape.Dims)-1]
	vocabSize := outShape.Dims[len(outShape.Dims)-1]

	inputTensor := engine.InputTensors()[0]
	inputSlice := inputTensor.Int32s()

	result := js.Global().Get("Object").New()
	generatedTokens := js.Global().Get("Array").New()
	stepTimings := js.Global().Get("Array").New()

	totalStart := time.Now()

	for step := 0; step < maxTokens; step++ {
		contextLen := len(tokens)
		if contextLen > seqLen {
			tokens = tokens[contextLen-seqLen:]
			contextLen = seqLen
		}

		// Zero-fill then copy
		for i := range inputSlice {
			inputSlice[i] = 0
		}
		copy(inputSlice[:contextLen], tokens)

		stepStart := time.Now()
		outputs, err := engine.InferTensor()
		stepElapsed := time.Since(stepStart)

		if err != nil {
			result.Set("error", fmt.Sprintf("inference error step %d: %v", step, err))
			return result
		}

		logits := outputs[0].Float32s()
		lastPos := contextLen - 1
		rowStart := lastPos * vocabSize
		posLogits := logits[rowStart : rowStart+vocabSize]

		var nextToken int32
		if temperature == 0 {
			nextToken = int32(argmax(posLogits))
		} else {
			nextToken = int32(sampleWithTemp(posLogits, temperature))
		}

		tokens = append(tokens, nextToken)
		generatedTokens.Call("push", int(nextToken))

		timing := js.Global().Get("Object").New()
		timing.Set("step", step)
		timing.Set("token", int(nextToken))
		timing.Set("timeMs", float64(stepElapsed.Microseconds())/1000.0)
		stepTimings.Call("push", timing)
	}

	totalElapsed := time.Since(totalStart)

	result.Set("tokens", generatedTokens)
	result.Set("stepTimings", stepTimings)
	result.Set("totalMs", float64(totalElapsed.Microseconds())/1000.0)
	result.Set("tokensPerSec", float64(maxTokens)/totalElapsed.Seconds())
	result.Set("seqLen", seqLen)
	result.Set("vocabSize", vocabSize)
	result.Set("inputTokens", numTokens)
	result.Set("generatedCount", maxTokens)
	result.Set("error", js.Null())

	return result
}

// jsGPT2GenerateStep runs a single forward pass and returns the next token.
// Args: tokenIDs (Array of current token sequence), temperature (float)
// This is called by worker.js in a loop with yields between steps so the
// worker can post progress messages to the main thread.
func jsGPT2GenerateStep(_ js.Value, args []js.Value) interface{} {
	result := js.Global().Get("Object").New()

	if gptEngine == nil {
		result.Set("error", "GPT-2 model not loaded")
		return result
	}
	if len(args) < 2 {
		result.Set("error", "generateStep requires (tokenIDs, temperature)")
		return result
	}

	tokenArr := args[0]
	numTokens := tokenArr.Get("length").Int()
	tokens := make([]int32, numTokens)
	for i := 0; i < numTokens; i++ {
		tokens[i] = int32(tokenArr.Index(i).Int())
	}
	temperature := float32(args[1].Float())

	model := gptModel
	engine := gptEngine

	inShape := model.Metadata.InputShapes[0]
	outShape := model.Metadata.OutputShapes[0]
	seqLen := inShape.Dims[len(inShape.Dims)-1]
	vocabSize := outShape.Dims[len(outShape.Dims)-1]

	contextLen := numTokens
	if contextLen > seqLen {
		offset := contextLen - seqLen
		for i := 0; i < seqLen; i++ {
			tokens[i] = tokens[i+offset]
		}
		tokens = tokens[:seqLen]
		contextLen = seqLen
	}

	inputTensor := engine.InputTensors()[0]
	inputSlice := inputTensor.Int32s()

	for i := range inputSlice {
		inputSlice[i] = 0
	}
	copy(inputSlice[:contextLen], tokens)

	stepStart := time.Now()
	outputs, err := engine.InferTensor()
	stepElapsed := time.Since(stepStart)

	if err != nil {
		result.Set("error", fmt.Sprintf("inference error: %v", err))
		return result
	}

	logits := outputs[0].Float32s()
	lastPos := contextLen - 1
	rowStart := lastPos * vocabSize
	posLogits := logits[rowStart : rowStart+vocabSize]

	var nextToken int32
	if temperature == 0 {
		nextToken = int32(argmax(posLogits))
	} else {
		nextToken = int32(sampleWithTemp(posLogits, temperature))
	}

	result.Set("token", int(nextToken))
	result.Set("timeMs", float64(stepElapsed.Microseconds())/1000.0)
	result.Set("error", js.Null())
	return result
}

// jsGPT2MatMulBench benchmarks the MatMul operator at various sizes.
func jsGPT2MatMulBench(_ js.Value, _ []js.Value) interface{} {
	// Ensure WebGPU is initialised for standalone benchmark calls.
	inference.EnsureWebGPU()

	results := js.Global().Get("Array").New()

	sizes := [][3]int{
		{4, 4, 4}, {16, 16, 16}, {32, 32, 32},
		{64, 64, 64}, {128, 128, 128}, {256, 256, 256},
		{512, 512, 512},
		// GPT-2 typical shapes
		{1, 768, 768}, {1, 768, 3072}, {1, 768, 50257},
	}

	for _, s := range sizes {
		m, k, n := s[0], s[1], s[2]

		arenaSize := (m*k + k*n + m*n) * 4 * 2
		arena := inference.NewInferenceArena(arenaSize)
		a, err1 := arena.AllocTensor("A", []int{m, k}, inference.Float32)
		b, err2 := arena.AllocTensor("B", []int{k, n}, inference.Float32)
		c, err3 := arena.AllocTensor("C", []int{m, n}, inference.Float32)

		if err1 != nil || err2 != nil || err3 != nil {
			entry := js.Global().Get("Object").New()
			entry.Set("size", fmt.Sprintf("%dx%dx%d", m, k, n))
			entry.Set("error", "allocation failed")
			results.Call("push", entry)
			continue
		}

		for i := range a.Float32s() {
			a.Float32s()[i] = 0.01
		}
		for i := range b.Float32s() {
			b.Float32s()[i] = 0.01
		}

		op, _ := inference.GetOperator(inference.OpMatMul)

		// Warmup
		op.Execute([]*inference.Tensor{a, b}, []*inference.Tensor{c})

		iters := 50
		if m*k*n > 100000 {
			iters = 10
		}
		if m*k*n > 1000000 {
			iters = 3
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			op.Execute([]*inference.Tensor{a, b}, []*inference.Tensor{c})
		}
		elapsed := time.Since(start)

		flops := float64(2*int64(m)*int64(k)*int64(n)*int64(iters)) / elapsed.Seconds()

		entry := js.Global().Get("Object").New()
		entry.Set("size", fmt.Sprintf("%dx%dx%d", m, k, n))
		entry.Set("iters", iters)
		entry.Set("totalMs", float64(elapsed.Microseconds())/1000.0)
		entry.Set("avgMs", float64(elapsed.Microseconds())/float64(iters)/1000.0)
		entry.Set("gflops", flops/1e9)
		entry.Set("allocsPerOp", 0)
		results.Call("push", entry)
	}

	return results
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

func jsError(msg string) js.Value {
	return js.Global().Get("Error").New(msg)
}

// goAsync wraps a blocking Go function as a js.FuncOf that returns a JS Promise.
//
// In Go WASM, js.FuncOf callbacks run synchronously on the handleEvent goroutine.
// If the callback blocks (e.g., on a channel for WebGPU dispatch), the
// handleEvent goroutine's event.returned flag never gets set, and beforeIdle()
// returns (nil, false) — the scheduler deadlocks.
//
// goAsync solves this by:
//  1. Creating a JS Promise in the callback
//  2. Starting a new goroutine to do the blocking work
//  3. Returning the Promise immediately — handleEvent completes, event.returned = true
//  4. The goroutine can safely block on channels (WebGPU dispatch, etc.)
//  5. When the goroutine finishes, it resolves/rejects the Promise
//
// Worker.js must `await` the returned Promise.
func goAsync(fn func(this js.Value, args []js.Value) interface{}) js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		var resolve, reject js.Value
		executor := js.FuncOf(func(_ js.Value, promiseArgs []js.Value) interface{} {
			resolve = promiseArgs[0]
			reject = promiseArgs[1]
			return nil
		})
		promise := js.Global().Get("Promise").New(executor)
		executor.Release()

		// Capture this/args for the goroutine (args slice is only valid during the callback).
		thisVal := this
		argsCopy := make([]js.Value, len(args))
		copy(argsCopy, args)

		go func() {
			defer func() {
				if r := recover(); r != nil {
					reject.Invoke(js.Global().Get("Error").New(fmt.Sprintf("Go panic: %v", r)))
				}
			}()
			result := fn(thisVal, argsCopy)
			resolve.Invoke(result)
		}()

		return promise
	})
}

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

func sampleWithTemp(logits []float32, temp float32) int {
	n := len(logits)

	if temp != 1.0 {
		invT := 1.0 / temp
		for i := range logits {
			logits[i] *= invT
		}
	}

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

	// Simple random sample using timestamp-based pseudo-random
	seed := time.Now().UnixNano()
	r := float32(uint32(seed^(seed>>16))%10000) / 10000.0

	var cumulative float32
	for i, p := range logits {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return n - 1
}

// Suppress unused import warnings
var (
	_ = binary.LittleEndian
	_ = unsafe.Pointer(nil)
)

// ────────────────────────────────────────────────────────────────────────────
// Main — register all demo functions and block
// ────────────────────────────────────────────────────────────────────────────

func main() {
	// Initialise WebGPU pipeline early so all operators can use it.
	inference.EnsureWebGPU()

	demo := js.Global().Get("Object").New()

	// Demo 1: Basic features
	// All callbacks use goAsync to avoid blocking the handleEvent goroutine.
	// This is required for any callback that may block on a channel (e.g.,
	// WebGPU dispatch via <-doneChan in matMulF32). Without goAsync, the
	// handleEvent goroutine never sets event.returned = true, and the
	// Go WASM scheduler deadlocks in beforeIdle().
	demo.Set("info", goAsync(jsDemo1Info))
	demo.Set("testArena", goAsync(jsDemo1Arena))
	demo.Set("benchOperators", goAsync(jsDemo1Operators))
	demo.Set("testZeroCopy", goAsync(jsDemo1ZeroCopy))

	// Demo 2: GPT-2
	demo.Set("loadGPT2", goAsync(jsLoadGPT2))
	demo.Set("generateGPT2", goAsync(jsGPT2Generate))
	demo.Set("generateStep", goAsync(jsGPT2GenerateStep))
	demo.Set("benchMatMul", goAsync(jsGPT2MatMulBench))

	js.Global().Set("mempipeDemo", demo)

	fmt.Println("[Go/WASM] MemPipe demo module ready")

	// Keep-alive goroutine: prevents the Go runtime deadlock detector from
	// killing the program when a js.FuncOf handler blocks on a JS Promise
	// channel (e.g. WebGPU mapAsync inside await()). A sleeping goroutine
	// with an active timer is considered "alive" by the detector.
	go func() {
		for {
			time.Sleep(time.Hour * 24 * 365)
		}
	}()

	// Block forever — JS callbacks drive all further execution.
	select {}
}
