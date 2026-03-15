// Command ai_inference demonstrates zero-allocation inference with MemPipe.
//
// It programmatically builds a small MNIST-style MLP model (784→128→10),
// serializes it to .mpmodel format, loads it back, and runs inference.
// The steady-state inference loop has zero heap allocations.
//
// Usage:
//
//	go run ./examples/ai_inference
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/GoMemPipe/mempipe/inference"
)

func main() {
	fmt.Println("── MemPipe AI Inference Example ──")
	fmt.Println()

	// ── Step 1: Build a toy MNIST MLP model programmatically ──
	// In production you'd use mempipe-convert to convert a PyTorch/TF model.
	model := buildMNISTModel()
	fmt.Printf("Model: %s\n", model.Metadata.Name)
	fmt.Printf("  Input:   %v (%d elements)\n", model.Metadata.InputShapes[0].Dims, model.InputSize())
	fmt.Printf("  Output:  %v (%d elements)\n", model.Metadata.OutputShapes[0].Dims, model.OutputSize())
	fmt.Printf("  Weights: %d bytes\n", model.WeightsSize())
	fmt.Println()

	// ── Step 2: Serialize to .mpmodel bytes ──
	data, err := inference.SerializeModel(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "serialize: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Serialized .mpmodel: %d bytes\n", len(data))

	// ── Step 3: Load model from bytes (simulates reading from disk/network) ──
	loaded, err := inference.LoadModelFromBytes(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	// ── Step 4: Create inference engine (all memory allocated up front) ──
	engine, err := inference.NewEngine(loaded)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Arena: %d / %d bytes used\n", engine.ArenaUsed(), engine.ArenaTotal())
	fmt.Println()

	// ── Step 5: Run inference (byte-based API) ──
	fmt.Println("── Byte-based inference ──")
	inputBytes := generateFakeDigit(784) // 28x28 "image" as bytes
	output, err := engine.Infer(inputBytes)
	if err != nil {
		fmt.Fprintf(os.Stderr, "infer: %v\n", err)
		os.Exit(1)
	}
	printPredictions(output)

	// ── Step 6: Run zero-copy tensor inference (0 allocs) ──
	fmt.Println()
	fmt.Println("── Zero-copy tensor inference (0 allocs) ──")
	inputTensor := engine.InputTensors()[0]
	floats := inputTensor.Float32s()

	// Write a synthetic "7" pattern directly into arena memory
	for i := range floats {
		floats[i] = 0.0
	}
	// Draw a crude "7" in the 28x28 grid
	for x := 5; x < 23; x++ {
		floats[3*28+x] = 1.0 // top horizontal bar
	}
	for y := 4; y < 24; y++ {
		col := 22 - (y-4)*17/20
		if col >= 0 && col < 28 {
			floats[y*28+col] = 1.0 // diagonal stroke
		}
	}

	// Execute: zero allocations on this hot path
	outputs, err := engine.InferTensor()
	if err != nil {
		fmt.Fprintf(os.Stderr, "infer_tensor: %v\n", err)
		os.Exit(1)
	}

	// Read results directly from arena
	outFloats := outputs[0].Float32s()
	bestClass, bestProb := argmax(outFloats)
	fmt.Printf("Predicted digit: %d (confidence: %.4f)\n", bestClass, bestProb)
	fmt.Println("All probabilities:")
	for i, p := range outFloats {
		bar := ""
		for j := 0; j < int(p*40); j++ {
			bar += "█"
		}
		fmt.Printf("  [%d] %.4f %s\n", i, p, bar)
	}

	// ── Step 7: Benchmark steady-state inference ──
	fmt.Println()
	fmt.Println("── Throughput benchmark ──")
	const runs = 100_000
	start := time.Now()
	for i := 0; i < runs; i++ {
		engine.InferTensor()
	}
	elapsed := time.Since(start)
	fmt.Printf("%d inferences in %v\n", runs, elapsed)
	fmt.Printf("%.0f inferences/sec\n", float64(runs)/elapsed.Seconds())
	fmt.Printf("%.2f µs/inference\n", float64(elapsed.Microseconds())/float64(runs))
}

// buildMNISTModel creates a toy MLP: Dense(784→128) → ReLU → Dense(128→10) → Softmax
func buildMNISTModel() *inference.Model {
	tensorNames := []string{
		"input",   // 0: [1, 784]
		"w1",      // 1: [784, 128]
		"b1",      // 2: [128]
		"hidden1", // 3: [1, 128]
		"h1_relu", // 4: [1, 128]
		"w2",      // 5: [128, 10]
		"b2",      // 6: [10]
		"output",  // 7: [1, 10]
	}

	graph := []inference.OpNode{
		{Type: inference.OpDense, InputIndices: []int{0, 1, 2}, OutputIndices: []int{3}},
		{Type: inference.OpReLU, InputIndices: []int{3}, OutputIndices: []int{4}},
		{Type: inference.OpDense, InputIndices: []int{4, 5, 6}, OutputIndices: []int{7}},
		{Type: inference.OpSoftmax, InputIndices: []int{7}, OutputIndices: []int{7}},
	}

	// Xavier-initialized random weights
	rng := rand.New(rand.NewSource(42))
	w1 := xavierInit(rng, 784, 128)
	b1 := make([]float32, 128)
	w2 := xavierInit(rng, 128, 10)
	b2 := make([]float32, 10)

	var weights []float32
	weights = append(weights, w1...)
	weights = append(weights, b1...)
	weights = append(weights, w2...)
	weights = append(weights, b2...)

	blob := float32sToBytes(weights)

	return &inference.Model{
		Metadata: inference.Metadata{
			Name:         "mnist-mlp-demo",
			InputShapes:  []inference.Shape{{Dims: []int{1, 784}}},
			OutputShapes: []inference.Shape{{Dims: []int{1, 10}}},
		},
		TensorNames: tensorNames,
		TensorShapes: map[string]inference.Shape{
			"w1": {Dims: []int{784, 128}},
			"b1": {Dims: []int{128}},
			"w2": {Dims: []int{128, 10}},
			"b2": {Dims: []int{10}},
		},
		Graph:       graph,
		WeightsBlob: blob,
	}
}

// xavierInit produces Xavier-uniform initialized weights.
func xavierInit(rng *rand.Rand, fanIn, fanOut int) []float32 {
	limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
	w := make([]float32, fanIn*fanOut)
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * limit
	}
	return w
}

// float32sToBytes converts a float32 slice to little-endian bytes.
func float32sToBytes(fs []float32) []byte {
	buf := make([]byte, len(fs)*4)
	for i, v := range fs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// generateFakeDigit creates a fake 28x28 pixel image as float32 bytes.
func generateFakeDigit(pixels int) []byte {
	rng := rand.New(rand.NewSource(7))
	buf := make([]byte, pixels*4)
	for i := 0; i < pixels; i++ {
		v := rng.Float32()
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// printPredictions decodes output bytes and prints class probabilities.
func printPredictions(output []byte) {
	numClasses := len(output) / 4
	bestClass := 0
	bestProb := float32(0)
	for i := 0; i < numClasses; i++ {
		bits := binary.LittleEndian.Uint32(output[i*4:])
		prob := math.Float32frombits(bits)
		if prob > bestProb {
			bestProb = prob
			bestClass = i
		}
		fmt.Printf("  class %d: %.4f\n", i, prob)
	}
	fmt.Printf("Predicted: %d (%.4f)\n", bestClass, bestProb)
}

// argmax returns the index and value of the maximum element.
func argmax(fs []float32) (int, float32) {
	best := 0
	for i := 1; i < len(fs); i++ {
		if fs[i] > fs[best] {
			best = i
		}
	}
	return best, fs[best]
}
