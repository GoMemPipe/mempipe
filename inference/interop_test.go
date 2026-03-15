package inference_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/GoMemPipe/mempipe/inference"
)

// TestCrossLanguageInterop loads a .mpmodel file created by the Python
// converter and verifies the Go loader can parse it correctly.
func TestCrossLanguageInterop(t *testing.T) {
	// Find the zoo-generated model
	_, thisFile, _, _ := runtime.Caller(0)
	root := filepath.Dir(filepath.Dir(thisFile))
	modelPath := filepath.Join(root, "tools", "mempipe-convert", "zoo", "mnist_mlp.mpmodel")

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Zoo model not found at %s — run zoo/convert_mnist.py first", modelPath)
	}

	model, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if model.Metadata.Name != "mnist-mlp" {
		t.Errorf("name: got %q, want %q", model.Metadata.Name, "mnist-mlp")
	}
	if len(model.Graph) != 4 {
		t.Errorf("graph nodes: got %d, want 4", len(model.Graph))
	}
	if len(model.TensorNames) != 8 {
		t.Errorf("tensor names: got %d, want 8", len(model.TensorNames))
	}

	// Verify weight tensor shapes were preserved
	if s, ok := model.TensorShapes["w1"]; !ok {
		t.Error("w1 shape missing")
	} else if s.Dims[0] != 784 || s.Dims[1] != 128 {
		t.Errorf("w1 shape: got %v, want [784,128]", s.Dims)
	}

	if s, ok := model.TensorShapes["w2"]; !ok {
		t.Error("w2 shape missing")
	} else if s.Dims[0] != 128 || s.Dims[1] != 10 {
		t.Errorf("w2 shape: got %v, want [128,10]", s.Dims)
	}

	// Verify input/output shapes
	if len(model.Metadata.InputShapes) != 1 {
		t.Fatalf("input shapes: got %d", len(model.Metadata.InputShapes))
	}
	if !model.Metadata.InputShapes[0].Equal(inference.Shape{Dims: []int{1, 784}}) {
		t.Errorf("input shape: got %v", model.Metadata.InputShapes[0].Dims)
	}

	// Verify we can create an engine from this model
	engine, err := inference.NewEngine(model)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	// Run inference with zeros as input
	inputBytes := make([]byte, 784*4) // 784 float32 zeros
	output, err := engine.Infer(inputBytes)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	// Output should be 10 float32 softmax values summing to ~1.0
	if len(output) != 10*4 {
		t.Fatalf("output size: got %d, want %d", len(output), 10*4)
	}
}
