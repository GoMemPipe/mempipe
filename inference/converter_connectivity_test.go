package inference_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/GoMemPipe/mempipe/inference"
)

// TestMpmodelTransformerConnectivityFromEnv loads a .mpmodel whose path is set by
// MPMODEL_CONNECTIVITY_PATH (written by tools/mempipe-convert Python tests).
// It verifies two distinct inputs produce distinct outputs so the input tensor
// is actually wired through the graph (no disconnected intermediate tensors).
func TestMpmodelTransformerConnectivityFromEnv(t *testing.T) {
	path := os.Getenv("MPMODEL_CONNECTIVITY_PATH")
	if path == "" {
		t.Skip("MPMODEL_CONNECTIVITY_PATH not set (Python connectivity test sets this)")
	}

	model, err := inference.LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	engine, err := inference.NewEngine(model)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	if len(model.Metadata.InputShapes) < 1 {
		t.Fatalf("expected at least one input")
	}
	inElems := model.Metadata.InputShapes[0].NumElements()
	if inElems < 1 {
		t.Fatalf("input num elements: %d", inElems)
	}

	mkInput := func(seed float32) []byte {
		b := make([]byte, inElems*4)
		for i := 0; i < inElems; i++ {
			v := seed + float32(i)*0.25
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
		}
		return b
	}

	outA, err := engine.Infer(mkInput(1.0))
	if err != nil {
		t.Fatalf("Infer A: %v", err)
	}
	outB, err := engine.Infer(mkInput(100.0))
	if err != nil {
		t.Fatalf("Infer B: %v", err)
	}

	if bytes.Equal(outA, outB) {
		t.Fatalf("outputs are byte-identical — input likely disconnected from the graph body")
	}
}
