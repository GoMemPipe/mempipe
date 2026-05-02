package inference

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestInferReshape3DAttrs(t *testing.T) {
	inShape := Shape{Dims: []int{2, 8, 128}}
	attrs := make([]byte, 2+3*4)
	binary.LittleEndian.PutUint16(attrs[0:2], 3)
	binary.LittleEndian.PutUint32(attrs[2:6], uint32(2))
	binary.LittleEndian.PutUint32(attrs[6:10], uint32(8))
	binary.LittleEndian.PutUint32(attrs[10:14], uint32(128))

	out, err := inferOpOutputShapes(OpReshape, []Shape{inShape}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || len(out[0].Dims) != 3 {
		t.Fatalf("got %#v", out)
	}
	for i, d := range []int{2, 8, 128} {
		if out[0].Dims[i] != d {
			t.Fatalf("dim %d: got %d want %d", i, out[0].Dims[i], d)
		}
	}
}

func TestInferReshapeFromShapeInputHints(t *testing.T) {
	inShape := Shape{Dims: []int{2, 8, 128}}
	rx := &reshapeExtra{
		TensorNames:  []string{"x", "hint", "y"},
		InputIndices: []int{0, 1},
		Hints:        map[string][]int{"hint": {2, 64, 16}},
	}
	out, err := inferOpOutputShapesEx(OpReshape, []Shape{inShape}, nil, rx)
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0].Dims) != 3 || out[0].Dims[0] != 2 || out[0].Dims[1] != 64 || out[0].Dims[2] != 16 {
		t.Fatalf("got %v", out[0].Dims)
	}
}

// Stale attrs must not override ReshapeShapeHints; hints must not be "fixed"
// by the single-dimension attrs reconciliation path.
func TestReshapeHintOverridesStaleAttrs(t *testing.T) {
	inShape := Shape{Dims: []int{2, 8, 128}}
	attrs := make([]byte, 2+3*4)
	binary.LittleEndian.PutUint16(attrs[0:2], 3)
	binary.LittleEndian.PutUint32(attrs[2:6], uint32(2))
	binary.LittleEndian.PutUint32(attrs[6:10], uint32(4))  // wrong vs hint
	binary.LittleEndian.PutUint32(attrs[10:14], uint32(256)) // product still 2048 — rewriter could corrupt
	rx := &reshapeExtra{
		TensorNames:  []string{"x", "shape_const", "y"},
		InputIndices: []int{0, 1},
		Hints:        map[string][]int{"shape_const": {2, 64, 16}},
	}
	out, err := inferOpOutputShapesEx(OpReshape, []Shape{inShape}, attrs, rx)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 64, 16}
	if len(out[0].Dims) != 3 {
		t.Fatalf("got %v", out[0].Dims)
	}
	for i := range want {
		if out[0].Dims[i] != want[i] {
			t.Fatalf("dim %d: got %d want %d (attrs must not override hint)", i, out[0].Dims[i], want[i])
		}
	}
}

func TestInferShapesWithReshapeShapeHintsOption(t *testing.T) {
	tensorNames := []string{"in0", "shape_in", "out0"}
	graph := []OpNode{
		{Type: OpReshape, InputIndices: []int{0, 1}, OutputIndices: []int{2}, Attrs: nil},
	}
	inputShapes := map[string]Shape{
		"in0":   {Dims: []int{2, 8, 128}},
		"shape_in": {Dims: []int{3}}, // unused for dims — hint supplies target shape
	}
	opt := &InferShapeOptions{
		ReshapeShapeHints: map[string][]int{"shape_in": {2, 64, 16}},
	}
	shapes, err := InferShapes(graph, tensorNames, inputShapes, opt)
	if err != nil {
		t.Fatal(err)
	}
	got := shapes["out0"].Dims
	want := []int{2, 64, 16}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dim %d: got %d want %d", i, got[i], want[i])
		}
	}
}

func TestInferBatchedMatMulBroadcast2DWeight(t *testing.T) {
	a := Shape{Dims: []int{2, 3, 4}}
	b := Shape{Dims: []int{4, 5}}
	out, err := inferOpOutputShapes(OpBatchedMatMul, []Shape{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 3, 5}
	if len(out[0].Dims) != len(want) {
		t.Fatalf("got %v", out[0].Dims)
	}
	for i := range want {
		if out[0].Dims[i] != want[i] {
			t.Fatalf("dim %d: got %d want %d", i, out[0].Dims[i], want[i])
		}
	}
}

func TestBatchedMatMulExecute2DBroadcastNoPanic(t *testing.T) {
	arena := NewInferenceArena(64 * 1024)
	a, err := arena.AllocTensor("A", []int{2, 3, 4}, Float32)
	if err != nil {
		t.Fatal(err)
	}
	b, err := arena.AllocTensor("B", []int{4, 5}, Float32)
	if err != nil {
		t.Fatal(err)
	}
	c, err := arena.AllocTensor("C", []int{2, 3, 5}, Float32)
	if err != nil {
		t.Fatal(err)
	}
	ad := a.Float32s()
	for i := range ad {
		ad[i] = float32(i%7) * 0.1
	}
	bd := b.Float32s()
	for i := range bd {
		bd[i] = float32(i%3) * 0.25
	}
	op := batchedMatMulOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatal(err)
	}
	cd := c.Float32s()
	if math.IsNaN(float64(cd[0])) || math.IsNaN(float64(cd[len(cd)-1])) {
		t.Fatal("unexpected NaN in output")
	}
}

func TestInferTransposeRankMismatchNoPanic(t *testing.T) {
	// Permutation encoded for 3D while tensor is 2D — must fall back safely (no index panic).
	attrs := make([]byte, 2+3*2)
	binary.LittleEndian.PutUint16(attrs[0:2], 3)
	binary.LittleEndian.PutUint16(attrs[2:4], 0)
	binary.LittleEndian.PutUint16(attrs[4:6], 2)
	binary.LittleEndian.PutUint16(attrs[6:8], 1)

	inShape := Shape{Dims: []int{12, 34}}
	out, err := inferOpOutputShapes(OpTranspose, []Shape{inShape}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0].Dims) != 2 || out[0].Dims[0] != 34 || out[0].Dims[1] != 12 {
		t.Fatalf("expected reversed [34 12], got %v", out[0].Dims)
	}
}

func TestInferAvgPool2DKernel1x16DefaultStride(t *testing.T) {
	// ONNX AveragePool: strides default to 1, pads default to 0 when omitted from attrs blob.
	// Input [1,3,1,16], kernel [1,16] → output [1,3,1,1] (not a degenerate 0 spatial dim).
	in := Shape{Dims: []int{1, 3, 1, 16}}
	attrs := make([]byte, 4)
	binary.LittleEndian.PutUint16(attrs[0:2], 1)
	binary.LittleEndian.PutUint16(attrs[2:4], 16)

	out, err := inferOpOutputShapes(OpAvgPool2D, []Shape{in}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 3, 1, 1}
	if len(out) != 1 || len(out[0].Dims) != 4 {
		t.Fatalf("got %#v", out)
	}
	for i := range want {
		if out[0].Dims[i] != want[i] {
			t.Fatalf("dim %d: got %d want %d", i, out[0].Dims[i], want[i])
		}
	}
}

func TestInferPool2DLegacyNoAttrs(t *testing.T) {
	// Short attrs: legacy 2×2 kernel, stride 2 (matches old .mpmodel without attrs).
	in := Shape{Dims: []int{1, 1, 4, 4}}
	out, err := inferOpOutputShapes(OpAvgPool2D, []Shape{in}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || len(out[0].Dims) != 4 || out[0].Dims[2] != 2 || out[0].Dims[3] != 2 {
		t.Fatalf("got %v", out[0].Dims)
	}
}

func TestInferMatMul3DBroadcast(t *testing.T) {
	a := Shape{Dims: []int{2, 8, 128}}
	b := Shape{Dims: []int{128, 64}}
	out, err := inferOpOutputShapes(OpMatMul, []Shape{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 8, 64}
	if len(out[0].Dims) != len(want) {
		t.Fatalf("got %v", out[0].Dims)
	}
	for i := range want {
		if out[0].Dims[i] != want[i] {
			t.Fatalf("dim %d: got %d want %d", i, out[0].Dims[i], want[i])
		}
	}
}

func TestSoftmaxAxisMiddle3D(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{2, 3, 2}, Float32)
	b, _ := arena.AllocTensor("B", []int{2, 3, 2}, Float32)
	ad := a.Float32s()
	for i := range ad {
		ad[i] = float32(i%5) * 0.25
	}
	op := &softmaxOp{}
	attrs := make([]byte, 2)
	binary.LittleEndian.PutUint16(attrs[0:2], uint16(int16(1)))
	if err := op.SetAttrs(attrs); err != nil {
		t.Fatal(err)
	}
	if err := op.Execute([]*Tensor{a}, []*Tensor{b}); err != nil {
		t.Fatal(err)
	}
	bd := b.Float32s()
	sum := bd[0] + bd[2] + bd[4]
	if d := float64(sum - 1); d > 1e-4 || d < -1e-4 {
		t.Fatalf("softmax axis=1 fiber sum got %f want 1", sum)
	}
}
