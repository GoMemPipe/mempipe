package inference

import (
	"math"
	"testing"
)

// ── Model format: serialize/deserialize round-trip ──

func TestModelRoundTrip(t *testing.T) {
	model := &Model{
		Metadata: Metadata{
			Name:          "test-mlp",
			InputShapes:   []Shape{{Dims: []int{1, 784}}},
			OutputShapes:  []Shape{{Dims: []int{1, 10}}},
			PlatformHints: "native",
		},
		TensorNames: []string{"input", "w1", "b1", "hidden", "w2", "b2", "output"},
		Graph: []OpNode{
			{Type: OpDense, InputIndices: []int{0, 1, 2}, OutputIndices: []int{3}},
			{Type: OpReLU, InputIndices: []int{3}, OutputIndices: []int{3}},
			{Type: OpDense, InputIndices: []int{3, 4, 5}, OutputIndices: []int{6}},
			{Type: OpSoftmax, InputIndices: []int{6}, OutputIndices: []int{6}},
		},
		WeightsBlob: make([]byte, 1024),
	}

	data, err := SerializeModel(model)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	if string(data[0:4]) != MagicBytes {
		t.Errorf("magic: got %q, want %q", data[0:4], MagicBytes)
	}

	model2, err := LoadModelFromBytes(data)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if model2.Metadata.Name != "test-mlp" {
		t.Errorf("name: got %q", model2.Metadata.Name)
	}
	if len(model2.Graph) != 4 {
		t.Errorf("graph nodes: got %d, want 4", len(model2.Graph))
	}
	if len(model2.TensorNames) != 7 {
		t.Errorf("tensor names: got %d, want 7", len(model2.TensorNames))
	}
	if model2.WeightsSize() != 1024 {
		t.Errorf("weights: got %d, want 1024", model2.WeightsSize())
	}
	if !model2.Metadata.InputShapes[0].Equal(Shape{Dims: []int{1, 784}}) {
		t.Errorf("input shape mismatch")
	}
	if !model2.Metadata.OutputShapes[0].Equal(Shape{Dims: []int{1, 10}}) {
		t.Errorf("output shape mismatch")
	}
}

func TestModelValidation(t *testing.T) {
	m := &Model{}
	if err := m.Validate(); err == nil {
		t.Error("expected error for empty model")
	}
	m = &Model{
		Metadata: Metadata{
			InputShapes:  []Shape{{Dims: []int{1, 10}}},
			OutputShapes: []Shape{{Dims: []int{1, 10}}},
		},
	}
	if err := m.Validate(); err == nil {
		t.Error("expected error for model with no graph")
	}
}

func TestModelQuantizationFlags(t *testing.T) {
	model := &Model{
		Metadata: Metadata{
			Name:         "quant-test",
			InputShapes:  []Shape{{Dims: []int{1, 10}}},
			OutputShapes: []Shape{{Dims: []int{1, 10}}},
			QuantMethod:  "int8_symmetric",
			QuantScale:   0.05,
		},
		TensorNames: []string{"input", "output"},
		Graph:       []OpNode{{Type: OpReLU, InputIndices: []int{0}, OutputIndices: []int{1}}},
		WeightsBlob: make([]byte, 64),
	}
	data, err := SerializeModel(model)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	model2, err := LoadModelFromBytes(data)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if model2.Metadata.QuantMethod != "int8_symmetric" {
		t.Errorf("quant method: got %q", model2.Metadata.QuantMethod)
	}
}

func TestInvalidMagic(t *testing.T) {
	data := make([]byte, 128)
	copy(data[0:4], "BAAD")
	if _, err := LoadModelFromBytes(data); err == nil {
		t.Error("expected error for bad magic")
	}
}

func TestTruncatedData(t *testing.T) {
	if _, err := LoadModelFromBytes([]byte{1, 2, 3}); err == nil {
		t.Error("expected error for truncated data")
	}
}

// ── Tensor tests ──

func TestTensorAllocAndAccess(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, err := arena.AllocTensor("test", []int{2, 3}, Float32)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if tensor.NumElements() != 6 {
		t.Errorf("num elements: got %d, want 6", tensor.NumElements())
	}
	if tensor.Rank() != 2 {
		t.Errorf("rank: got %d, want 2", tensor.Rank())
	}
	if tensor.ByteSize() != 24 {
		t.Errorf("byte size: got %d, want 24", tensor.ByteSize())
	}
	tensor.SetF32(3.14, 0, 0)
	tensor.SetF32(2.71, 1, 2)
	if v := tensor.AtF32(0, 0); v != 3.14 {
		t.Errorf("AtF32(0,0): got %f, want 3.14", v)
	}
	if v := tensor.AtF32(1, 2); v != 2.71 {
		t.Errorf("AtF32(1,2): got %f, want 2.71", v)
	}
}

func TestTensorFloat32Slice(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, _ := arena.AllocTensor("test", []int{4}, Float32)
	data := tensor.Float32s()
	data[0], data[1], data[2], data[3] = 1.0, 2.0, 3.0, 4.0
	if tensor.AtF32(2) != 3.0 {
		t.Errorf("slice write not visible via AtF32")
	}
}

func TestTensorReshape(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, _ := arena.AllocTensor("test", []int{2, 6}, Float32)
	reshaped, err := tensor.Reshape(3, 4)
	if err != nil {
		t.Fatalf("reshape: %v", err)
	}
	if reshaped.shape[0] != 3 || reshaped.shape[1] != 4 {
		t.Errorf("reshape shape: got %v", reshaped.shape)
	}
	if reshaped.DataPtr() != tensor.DataPtr() {
		t.Error("reshape should share same data pointer")
	}
	if _, err = tensor.Reshape(3, 5); err == nil {
		t.Error("expected error for invalid reshape")
	}
}

func TestTensorSlice(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, _ := arena.AllocTensor("test", []int{4, 3}, Float32)
	data := tensor.Float32s()
	for i := range data {
		data[i] = float32(i)
	}
	sliced, err := tensor.Slice(1, 3)
	if err != nil {
		t.Fatalf("slice: %v", err)
	}
	if sliced.shape[0] != 2 || sliced.shape[1] != 3 {
		t.Errorf("slice shape: got %v", sliced.shape)
	}
	f := sliced.Float32s()
	if f[0] != 3.0 || f[1] != 4.0 || f[2] != 5.0 {
		t.Errorf("slice values: got %v", f[:3])
	}
}

func TestTensorCopyFromTo(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, _ := arena.AllocTensor("test", []int{4}, Float32)
	// [1.0, 2.0, 3.0, 4.0] as little-endian bytes
	input := []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}
	if err := tensor.CopyFrom(input); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if tensor.AtF32(0) != 1.0 || tensor.AtF32(3) != 4.0 {
		t.Errorf("CopyFrom values wrong")
	}
	output := make([]byte, 16)
	if err := tensor.CopyTo(output); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range input {
		if input[i] != output[i] {
			t.Errorf("CopyTo mismatch at byte %d", i)
		}
	}
}

func TestTensorZero(t *testing.T) {
	arena := NewInferenceArena(4096)
	tensor, _ := arena.AllocTensor("test", []int{4}, Float32)
	tensor.Float32s()[0] = 99.0
	tensor.Zero()
	if tensor.Float32s()[0] != 0 {
		t.Errorf("zero failed")
	}
}

func TestInferenceArenaOOM(t *testing.T) {
	arena := NewInferenceArena(128)
	if _, err := arena.AllocTensor("big", []int{1000}, Float32); err == nil {
		t.Error("expected OOM error")
	}
}

// ── Operator tests ──

func TestMatMulOp(t *testing.T) {
	arena := NewInferenceArena(8192)
	a, _ := arena.AllocTensor("A", []int{2, 3}, Float32)
	b, _ := arena.AllocTensor("B", []int{3, 2}, Float32)
	c, _ := arena.AllocTensor("C", []int{2, 2}, Float32)

	ad := a.Float32s()
	ad[0], ad[1], ad[2] = 1, 2, 3
	ad[3], ad[4], ad[5] = 4, 5, 6

	bd := b.Float32s()
	bd[0], bd[1] = 7, 8
	bd[2], bd[3] = 9, 10
	bd[4], bd[5] = 11, 12

	op := &matMulOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	cd := c.Float32s()
	want := []float32{58, 64, 139, 154}
	for i, w := range want {
		if cd[i] != w {
			t.Errorf("C[%d]: got %f, want %f", i, cd[i], w)
		}
	}
}

func TestDenseOp(t *testing.T) {
	arena := NewInferenceArena(8192)
	a, _ := arena.AllocTensor("input", []int{1, 3}, Float32)
	w, _ := arena.AllocTensor("weights", []int{3, 2}, Float32)
	bias, _ := arena.AllocTensor("bias", []int{2}, Float32)
	out, _ := arena.AllocTensor("output", []int{1, 2}, Float32)

	a.Float32s()[0], a.Float32s()[1], a.Float32s()[2] = 1, 2, 3
	wd := w.Float32s()
	wd[0], wd[1] = 1, 0
	wd[2], wd[3] = 0, 1
	wd[4], wd[5] = 1, 1
	bias.Float32s()[0], bias.Float32s()[1] = 0.5, -0.5

	op := &denseOp{}
	if err := op.Execute([]*Tensor{a, w, bias}, []*Tensor{out}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	od := out.Float32s()
	if od[0] != 4.5 || od[1] != 4.5 {
		t.Errorf("dense: got [%f, %f], want [4.5, 4.5]", od[0], od[1])
	}
}

func TestAddOp(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{4}, Float32)
	b, _ := arena.AllocTensor("B", []int{4}, Float32)
	c, _ := arena.AllocTensor("C", []int{4}, Float32)
	for i := range a.Float32s() {
		a.Float32s()[i] = float32(i)
		b.Float32s()[i] = float32(i * 10)
	}
	op := &addOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	for i, v := range c.Float32s() {
		want := float32(i + i*10)
		if v != want {
			t.Errorf("C[%d]: got %f, want %f", i, v, want)
		}
	}
}

func TestAddBroadcast(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{6}, Float32)
	b, _ := arena.AllocTensor("B", []int{3}, Float32)
	c, _ := arena.AllocTensor("C", []int{6}, Float32)
	for i := range a.Float32s() {
		a.Float32s()[i] = float32(i)
	}
	b.Float32s()[0], b.Float32s()[1], b.Float32s()[2] = 100, 200, 300
	op := &addOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	want := []float32{100, 201, 302, 103, 204, 305}
	for i, w := range want {
		if c.Float32s()[i] != w {
			t.Errorf("C[%d]: got %f, want %f", i, c.Float32s()[i], w)
		}
	}
}

func TestReLUOp(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{5}, Float32)
	b, _ := arena.AllocTensor("B", []int{5}, Float32)
	ad := a.Float32s()
	ad[0], ad[1], ad[2], ad[3], ad[4] = -2, -1, 0, 1, 2
	op := &reluOp{}
	if err := op.Execute([]*Tensor{a}, []*Tensor{b}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	want := []float32{0, 0, 0, 1, 2}
	for i, w := range want {
		if b.Float32s()[i] != w {
			t.Errorf("B[%d]: got %f, want %f", i, b.Float32s()[i], w)
		}
	}
}

func TestSigmoidOp(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{3}, Float32)
	b, _ := arena.AllocTensor("B", []int{3}, Float32)
	ad := a.Float32s()
	ad[0], ad[1], ad[2] = -10, 0, 10
	op := &sigmoidOp{}
	if err := op.Execute([]*Tensor{a}, []*Tensor{b}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	bd := b.Float32s()
	if bd[1] != 0.5 {
		t.Errorf("sigmoid(0): got %f", bd[1])
	}
	if bd[0] >= 0.001 {
		t.Errorf("sigmoid(-10) too large: %f", bd[0])
	}
	if bd[2] <= 0.999 {
		t.Errorf("sigmoid(10) too small: %f", bd[2])
	}
}

func TestSoftmaxOp(t *testing.T) {
	arena := NewInferenceArena(4096)
	a, _ := arena.AllocTensor("A", []int{1, 4}, Float32)
	b, _ := arena.AllocTensor("B", []int{1, 4}, Float32)
	ad := a.Float32s()
	ad[0], ad[1], ad[2], ad[3] = 1, 2, 3, 4
	op := &softmaxOp{}
	if err := op.Execute([]*Tensor{a}, []*Tensor{b}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	bd := b.Float32s()
	var sum float32
	for _, v := range bd {
		sum += v
	}
	if d := float64(sum - 1.0); d > 1e-5 || d < -1e-5 {
		t.Errorf("softmax sum: %f", sum)
	}
	for i := 1; i < 4; i++ {
		if bd[i] <= bd[i-1] {
			t.Errorf("not monotonic at %d", i)
		}
	}
}

func TestConv2DOp(t *testing.T) {
	arena := NewInferenceArena(65536)
	input, _ := arena.AllocTensor("input", []int{1, 1, 4, 4}, Float32)
	kernel, _ := arena.AllocTensor("kernel", []int{1, 1, 2, 2}, Float32)
	output, _ := arena.AllocTensor("output", []int{1, 1, 3, 3}, Float32)
	for i := range input.Float32s() {
		input.Float32s()[i] = 1.0
	}
	kd := kernel.Float32s()
	kd[0], kd[1], kd[2], kd[3] = 1, 1, 1, 1
	op := &conv2dOp{}
	if err := op.Execute([]*Tensor{input, kernel}, []*Tensor{output}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	for i, v := range output.Float32s() {
		if v != 4.0 {
			t.Errorf("output[%d]: got %f, want 4.0", i, v)
		}
	}
}

func TestMaxPool2DOp(t *testing.T) {
	arena := NewInferenceArena(65536)
	input, _ := arena.AllocTensor("input", []int{1, 1, 4, 4}, Float32)
	output, _ := arena.AllocTensor("output", []int{1, 1, 2, 2}, Float32)
	for i := range input.Float32s() {
		input.Float32s()[i] = float32(i)
	}
	op := &maxPool2dOp{}
	if err := op.Execute([]*Tensor{input}, []*Tensor{output}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	want := []float32{5, 7, 13, 15}
	for i, w := range want {
		if output.Float32s()[i] != w {
			t.Errorf("pool[%d]: got %f, want %f", i, output.Float32s()[i], w)
		}
	}
}

func TestAvgPool2DOp(t *testing.T) {
	arena := NewInferenceArena(65536)
	input, _ := arena.AllocTensor("input", []int{1, 1, 4, 4}, Float32)
	output, _ := arena.AllocTensor("output", []int{1, 1, 2, 2}, Float32)
	for i := range input.Float32s() {
		input.Float32s()[i] = float32(i)
	}
	op := &avgPool2dOp{}
	if err := op.Execute([]*Tensor{input}, []*Tensor{output}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	want := []float32{2.5, 4.5, 10.5, 12.5}
	for i, w := range want {
		if output.Float32s()[i] != w {
			t.Errorf("pool[%d]: got %f, want %f", i, output.Float32s()[i], w)
		}
	}
}

func TestBatchNormOp(t *testing.T) {
	arena := NewInferenceArena(65536)
	x, _ := arena.AllocTensor("x", []int{1, 2}, Float32)
	gamma, _ := arena.AllocTensor("gamma", []int{2}, Float32)
	beta, _ := arena.AllocTensor("beta", []int{2}, Float32)
	mn, _ := arena.AllocTensor("mean", []int{2}, Float32)
	vr, _ := arena.AllocTensor("var", []int{2}, Float32)
	y, _ := arena.AllocTensor("y", []int{1, 2}, Float32)

	x.Float32s()[0], x.Float32s()[1] = 2.0, 4.0
	gamma.Float32s()[0], gamma.Float32s()[1] = 1.0, 1.0
	beta.Float32s()[0], beta.Float32s()[1] = 0.0, 0.0
	mn.Float32s()[0], mn.Float32s()[1] = 0.0, 0.0
	vr.Float32s()[0], vr.Float32s()[1] = 1.0, 1.0

	op := &batchNormOp{}
	if err := op.Execute([]*Tensor{x, gamma, beta, mn, vr}, []*Tensor{y}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	yd := y.Float32s()
	exp0 := float32(2.0 / math.Sqrt(1.0+1e-5))
	if d := yd[0] - exp0; d > 0.001 || d < -0.001 {
		t.Errorf("batchnorm[0]: got %f, want ~%f", yd[0], exp0)
	}
}

// ── Quantization tests ──

func TestQuantizeSymmetric(t *testing.T) {
	arena := NewInferenceArena(8192)
	src, _ := arena.AllocTensor("src", []int{5}, Float32)
	dst, _ := arena.AllocTensor("dst", []int{5}, Int8)
	sd := src.Float32s()
	sd[0], sd[1], sd[2], sd[3], sd[4] = -1.0, -0.5, 0.0, 0.5, 1.0
	scale, err := QuantizeSymmetric(src, dst)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	if scale <= 0 {
		t.Errorf("scale: %f", scale)
	}
	dd := dst.Int8s()
	if dd[2] != 0 {
		t.Errorf("quant(0): got %d", dd[2])
	}
	if dd[4] != 127 {
		t.Errorf("quant(1): got %d", dd[4])
	}
	if dd[0] != -127 {
		t.Errorf("quant(-1): got %d", dd[0])
	}
}

func TestQuantizeAsymmetric(t *testing.T) {
	arena := NewInferenceArena(8192)
	src, _ := arena.AllocTensor("src", []int{4}, Float32)
	dst, _ := arena.AllocTensor("dst", []int{4}, Int8)
	sd := src.Float32s()
	sd[0], sd[1], sd[2], sd[3] = 0.0, 0.25, 0.5, 1.0
	scale, _, err := QuantizeAsymmetric(src, dst)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	if scale <= 0 {
		t.Errorf("scale: %f", scale)
	}
}

func TestDequantizeInt8ToFloat32(t *testing.T) {
	arena := NewInferenceArena(8192)
	src, _ := arena.AllocTensor("src", []int{3}, Int8)
	dst, _ := arena.AllocTensor("dst", []int{3}, Float32)
	sd := src.Int8s()
	sd[0], sd[1], sd[2] = -127, 0, 127
	if err := DequantizeInt8ToFloat32(src, dst, 1.0/127.0, 0); err != nil {
		t.Fatalf("dequantize: %v", err)
	}
	dd := dst.Float32s()
	if d := dd[0] - (-1.0); d > 0.01 || d < -0.01 {
		t.Errorf("deq(-127): got %f", dd[0])
	}
	if dd[1] != 0 {
		t.Errorf("deq(0): got %f", dd[1])
	}
	if d := dd[2] - 1.0; d > 0.01 || d < -0.01 {
		t.Errorf("deq(127): got %f", dd[2])
	}
}

func TestF16RoundTrip(t *testing.T) {
	vals := []float32{0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625}
	for _, v := range vals {
		h := F32ToF16Bits(v)
		back := F16BitsToF32(h)
		if v != back {
			t.Errorf("F16 round-trip: %f -> %d -> %f", v, h, back)
		}
	}
}

func TestF32ToF16TensorConversion(t *testing.T) {
	arena := NewInferenceArena(8192)
	src, _ := arena.AllocTensor("src", []int{4}, Float32)
	mid, _ := arena.AllocTensor("mid", []int{4}, Float16)
	dst, _ := arena.AllocTensor("dst", []int{4}, Float32)
	sd := src.Float32s()
	sd[0], sd[1], sd[2], sd[3] = 0, 1.0, -1.0, 0.5
	if err := F32ToF16(src, mid); err != nil {
		t.Fatalf("F32ToF16: %v", err)
	}
	if err := F16ToF32(mid, dst); err != nil {
		t.Fatalf("F16ToF32: %v", err)
	}
	dd := dst.Float32s()
	for i := range sd {
		if sd[i] != dd[i] {
			t.Errorf("[%d]: %f -> %f", i, sd[i], dd[i])
		}
	}
}

func TestMatMulInt8(t *testing.T) {
	arena := NewInferenceArena(8192)
	a, _ := arena.AllocTensor("A", []int{2, 3}, Int8)
	b, _ := arena.AllocTensor("B", []int{3, 2}, Int8)
	c, _ := arena.AllocTensor("C", []int{2, 2}, Int32)
	ad := a.Int8s()
	ad[0], ad[1], ad[2] = 1, 2, 3
	ad[3], ad[4], ad[5] = 4, 5, 6
	bd := b.Int8s()
	bd[0], bd[1] = 7, 8
	bd[2], bd[3] = 9, 10
	bd[4], bd[5] = 11, 12
	MatMulInt8(a, b, c, 1.0, 1.0)
	cd := c.Int32s()
	want := []int32{58, 64, 139, 154}
	for i, w := range want {
		if cd[i] != w {
			t.Errorf("C[%d]: got %d, want %d", i, cd[i], w)
		}
	}
}

// ── Shape inference tests ──

func TestShapeInference(t *testing.T) {
	tensorNames := []string{"input", "w1", "hidden", "output"}
	graph := []OpNode{
		{Type: OpMatMul, InputIndices: []int{0, 1}, OutputIndices: []int{2}},
		{Type: OpReLU, InputIndices: []int{2}, OutputIndices: []int{3}},
	}
	inputShapes := map[string]Shape{
		"input": {Dims: []int{1, 784}},
		"w1":    {Dims: []int{784, 128}},
	}
	shapes, err := InferShapes(graph, tensorNames, inputShapes)
	if err != nil {
		t.Fatalf("infer: %v", err)
	}
	h := shapes["hidden"]
	if h.Dims[0] != 1 || h.Dims[1] != 128 {
		t.Errorf("hidden shape: %v", h.Dims)
	}
	o := shapes["output"]
	if o.Dims[0] != 1 || o.Dims[1] != 128 {
		t.Errorf("output shape: %v", o.Dims)
	}
}

// ── Operator registry tests ──

func TestOperatorRegistry(t *testing.T) {
	ops := []OpType{
		OpMatMul, OpDense, OpAdd, OpReLU, OpSigmoid, OpSoftmax,
		OpConv2D, OpMaxPool2D, OpAvgPool2D, OpBatchNorm, OpFlatten,
		OpReshape, OpQuantize, OpDequantize,
	}
	for _, op := range ops {
		if _, err := GetOperator(op); err != nil {
			t.Errorf("GetOperator(%s): %v", op, err)
		}
	}
}

// ── Engine integration test (simple MLP) ──

func TestEngineMLPInference(t *testing.T) {
	model := buildSimpleMLP(t)
	engine, err := NewEngine(model)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	if engine.ArenaUsed() == 0 {
		t.Error("arena should be non-empty")
	}
	inputData := float32sToBytes([]float32{1.0, 0.5, -0.5, -1.0})
	output, err := engine.Infer(inputData)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if len(output) != 8 {
		t.Fatalf("output size: got %d, want 8", len(output))
	}
	outFloats := bytesToFloat32s(output)
	var sum float32
	for _, v := range outFloats {
		sum += v
		if v < 0 || v > 1 {
			t.Errorf("softmax output out of range: %f", v)
		}
	}
	if d := math.Abs(float64(sum - 1.0)); d > 1e-4 {
		t.Errorf("softmax sum: %f", sum)
	}
}

func TestEngineTensorInference(t *testing.T) {
	model := buildSimpleMLP(t)
	engine, err := NewEngine(model)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	in := engine.InputTensors()[0]
	ind := in.Float32s()
	ind[0], ind[1], ind[2], ind[3] = 1.0, 0.5, -0.5, -1.0
	outs, err := engine.InferTensor()
	if err != nil {
		t.Fatalf("InferTensor: %v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outs))
	}
	var sum float32
	for _, v := range outs[0].Float32s() {
		sum += v
	}
	if d := math.Abs(float64(sum - 1.0)); d > 1e-4 {
		t.Errorf("softmax sum: %f", sum)
	}
}

func TestEngineBatchInference(t *testing.T) {
	model := buildSimpleMLP(t)
	engine, err := NewEngine(model)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	inputs := [][]byte{
		float32sToBytes([]float32{1, 0, 0, 0}),
		float32sToBytes([]float32{0, 1, 0, 0}),
		float32sToBytes([]float32{0, 0, 1, 0}),
	}
	outs, err := engine.InferBatch(inputs)
	if err != nil {
		t.Fatalf("InferBatch: %v", err)
	}
	if len(outs) != 3 {
		t.Fatalf("expected 3 outputs, got %d", len(outs))
	}
	for i, out := range outs {
		fs := bytesToFloat32s(out)
		var sum float32
		for _, v := range fs {
			sum += v
		}
		if d := math.Abs(float64(sum - 1.0)); d > 1e-4 {
			t.Errorf("batch[%d] softmax sum: %f", i, sum)
		}
	}
}

// ── Helpers ──

func buildSimpleMLP(t *testing.T) *Model {
	t.Helper()
	tensorNames := []string{
		"input", "w1", "b1", "h1", "h1_relu", "w2", "b2", "output",
	}
	graph := []OpNode{
		{Type: OpDense, InputIndices: []int{0, 1, 2}, OutputIndices: []int{3}},
		{Type: OpReLU, InputIndices: []int{3}, OutputIndices: []int{4}},
		{Type: OpDense, InputIndices: []int{4, 5, 6}, OutputIndices: []int{7}},
		{Type: OpSoftmax, InputIndices: []int{7}, OutputIndices: []int{7}},
	}
	w1 := []float32{0.1, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.1, 0.2, 0.2, 0.3, -0.4}
	b1 := []float32{0.01, -0.01, 0.02}
	w2 := []float32{0.5, -0.3, -0.2, 0.4, 0.1, 0.3}
	b2 := []float32{0.01, -0.01}

	var weights []float32
	weights = append(weights, w1...)
	weights = append(weights, b1...)
	weights = append(weights, w2...)
	weights = append(weights, b2...)
	blob := float32sToBytes(weights)

	return &Model{
		Metadata: Metadata{
			Name:         "test-mlp",
			InputShapes:  []Shape{{Dims: []int{1, 4}}},
			OutputShapes: []Shape{{Dims: []int{1, 2}}},
		},
		TensorNames: tensorNames,
		TensorShapes: map[string]Shape{
			"w1": {Dims: []int{4, 3}},
			"b1": {Dims: []int{3}},
			"w2": {Dims: []int{3, 2}},
			"b2": {Dims: []int{2}},
		},
		Graph:       graph,
		WeightsBlob: blob,
	}
}

func float32sToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		b[i*4+0] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToFloat32s(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(bits)
	}
	return f
}
