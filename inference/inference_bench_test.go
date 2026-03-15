package inference

import (
	"math"
	"testing"
)

// ── MatMul benchmarks at various sizes ──

func benchMatMul(b *testing.B, M, K, N int) {
	arena := NewInferenceArena(M*K*4 + K*N*4 + M*N*4 + 3*64)
	a, _ := arena.AllocTensor("A", []int{M, K}, Float32)
	bTensor, _ := arena.AllocTensor("B", []int{K, N}, Float32)
	c, _ := arena.AllocTensor("C", []int{M, N}, Float32)
	for i := range a.Float32s() {
		a.Float32s()[i] = 0.01
	}
	for i := range bTensor.Float32s() {
		bTensor.Float32s()[i] = 0.01
	}
	op := &matMulOp{}
	b.SetBytes(int64(M*K+K*N+M*N) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{a, bTensor}, []*Tensor{c})
	}
}

func BenchmarkMatMul_4x4(b *testing.B)     { benchMatMul(b, 4, 4, 4) }
func BenchmarkMatMul_16x16(b *testing.B)   { benchMatMul(b, 16, 16, 16) }
func BenchmarkMatMul_64x64(b *testing.B)   { benchMatMul(b, 64, 64, 64) }
func BenchmarkMatMul_128x128(b *testing.B) { benchMatMul(b, 128, 128, 128) }
func BenchmarkMatMul_256x256(b *testing.B) { benchMatMul(b, 256, 256, 256) }
func BenchmarkMatMul_512x512(b *testing.B) { benchMatMul(b, 512, 512, 512) }

// ── MatMul large-matrix benchmarks (SIMD / platform-optimized path) ──

func BenchmarkMatMul_1024x1024(b *testing.B) { benchMatMul(b, 1024, 1024, 1024) }

// BenchmarkMatMul_NonSquare tests non-square matrices to exercise
// the micro-kernel remainder handling in the SIMD path.
func BenchmarkMatMul_127x255(b *testing.B) { benchMatMul(b, 127, 255, 63) }
func BenchmarkMatMul_5x5(b *testing.B)     { benchMatMul(b, 5, 5, 5) }

// ── Dense (MatMul + Bias) benchmark ──

func BenchmarkDense_128x64(b *testing.B) {
	arena := NewInferenceArena(128*64*4 + 64*32*4 + 32*4 + 128*32*4 + 4*64)
	x, _ := arena.AllocTensor("x", []int{128, 64}, Float32)
	w, _ := arena.AllocTensor("w", []int{64, 32}, Float32)
	bias, _ := arena.AllocTensor("b", []int{32}, Float32)
	out, _ := arena.AllocTensor("out", []int{128, 32}, Float32)
	for i := range x.Float32s() {
		x.Float32s()[i] = 0.01
	}
	for i := range w.Float32s() {
		w.Float32s()[i] = 0.01
	}
	op := &denseOp{}
	b.SetBytes(int64(128*64+64*32+128*32) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{x, w, bias}, []*Tensor{out})
	}
}

// ── Activation benchmarks ──

func benchActivation(b *testing.B, op Operator, n int) {
	arena := NewInferenceArena(n*4*2 + 2*64)
	in, _ := arena.AllocTensor("in", []int{n}, Float32)
	out, _ := arena.AllocTensor("out", []int{n}, Float32)
	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i%20) - 10.0
	}
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	}
}

func BenchmarkReLU_1024(b *testing.B)    { benchActivation(b, &reluOp{}, 1024) }
func BenchmarkReLU_65536(b *testing.B)   { benchActivation(b, &reluOp{}, 65536) }
func BenchmarkSigmoid_1024(b *testing.B) { benchActivation(b, &sigmoidOp{}, 1024) }

func BenchmarkSoftmax_128x10(b *testing.B) {
	arena := NewInferenceArena(128*10*4*2 + 2*64)
	in, _ := arena.AllocTensor("in", []int{128, 10}, Float32)
	out, _ := arena.AllocTensor("out", []int{128, 10}, Float32)
	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i%20) - 10.0
	}
	op := &softmaxOp{}
	b.SetBytes(128 * 10 * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	}
}

// ── Conv2D benchmark ──

func BenchmarkConv2D_32x32x3x16(b *testing.B) {
	arena := NewInferenceArena(1*3*32*32*4 + 16*3*3*3*4 + 1*16*30*30*4 + 3*64)
	input, _ := arena.AllocTensor("in", []int{1, 3, 32, 32}, Float32)
	kernel, _ := arena.AllocTensor("k", []int{16, 3, 3, 3}, Float32)
	output, _ := arena.AllocTensor("out", []int{1, 16, 30, 30}, Float32)
	for i := range input.Float32s() {
		input.Float32s()[i] = 0.01
	}
	for i := range kernel.Float32s() {
		kernel.Float32s()[i] = 0.01
	}
	op := &conv2dOp{}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{input, kernel}, []*Tensor{output})
	}
}

// ── Quantization benchmarks ──

func BenchmarkQuantizeSymmetric_1024(b *testing.B) {
	arena := NewInferenceArena(1024*4 + 1024 + 2*64)
	src, _ := arena.AllocTensor("src", []int{1024}, Float32)
	dst, _ := arena.AllocTensor("dst", []int{1024}, Int8)
	for i := range src.Float32s() {
		src.Float32s()[i] = float32(i%256-128) / 128.0
	}
	b.SetBytes(1024 * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		QuantizeSymmetric(src, dst)
	}
}

func BenchmarkMatMulInt8_64x64(b *testing.B) {
	arena := NewInferenceArena(64*64 + 64*64 + 64*64*4 + 3*64)
	a, _ := arena.AllocTensor("A", []int{64, 64}, Int8)
	bT, _ := arena.AllocTensor("B", []int{64, 64}, Int8)
	c, _ := arena.AllocTensor("C", []int{64, 64}, Int32)
	for i := range a.Int8s() {
		a.Int8s()[i] = int8(i % 127)
	}
	for i := range bT.Int8s() {
		bT.Int8s()[i] = int8(i % 127)
	}
	b.SetBytes(int64(64*64+64*64+64*64) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MatMulInt8(a, bT, c, 1.0, 1.0)
	}
}

// ── Engine MLP benchmark (end-to-end) ──

func BenchmarkEngine_MLP_Infer(b *testing.B) {
	model := buildBenchMLP()
	engine, err := NewEngine(model)
	if err != nil {
		b.Fatalf("NewEngine: %v", err)
	}
	input := make([]byte, 4*4) // 1x4 float32
	for i := 0; i < 4; i++ {
		bits := math.Float32bits(float32(i) * 0.1)
		input[i*4+0] = byte(bits)
		input[i*4+1] = byte(bits >> 8)
		input[i*4+2] = byte(bits >> 16)
		input[i*4+3] = byte(bits >> 24)
	}
	b.SetBytes(int64(len(input)))
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		engine.Infer(input)
	}
}

func BenchmarkEngine_MLP_InferTensor(b *testing.B) {
	model := buildBenchMLP()
	engine, err := NewEngine(model)
	if err != nil {
		b.Fatalf("NewEngine: %v", err)
	}
	in := engine.InputTensors()[0]
	in.Float32s()[0], in.Float32s()[1] = 0.1, 0.2
	in.Float32s()[2], in.Float32s()[3] = 0.3, 0.4
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		engine.InferTensor()
	}
}

// ── Arena alloc benchmark ──

func BenchmarkArenaAlloc_100Tensors(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		arena := NewInferenceArena(100 * 256 * 4)
		for j := 0; j < 100; j++ {
			arena.AllocTensor("t", []int{16, 16}, Float32)
		}
	}
}

func BenchmarkTensorZero_4096(b *testing.B) {
	arena := NewInferenceArena(4096*4 + 64)
	t, _ := arena.AllocTensor("t", []int{4096}, Float32)
	b.SetBytes(4096 * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		t.Zero()
	}
}

func buildBenchMLP() *Model {
	tensorNames := []string{
		"input", "w1", "b1", "h1", "h1_relu", "w2", "b2", "output",
	}
	graph := []OpNode{
		{Type: OpDense, InputIndices: []int{0, 1, 2}, OutputIndices: []int{3}},
		{Type: OpReLU, InputIndices: []int{3}, OutputIndices: []int{4}},
		{Type: OpDense, InputIndices: []int{4, 5, 6}, OutputIndices: []int{7}},
		{Type: OpSoftmax, InputIndices: []int{7}, OutputIndices: []int{7}},
	}
	w1 := make([]float32, 4*3)
	b1 := make([]float32, 3)
	w2 := make([]float32, 3*2)
	b2 := make([]float32, 2)
	for i := range w1 {
		w1[i] = 0.1
	}
	for i := range w2 {
		w2[i] = 0.1
	}
	var weights []float32
	weights = append(weights, w1...)
	weights = append(weights, b1...)
	weights = append(weights, w2...)
	weights = append(weights, b2...)

	blob := make([]byte, len(weights)*4)
	for i, v := range weights {
		bits := math.Float32bits(v)
		blob[i*4+0] = byte(bits)
		blob[i*4+1] = byte(bits >> 8)
		blob[i*4+2] = byte(bits >> 16)
		blob[i*4+3] = byte(bits >> 24)
	}

	return &Model{
		Metadata: Metadata{
			Name:         "bench-mlp",
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
