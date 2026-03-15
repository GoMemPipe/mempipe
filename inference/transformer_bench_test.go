package inference

import (
	"testing"
)

// ── GELU benchmarks ──

func benchGELU(b *testing.B, n int) {
	arena := NewInferenceArena(n*4*2 + 128)
	in, _ := arena.AllocTensor("in", []int{n}, Float32)
	out, _ := arena.AllocTensor("out", []int{n}, Float32)
	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i) * 0.001
	}
	op := &geluOp{}
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	}
}

func BenchmarkGELU_1024(b *testing.B)   { benchGELU(b, 1024) }
func BenchmarkGELU_98304(b *testing.B)  { benchGELU(b, 1*128*768) }  // GPT-2 sized
func BenchmarkGELU_393216(b *testing.B) { benchGELU(b, 1*128*3072) } // GPT-2 FFN sized

// ── LayerNorm benchmarks ──

func benchLayerNorm(b *testing.B, rows, cols int) {
	arenaSize := (rows*cols + cols + cols + rows*cols) * 4
	arena := NewInferenceArena(arenaSize + 1024)
	x, _ := arena.AllocTensor("x", []int{rows, cols}, Float32)
	gamma, _ := arena.AllocTensor("gamma", []int{cols}, Float32)
	beta, _ := arena.AllocTensor("beta", []int{cols}, Float32)
	out, _ := arena.AllocTensor("out", []int{rows, cols}, Float32)

	for i := range x.Float32s() {
		x.Float32s()[i] = float32(i) * 0.001
	}
	for i := range gamma.Float32s() {
		gamma.Float32s()[i] = 1.0
	}

	op := &layerNormOp{}
	b.SetBytes(int64(rows*cols) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{x, gamma, beta}, []*Tensor{out})
	}
}

func BenchmarkLayerNorm_128x768(b *testing.B) { benchLayerNorm(b, 128, 768) }  // GPT-2 seq×hidden
func BenchmarkLayerNorm_512x768(b *testing.B) { benchLayerNorm(b, 512, 768) }  // longer seq
func BenchmarkLayerNorm_128x1024(b *testing.B) { benchLayerNorm(b, 128, 1024) } // GPT-2 Medium

// ── Gather benchmarks ──

func benchGather(b *testing.B, vocabSize, embedDim, seqLen int) {
	arenaSize := (vocabSize*embedDim + seqLen + seqLen*embedDim) * 4
	arena := NewInferenceArena(arenaSize + 1024)

	weights, _ := arena.AllocTensor("w", []int{vocabSize, embedDim}, Float32)
	indices, _ := arena.AllocTensor("idx", []int{seqLen}, Float32)
	out, _ := arena.AllocTensor("out", []int{seqLen, embedDim}, Float32)

	_ = weights
	idxData := indices.Int32s()
	for i := range idxData {
		idxData[i] = int32(i % vocabSize)
	}

	op := &gatherOp{}
	b.SetBytes(int64(seqLen*embedDim) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{weights, indices}, []*Tensor{out})
	}
}

func BenchmarkGather_50257x768_128(b *testing.B) { benchGather(b, 50257, 768, 128) } // GPT-2
func BenchmarkGather_50257x768_512(b *testing.B) { benchGather(b, 50257, 768, 512) }
func BenchmarkGather_256x64_8(b *testing.B)      { benchGather(b, 256, 64, 8) }

// ── BatchedMatMul benchmarks ──

func benchBMM(b *testing.B, batch, m, k, n int) {
	arenaSize := (batch*m*k + batch*k*n + batch*m*n) * 4
	arena := NewInferenceArena(arenaSize + 1024)

	a, _ := arena.AllocTensor("a", []int{batch, m, k}, Float32)
	bt, _ := arena.AllocTensor("b", []int{batch, k, n}, Float32)
	c, _ := arena.AllocTensor("c", []int{batch, m, n}, Float32)

	for i := range a.Float32s() {
		a.Float32s()[i] = 0.01
	}
	for i := range bt.Float32s() {
		bt.Float32s()[i] = 0.01
	}

	op := &batchedMatMulOp{}
	b.SetBytes(int64(batch*(m*k+k*n+m*n)) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{a, bt}, []*Tensor{c})
	}
}

func BenchmarkBMM_12x128x64(b *testing.B)  { benchBMM(b, 12, 128, 64, 128) }  // GPT-2 QK^T
func BenchmarkBMM_12x128x128(b *testing.B) { benchBMM(b, 12, 128, 128, 64) }  // GPT-2 Attn×V
func BenchmarkBMM_4x64x16(b *testing.B)    { benchBMM(b, 4, 64, 16, 64) }     // tiny

// ── Mul benchmark ──

func BenchmarkMul_98304(b *testing.B) {
	n := 1 * 128 * 768
	arena := NewInferenceArena(n*4*3 + 256)
	a, _ := arena.AllocTensor("a", []int{n}, Float32)
	bt, _ := arena.AllocTensor("b", []int{n}, Float32)
	c, _ := arena.AllocTensor("c", []int{n}, Float32)
	for i := range a.Float32s() {
		a.Float32s()[i] = 0.01
	}
	for i := range bt.Float32s() {
		bt.Float32s()[i] = 0.01
	}
	op := &mulOp{}
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{a, bt}, []*Tensor{c})
	}
}

// ── Tanh benchmark ──

func BenchmarkTanh_98304(b *testing.B) {
	n := 1 * 128 * 768
	arena := NewInferenceArena(n*4*2 + 256)
	in, _ := arena.AllocTensor("in", []int{n}, Float32)
	out, _ := arena.AllocTensor("out", []int{n}, Float32)
	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i) * 0.001
	}
	op := &tanhOp{}
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	}
}
