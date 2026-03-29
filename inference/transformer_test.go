package inference

import (
	"math"
	"testing"
)

// ── GELU tests ──

func TestGELU(t *testing.T) {
	arena := NewInferenceArena(8*4*2 + 128)
	in, _ := arena.AllocTensor("in", []int{8}, Float32)
	out, _ := arena.AllocTensor("out", []int{8}, Float32)

	// Test values: 0, 1, -1, 2, -2, 0.5, -0.5, 3
	inData := in.Float32s()
	inData[0] = 0
	inData[1] = 1
	inData[2] = -1
	inData[3] = 2
	inData[4] = -2
	inData[5] = 0.5
	inData[6] = -0.5
	inData[7] = 3

	op := &geluOp{}
	if err := op.Execute([]*Tensor{in}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()

	// Expected values (tanh approximation)
	// GELU(0)  = 0
	// GELU(1)  ≈ 0.8412
	// GELU(-1) ≈ -0.1588
	// GELU(2)  ≈ 1.9545
	// GELU(-2) ≈ -0.0455
	expected := []float32{0.0, 0.8412, -0.1588, 1.9545, -0.0455, 0.3457, -0.1543, 2.9960}

	for i, exp := range expected {
		if diff := math.Abs(float64(outData[i] - exp)); diff > 0.01 {
			t.Errorf("GELU[%d]: got %f, want ~%f (diff=%f)", i, outData[i], exp, diff)
		}
	}
}

func TestGELU_ZeroAllocs(t *testing.T) {
	arena := NewInferenceArena(256*4*2 + 128)
	in, _ := arena.AllocTensor("in", []int{256}, Float32)
	out, _ := arena.AllocTensor("out", []int{256}, Float32)
	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i) * 0.01
	}
	op := &geluOp{}
	allocs := testing.AllocsPerRun(100, func() {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	})
	if allocs > 0 {
		t.Errorf("GELU Execute allocated: %f allocs", allocs)
	}
}

// ── LayerNorm tests ──

func TestLayerNorm(t *testing.T) {
	// 2 rows × 4 features
	rows, cols := 2, 4
	arenaSize := (rows*cols + cols + cols + rows*cols) * 4
	arena := NewInferenceArena(arenaSize + 512)

	x, _ := arena.AllocTensor("x", []int{rows, cols}, Float32)
	gamma, _ := arena.AllocTensor("gamma", []int{cols}, Float32)
	beta, _ := arena.AllocTensor("beta", []int{cols}, Float32)
	out, _ := arena.AllocTensor("out", []int{rows, cols}, Float32)

	// Input: [[1, 2, 3, 4], [2, 4, 6, 8]]
	xData := x.Float32s()
	xData[0], xData[1], xData[2], xData[3] = 1, 2, 3, 4
	xData[4], xData[5], xData[6], xData[7] = 2, 4, 6, 8

	// gamma=1, beta=0
	gData := gamma.Float32s()
	for i := range gData {
		gData[i] = 1.0
	}
	bData := beta.Float32s()
	for i := range bData {
		bData[i] = 0.0
	}

	op := &layerNormOp{}
	if err := op.Execute([]*Tensor{x, gamma, beta}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()

	// Row 0: mean=2.5, var=1.25, std=sqrt(1.25+1e-5)≈1.1180
	// Normalized: [-1.342, -0.447, 0.447, 1.342]
	// Row 1: mean=5.0, var=5.0, std=sqrt(5+1e-5)≈2.2361
	// Normalized: [-1.342, -0.447, 0.447, 1.342]

	for r := 0; r < rows; r++ {
		row := outData[r*cols : (r+1)*cols]
		// Check mean ≈ 0
		var sum float32
		for _, v := range row {
			sum += v
		}
		mean := sum / float32(cols)
		if math.Abs(float64(mean)) > 1e-4 {
			t.Errorf("row %d: mean = %f, want ~0", r, mean)
		}
		// Check variance ≈ 1
		var varSum float32
		for _, v := range row {
			d := v - mean
			varSum += d * d
		}
		variance := varSum / float32(cols)
		if math.Abs(float64(variance)-1.0) > 0.01 {
			t.Errorf("row %d: variance = %f, want ~1", r, variance)
		}
	}
}

func TestLayerNorm_3D(t *testing.T) {
	// [batch=1, seq=2, hidden=4]
	arena := NewInferenceArena((1*2*4+4+4+1*2*4)*4 + 512)
	x, _ := arena.AllocTensor("x", []int{1, 2, 4}, Float32)
	gamma, _ := arena.AllocTensor("gamma", []int{4}, Float32)
	beta, _ := arena.AllocTensor("beta", []int{4}, Float32)
	out, _ := arena.AllocTensor("out", []int{1, 2, 4}, Float32)

	xData := x.Float32s()
	for i := range xData {
		xData[i] = float32(i + 1)
	}
	for i := range gamma.Float32s() {
		gamma.Float32s()[i] = 1.0
	}

	op := &layerNormOp{}
	if err := op.Execute([]*Tensor{x, gamma, beta}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	// Verify output is not all zeros
	outData := out.Float32s()
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("LayerNorm 3D output is all zeros")
	}
}

func TestLayerNorm_ZeroAllocs(t *testing.T) {
	arena := NewInferenceArena((128*768+768+768+128*768)*4 + 1024)
	x, _ := arena.AllocTensor("x", []int{128, 768}, Float32)
	gamma, _ := arena.AllocTensor("gamma", []int{768}, Float32)
	beta, _ := arena.AllocTensor("beta", []int{768}, Float32)
	out, _ := arena.AllocTensor("out", []int{128, 768}, Float32)

	for i := range gamma.Float32s() {
		gamma.Float32s()[i] = 1.0
	}

	op := &layerNormOp{}
	allocs := testing.AllocsPerRun(100, func() {
		op.Execute([]*Tensor{x, gamma, beta}, []*Tensor{out})
	})
	if allocs > 0 {
		t.Errorf("LayerNorm Execute allocated: %f allocs", allocs)
	}
}

// ── Gather tests ──

func TestGather(t *testing.T) {
	vocabSize, embedDim := 10, 4
	seqLen := 3
	arenaSize := (vocabSize*embedDim + seqLen + seqLen*embedDim) * 4
	arena := NewInferenceArena(arenaSize + 512)

	weights, _ := arena.AllocTensor("w", []int{vocabSize, embedDim}, Float32)
	indices, _ := arena.AllocTensor("idx", []int{seqLen}, Float32) // Int32 data stored in float32-sized arena
	out, _ := arena.AllocTensor("out", []int{seqLen, embedDim}, Float32)

	// Fill weight matrix: row i = [i*10, i*10+1, i*10+2, i*10+3]
	wData := weights.Float32s()
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embedDim; j++ {
			wData[i*embedDim+j] = float32(i*10 + j)
		}
	}

	// Set indices: [2, 5, 7]
	idxData := indices.Int32s()
	idxData[0] = 2
	idxData[1] = 5
	idxData[2] = 7

	op := &gatherOp{}
	if err := op.Execute([]*Tensor{weights, indices}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()
	// Row 0 should be weights[2] = [20, 21, 22, 23]
	for j := 0; j < embedDim; j++ {
		if outData[j] != float32(20+j) {
			t.Errorf("gather[0][%d]: got %f, want %f", j, outData[j], float32(20+j))
		}
	}
	// Row 1 should be weights[5] = [50, 51, 52, 53]
	for j := 0; j < embedDim; j++ {
		if outData[embedDim+j] != float32(50+j) {
			t.Errorf("gather[1][%d]: got %f, want %f", j, outData[embedDim+j], float32(50+j))
		}
	}
	// Row 2 should be weights[7] = [70, 71, 72, 73]
	for j := 0; j < embedDim; j++ {
		if outData[2*embedDim+j] != float32(70+j) {
			t.Errorf("gather[2][%d]: got %f, want %f", j, outData[2*embedDim+j], float32(70+j))
		}
	}
}

func TestGather_ZeroAllocs(t *testing.T) {
	arena := NewInferenceArena((50000*768+128+128*768)*4 + 1024)
	weights, _ := arena.AllocTensor("w", []int{50000, 768}, Float32)
	indices, _ := arena.AllocTensor("idx", []int{128}, Float32)
	out, _ := arena.AllocTensor("out", []int{128, 768}, Float32)

	_ = weights
	// Set valid indices
	for i := range indices.Int32s() {
		indices.Int32s()[i] = int32(i % 50000)
	}

	op := &gatherOp{}
	allocs := testing.AllocsPerRun(10, func() {
		op.Execute([]*Tensor{weights, indices}, []*Tensor{out})
	})
	if allocs > 0 {
		t.Errorf("Gather Execute allocated: %f allocs", allocs)
	}
}

// ── BatchedMatMul tests ──

func TestBatchedMatMul(t *testing.T) {
	batch, m, k, n := 2, 3, 4, 5
	arenaSize := (batch*m*k + batch*k*n + batch*m*n) * 4
	arena := NewInferenceArena(arenaSize + 512)

	a, _ := arena.AllocTensor("a", []int{batch, m, k}, Float32)
	b, _ := arena.AllocTensor("b", []int{batch, k, n}, Float32)
	c, _ := arena.AllocTensor("c", []int{batch, m, n}, Float32)

	// Fill with known values
	aData := a.Float32s()
	bData := b.Float32s()
	for i := range aData {
		aData[i] = 1.0
	}
	for i := range bData {
		bData[i] = 1.0
	}

	op := &batchedMatMulOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatal(err)
	}

	// With all-ones: each element of C should be k (=4)
	cData := c.Float32s()
	for i, v := range cData {
		if v != float32(k) {
			t.Errorf("BMM[%d]: got %f, want %f", i, v, float32(k))
			break
		}
	}
}

func TestBatchedMatMul_ZeroAllocs(t *testing.T) {
	batch, m, k, n := 12, 128, 64, 128
	arenaSize := (batch*m*k + batch*k*n + batch*m*n) * 4
	arena := NewInferenceArena(arenaSize + 1024)

	a, _ := arena.AllocTensor("a", []int{batch, m, k}, Float32)
	b, _ := arena.AllocTensor("b", []int{batch, k, n}, Float32)
	c, _ := arena.AllocTensor("c", []int{batch, m, n}, Float32)

	_ = a
	_ = b

	op := &batchedMatMulOp{}
	allocs := testing.AllocsPerRun(5, func() {
		op.Execute([]*Tensor{a, b}, []*Tensor{c})
	})
	if allocs > 0 {
		t.Errorf("BatchedMatMul Execute allocated: %f allocs", allocs)
	}
}

// ── Mul tests ──

func TestMul(t *testing.T) {
	arena := NewInferenceArena(4*4*3 + 256)
	a, _ := arena.AllocTensor("a", []int{4}, Float32)
	b, _ := arena.AllocTensor("b", []int{4}, Float32)
	c, _ := arena.AllocTensor("c", []int{4}, Float32)

	aData := a.Float32s()
	bData := b.Float32s()
	aData[0], aData[1], aData[2], aData[3] = 2, 3, 4, 5
	bData[0], bData[1], bData[2], bData[3] = 10, 20, 30, 40

	op := &mulOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatal(err)
	}

	cData := c.Float32s()
	expected := []float32{20, 60, 120, 200}
	for i, exp := range expected {
		if cData[i] != exp {
			t.Errorf("Mul[%d]: got %f, want %f", i, cData[i], exp)
		}
	}
}

// ── Sub tests ──

func TestSub(t *testing.T) {
	arena := NewInferenceArena(4*4*3 + 256)
	a, _ := arena.AllocTensor("a", []int{4}, Float32)
	b, _ := arena.AllocTensor("b", []int{4}, Float32)
	c, _ := arena.AllocTensor("c", []int{4}, Float32)

	aData := a.Float32s()
	bData := b.Float32s()
	aData[0], aData[1], aData[2], aData[3] = 10, 20, 30, 40
	bData[0], bData[1], bData[2], bData[3] = 1, 2, 3, 4

	op := &subOp{}
	if err := op.Execute([]*Tensor{a, b}, []*Tensor{c}); err != nil {
		t.Fatal(err)
	}

	cData := c.Float32s()
	expected := []float32{9, 18, 27, 36}
	for i, exp := range expected {
		if cData[i] != exp {
			t.Errorf("Sub[%d]: got %f, want %f", i, cData[i], exp)
		}
	}
}

// ── Tanh tests ──

func TestTanh(t *testing.T) {
	arena := NewInferenceArena(4*4*2 + 256)
	in, _ := arena.AllocTensor("in", []int{4}, Float32)
	out, _ := arena.AllocTensor("out", []int{4}, Float32)

	inData := in.Float32s()
	inData[0], inData[1], inData[2], inData[3] = 0, 1, -1, 2

	op := &tanhOp{}
	if err := op.Execute([]*Tensor{in}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()
	expected := []float64{0.0, math.Tanh(1), math.Tanh(-1), math.Tanh(2)}
	for i, exp := range expected {
		if diff := math.Abs(float64(outData[i]) - exp); diff > 1e-6 {
			t.Errorf("Tanh[%d]: got %f, want %f", i, outData[i], exp)
		}
	}
}

// ── Where tests ──

func TestWhere(t *testing.T) {
	arena := NewInferenceArena(4*4*4 + 256)
	cond, _ := arena.AllocTensor("cond", []int{4}, Float32)
	a, _ := arena.AllocTensor("a", []int{4}, Float32)
	b, _ := arena.AllocTensor("b", []int{4}, Float32)
	out, _ := arena.AllocTensor("out", []int{4}, Float32)

	condData := cond.Float32s()
	aData := a.Float32s()
	bData := b.Float32s()

	condData[0], condData[1], condData[2], condData[3] = 1, 0, 1, 0
	aData[0], aData[1], aData[2], aData[3] = 10, 20, 30, 40
	bData[0], bData[1], bData[2], bData[3] = -1, -2, -3, -4

	op := &whereOp{}
	if err := op.Execute([]*Tensor{cond, a, b}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()
	expected := []float32{10, -2, 30, -4}
	for i, exp := range expected {
		if outData[i] != exp {
			t.Errorf("Where[%d]: got %f, want %f", i, outData[i], exp)
		}
	}
}

// ── Split tests ──

func TestSplit(t *testing.T) {
	// Split [6, 4] along axis=0 into 3 parts of [2, 4]
	arena := NewInferenceArena((6*4+2*4*3)*4 + 512)
	src, _ := arena.AllocTensor("src", []int{6, 4}, Float32)
	out0, _ := arena.AllocTensor("o0", []int{2, 4}, Float32)
	out1, _ := arena.AllocTensor("o1", []int{2, 4}, Float32)
	out2, _ := arena.AllocTensor("o2", []int{2, 4}, Float32)

	srcData := src.Float32s()
	for i := range srcData {
		srcData[i] = float32(i)
	}

	op := &splitOp{}
	if err := op.Execute([]*Tensor{src}, []*Tensor{out0, out1, out2}); err != nil {
		t.Fatal(err)
	}

	// out0 should be [0,1,2,3, 4,5,6,7]
	o0 := out0.Float32s()
	for i := 0; i < 8; i++ {
		if o0[i] != float32(i) {
			t.Errorf("Split out0[%d]: got %f, want %f", i, o0[i], float32(i))
		}
	}
	// out1 should be [8,9,10,11, 12,13,14,15]
	o1 := out1.Float32s()
	for i := 0; i < 8; i++ {
		if o1[i] != float32(i+8) {
			t.Errorf("Split out1[%d]: got %f, want %f", i, o1[i], float32(i+8))
		}
	}
	// out2 should be [16,17,18,19, 20,21,22,23]
	o2 := out2.Float32s()
	for i := 0; i < 8; i++ {
		if o2[i] != float32(i+16) {
			t.Errorf("Split out2[%d]: got %f, want %f", i, o2[i], float32(i+16))
		}
	}
}

// ── Shape inference tests ──

func TestShapeInference_Transformer(t *testing.T) {
	tests := []struct {
		name   string
		op     OpType
		inputs []Shape
		want   []Shape
	}{
		{
			name:   "GELU",
			op:     OpGELU,
			inputs: []Shape{{Dims: []int{1, 128, 768}}},
			want:   []Shape{{Dims: []int{1, 128, 768}}},
		},
		{
			name:   "Tanh",
			op:     OpTanh,
			inputs: []Shape{{Dims: []int{4, 64}}},
			want:   []Shape{{Dims: []int{4, 64}}},
		},
		{
			name:   "LayerNorm",
			op:     OpLayerNorm,
			inputs: []Shape{{Dims: []int{1, 128, 768}}, {Dims: []int{768}}, {Dims: []int{768}}},
			want:   []Shape{{Dims: []int{1, 128, 768}}},
		},
		{
			name:   "Mul",
			op:     OpMul,
			inputs: []Shape{{Dims: []int{8, 64}}, {Dims: []int{8, 64}}},
			want:   []Shape{{Dims: []int{8, 64}}},
		},
		{
			name:   "Sub",
			op:     OpSub,
			inputs: []Shape{{Dims: []int{8, 64}}, {Dims: []int{8, 64}}},
			want:   []Shape{{Dims: []int{8, 64}}},
		},
		{
			name:   "Gather",
			op:     OpGather,
			inputs: []Shape{{Dims: []int{50000, 768}}, {Dims: []int{128}}},
			want:   []Shape{{Dims: []int{128, 768}}},
		},
		{
			name:   "BatchedMatMul",
			op:     OpBatchedMatMul,
			inputs: []Shape{{Dims: []int{12, 128, 64}}, {Dims: []int{12, 64, 128}}},
			want:   []Shape{{Dims: []int{12, 128, 128}}},
		},
		{
			name:   "Where",
			op:     OpWhere,
			inputs: []Shape{{Dims: []int{128, 128}}, {Dims: []int{128, 128}}, {Dims: []int{128, 128}}},
			want:   []Shape{{Dims: []int{128, 128}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := inferOpOutputShapes(tc.op, tc.inputs, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %d shapes, want %d", len(got), len(tc.want))
			}
			for i, g := range got {
				if !g.Equal(tc.want[i]) {
					t.Errorf("shape[%d]: got %v, want %v", i, g.Dims, tc.want[i].Dims)
				}
			}
		})
	}
}

// ── RoPE tests ──

func TestRoPE_Basic(t *testing.T) {
	// 4D input: [batch=1, seq_len=4, num_heads=2, head_dim=4]
	batch, seqLen, numHeads, headDim := 1, 4, 2, 4
	total := batch * seqLen * numHeads * headDim
	arenaSize := total * 4 * 2 // input + output
	arena := NewInferenceArena(arenaSize + 512)

	in, _ := arena.AllocTensor("in", []int{batch, seqLen, numHeads, headDim}, Float32)
	out, _ := arena.AllocTensor("out", []int{batch, seqLen, numHeads, headDim}, Float32)

	// Fill with sequential values
	inData := in.Float32s()
	for i := range inData {
		inData[i] = float32(i) * 0.1
	}

	op := &ropeOp{baseFreq: 10000.0}
	if err := op.Execute([]*Tensor{in}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	outData := out.Float32s()

	// Position 0: cos(0)=1, sin(0)=0, so output should match input
	for h := 0; h < numHeads; h++ {
		off := h * headDim
		for d := 0; d < headDim; d++ {
			if diff := math.Abs(float64(outData[off+d] - inData[off+d])); diff > 1e-5 {
				t.Errorf("RoPE pos=0 head=%d dim=%d: got %f, want %f", h, d, outData[off+d], inData[off+d])
			}
		}
	}

	// Positions > 0 should differ from input (rotation applied)
	seqStride := numHeads * headDim
	for s := 1; s < seqLen; s++ {
		off := s * seqStride
		same := 0
		for d := 0; d < numHeads*headDim; d++ {
			if outData[off+d] == inData[off+d] {
				same++
			}
		}
		if same == numHeads*headDim {
			t.Errorf("RoPE pos=%d: output identical to input (rotation not applied)", s)
		}
	}
}

func TestRoPE_3D(t *testing.T) {
	// 3D input: [batch=1, seq_len=4, dim=8]
	batch, seqLen, dim := 1, 4, 8
	total := batch * seqLen * dim
	arenaSize := total * 4 * 2
	arena := NewInferenceArena(arenaSize + 512)

	in, _ := arena.AllocTensor("in", []int{batch, seqLen, dim}, Float32)
	out, _ := arena.AllocTensor("out", []int{batch, seqLen, dim}, Float32)

	inData := in.Float32s()
	for i := range inData {
		inData[i] = 1.0
	}

	op := &ropeOp{baseFreq: 10000.0}
	if err := op.Execute([]*Tensor{in}, []*Tensor{out}); err != nil {
		t.Fatal(err)
	}

	// Output shape should be preserved
	outShape := out.Shape()
	if len(outShape) != 3 || outShape[0] != 1 || outShape[1] != 4 || outShape[2] != 8 {
		t.Errorf("RoPE 3D output shape: got %v", outShape)
	}
}

func TestRoPE_OddHeadDim_Error(t *testing.T) {
	arena := NewInferenceArena(1024)
	in, _ := arena.AllocTensor("in", []int{1, 4, 3}, Float32) // dim=3 (odd)
	out, _ := arena.AllocTensor("out", []int{1, 4, 3}, Float32)

	op := &ropeOp{baseFreq: 10000.0}
	err := op.Execute([]*Tensor{in}, []*Tensor{out})
	if err == nil {
		t.Error("expected error for odd head_dim")
	}
}

func TestRoPE_ShapeInference(t *testing.T) {
	shape := Shape{Dims: []int{1, 128, 12, 64}}
	got, err := inferOpOutputShapes(OpRoPE, []Shape{shape}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !got[0].Equal(shape) {
		t.Errorf("RoPE shape: got %v, want %v", got[0].Dims, shape.Dims)
	}
}

func TestRoPE_DTypeAware(t *testing.T) {
	op := &ropeOp{}
	info := &DTypeInfo{
		InputDTypes: []DType{Float16},
	}
	if err := op.SetDTypeInfo(info); err != nil {
		t.Fatal(err)
	}
	if op.dtype != Float16 {
		t.Errorf("expected dtype Float16, got %v", op.dtype)
	}
}

func TestRoPE_SetAttrs(t *testing.T) {
	op := &ropeOp{}
	// No attrs — default base freq
	if err := op.SetAttrs(nil); err != nil {
		t.Fatal(err)
	}
	if op.baseFreq != 10000.0 {
		t.Errorf("default baseFreq: got %f, want 10000.0", op.baseFreq)
	}

	// Custom base freq
	attrs := make([]byte, 4)
	bits := math.Float32bits(1000.0)
	attrs[0] = byte(bits)
	attrs[1] = byte(bits >> 8)
	attrs[2] = byte(bits >> 16)
	attrs[3] = byte(bits >> 24)
	if err := op.SetAttrs(attrs); err != nil {
		t.Fatal(err)
	}
	if op.baseFreq != 1000.0 {
		t.Errorf("custom baseFreq: got %f, want 1000.0", op.baseFreq)
	}
}

func TestRoPE_ZeroAllocs(t *testing.T) {
	total := 1 * 32 * 4 * 64 // batch * seq * heads * dim
	arena := NewInferenceArena(total*4*2 + 512)
	in, _ := arena.AllocTensor("in", []int{1, 32, 4, 64}, Float32)
	out, _ := arena.AllocTensor("out", []int{1, 32, 4, 64}, Float32)

	for i := range in.Float32s() {
		in.Float32s()[i] = float32(i) * 0.001
	}

	op := &ropeOp{baseFreq: 10000.0}
	allocs := testing.AllocsPerRun(50, func() {
		op.Execute([]*Tensor{in}, []*Tensor{out})
	})
	if allocs > 0 {
		t.Errorf("RoPE Execute allocated: %f allocs", allocs)
	}
}

// ── Matryoshka slicing tests ──

func TestSliceLastDim(t *testing.T) {
	arena := NewInferenceArena(2*768*4 + 512)
	embed, _ := arena.AllocTensor("embed", []int{2, 768}, Float32)

	data := embed.Float32s()
	for i := range data {
		data[i] = float32(i)
	}

	// Slice to 256 dims
	sliced, err := embed.SliceLastDim(256)
	if err != nil {
		t.Fatal(err)
	}

	shape := sliced.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 256 {
		t.Fatalf("SliceLastDim shape: got %v, want [2, 256]", shape)
	}

	// Verify zero-copy: first element should be same pointer
	if sliced.DataPtr() != embed.DataPtr() {
		t.Error("SliceLastDim should be zero-copy (same base pointer)")
	}

	// Verify data via NarrowFloat32s: row 0 should be [0..255], row 1 should start at 768
	slicedData := sliced.NarrowFloat32s()
	for i := 0; i < 256; i++ {
		if slicedData[i] != float32(i) {
			t.Errorf("SliceLastDim row0[%d]: got %f, want %f", i, slicedData[i], float32(i))
			break
		}
	}
	for i := 0; i < 256; i++ {
		want := float32(768 + i)
		if slicedData[256+i] != want {
			t.Errorf("SliceLastDim row1[%d]: got %f, want %f", i, slicedData[256+i], want)
			break
		}
	}
}

func TestSliceLastDim_128(t *testing.T) {
	arena := NewInferenceArena(1*768*4 + 512)
	embed, _ := arena.AllocTensor("embed", []int{1, 768}, Float32)
	data := embed.Float32s()
	for i := range data {
		data[i] = float32(i)
	}

	sliced, err := embed.SliceLastDim(128)
	if err != nil {
		t.Fatal(err)
	}
	if sliced.Shape()[1] != 128 {
		t.Errorf("got dim %d, want 128", sliced.Shape()[1])
	}
}

func TestSliceLastDim_3D(t *testing.T) {
	arena := NewInferenceArena(2*4*768*4 + 512)
	t3d, _ := arena.AllocTensor("t3d", []int{2, 4, 768}, Float32)
	sliced, err := t3d.SliceLastDim(256)
	if err != nil {
		t.Fatal(err)
	}
	shape := sliced.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 4 || shape[2] != 256 {
		t.Errorf("SliceLastDim 3D: got %v, want [2, 4, 256]", shape)
	}
}

func TestSliceLastDim_Errors(t *testing.T) {
	arena := NewInferenceArena(1*768*4 + 512)
	embed, _ := arena.AllocTensor("embed", []int{1, 768}, Float32)

	// count = 0
	if _, err := embed.SliceLastDim(0); err == nil {
		t.Error("expected error for count=0")
	}
	// count > dim
	if _, err := embed.SliceLastDim(1000); err == nil {
		t.Error("expected error for count > dim")
	}
	// count < 0
	if _, err := embed.SliceLastDim(-1); err == nil {
		t.Error("expected error for negative count")
	}
}

func TestSliceLastDim_FullDim(t *testing.T) {
	arena := NewInferenceArena(1*768*4 + 512)
	embed, _ := arena.AllocTensor("embed", []int{1, 768}, Float32)

	sliced, err := embed.SliceLastDim(768)
	if err != nil {
		t.Fatal(err)
	}
	if sliced.Shape()[1] != 768 {
		t.Errorf("full dim: got %d, want 768", sliced.Shape()[1])
	}
}

// ── Arena overflow safety tests ──

func TestArenaInt64Strides(t *testing.T) {
	// Test that computeStrides handles large dimensions without int overflow.
	// A tensor with shape [1, 8192, 1024, 1024] would have strides that
	// require int64 intermediates on 32-bit systems.
	shape := []int{1, 8192, 1024}
	strides := computeStrides(shape, Float32)

	// stride[2] = 4 (float32 size)
	// stride[1] = 4 * 1024 = 4096
	// stride[0] = 4096 * 8192 = 33554432
	if strides[2] != 4 {
		t.Errorf("stride[2]: got %d, want 4", strides[2])
	}
	if strides[1] != 4*1024 {
		t.Errorf("stride[1]: got %d, want %d", strides[1], 4*1024)
	}
	if strides[0] != 4*1024*8192 {
		t.Errorf("stride[0]: got %d, want %d", strides[0], 4*1024*8192)
	}
}

func TestSetShape_Int64Safety(t *testing.T) {
	// Allocate a tensor with a large shape, then SetShape to a smaller one
	arena := NewInferenceArena(1024 * 1024) // 1MB
	t1, err := arena.AllocTensor("big", []int{1, 256, 1024}, Float32)
	if err != nil {
		t.Fatal(err)
	}

	// Shrink to smaller shape
	if err := t1.SetShape([]int{1, 128, 1024}); err != nil {
		t.Errorf("SetShape to smaller should succeed: %v", err)
	}

	// Try to expand beyond allocation
	if err := t1.SetShape([]int{1, 512, 1024}); err == nil {
		t.Error("SetShape to larger should fail")
	}
}

// ── Dynamic context window tests ──

func TestDynamicContextWindow_8192(t *testing.T) {
	// Verify that the engine supports allocating and reshaping tensors
	// up to 8192 token context windows.
	maxSeq := 8192
	embedDim := 768

	// Allocate arena big enough for max context
	arenaSize := maxSeq * embedDim * 4 * 2 // input + output
	arena := NewInferenceArena(arenaSize + 1024)

	// Allocate at max size
	in, err := arena.AllocTensor("in", []int{1, maxSeq, embedDim}, Float32)
	if err != nil {
		t.Fatalf("AllocTensor at maxSeq=%d: %v", maxSeq, err)
	}

	// SetShape to actual runtime size
	if err := in.SetShape([]int{1, 128, embedDim}); err != nil {
		t.Fatalf("SetShape to 128 tokens: %v", err)
	}
	if in.Shape()[1] != 128 {
		t.Errorf("seq_len after reshape: got %d, want 128", in.Shape()[1])
	}

	// Reshape back to a different size
	if err := in.SetShape([]int{1, 4096, embedDim}); err != nil {
		t.Fatalf("SetShape to 4096 tokens: %v", err)
	}
	if in.Shape()[1] != 4096 {
		t.Errorf("seq_len after reshape: got %d, want 4096", in.Shape()[1])
	}

	// Reshape to full 8192
	if err := in.SetShape([]int{1, maxSeq, embedDim}); err != nil {
		t.Fatalf("SetShape to %d tokens: %v", maxSeq, err)
	}
	if in.Shape()[1] != maxSeq {
		t.Errorf("seq_len after reshape: got %d, want %d", in.Shape()[1], maxSeq)
	}
}
