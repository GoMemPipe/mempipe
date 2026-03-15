package inference

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// Operator executes a neural network operation on tensors.
// All implementations must be zero-allocation in Execute.
type Operator interface {
	// Execute runs the operator on the given input and output tensors.
	// All tensors are pre-allocated in the arena. Must be zero-alloc.
	Execute(inputs, outputs []*Tensor) error

	// OutputShape computes the output shape(s) given input shapes.
	OutputShape(inputShapes []Shape) ([]Shape, error)
}

// Initializable is an optional lifecycle interface for operators that
// require one-time setup (e.g. compiling GPU shaders, pre-allocating
// scratch buffers). If an Operator also implements Initializable, the
// Engine calls Init exactly once during model compilation — never on
// the hot path.
type Initializable interface {
	Init(arena *InferenceArena) error
}

// Attributable is an optional interface for operators that read
// encoded attributes from the OpNode.Attrs blob at compile time.
// If an Operator also implements Attributable, the Engine calls
// SetAttrs exactly once before the first Execute.
type Attributable interface {
	SetAttrs(attrs []byte) error
}

// ────────────────────────────────────────────────────────────────────────────
// Operator Registry
// ────────────────────────────────────────────────────────────────────────────

var (
	operatorsMu rwMutex
	operators   = map[OpType]func() Operator{}
)

// RegisterOperator registers a factory for the given operator type.
func RegisterOperator(opType OpType, factory func() Operator) {
	operatorsMu.Lock()
	operators[opType] = factory
	operatorsMu.Unlock()
}

// GetOperator returns a new instance of the operator for the given type.
func GetOperator(opType OpType) (Operator, error) {
	operatorsMu.RLock()
	factory, ok := operators[opType]
	operatorsMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("no operator registered for %s", opType)
	}
	return factory(), nil
}

// ────────────────────────────────────────────────────────────────────────────
// Built-in operators — auto-registered
// ────────────────────────────────────────────────────────────────────────────

func init() {
	RegisterOperator(OpMatMul, func() Operator { return &matMulOp{} })
	RegisterOperator(OpDense, func() Operator { return &denseOp{} })
	RegisterOperator(OpAdd, func() Operator { return &addOp{} })
	RegisterOperator(OpReLU, func() Operator { return &reluOp{} })
	RegisterOperator(OpSigmoid, func() Operator { return &sigmoidOp{} })
	RegisterOperator(OpSoftmax, func() Operator { return &softmaxOp{} })
	RegisterOperator(OpConv2D, func() Operator { return &conv2dOp{} })
	RegisterOperator(OpMaxPool2D, func() Operator { return &maxPool2dOp{} })
	RegisterOperator(OpAvgPool2D, func() Operator { return &avgPool2dOp{} })
	RegisterOperator(OpBatchNorm, func() Operator { return &batchNormOp{} })
	RegisterOperator(OpFlatten, func() Operator { return &flattenOp{} })
	RegisterOperator(OpReshape, func() Operator { return &reshapeOp{} })
	RegisterOperator(OpQuantize, func() Operator { return &quantizeOp{} })
	RegisterOperator(OpDequantize, func() Operator { return &dequantizeOp{} })
	// Transformer operators
	RegisterOperator(OpGELU, func() Operator { return &geluOp{} })
	RegisterOperator(OpLayerNorm, func() Operator { return &layerNormOp{} })
	RegisterOperator(OpGather, func() Operator { return &gatherOp{} })
	RegisterOperator(OpBatchedMatMul, func() Operator { return &batchedMatMulOp{} })
	RegisterOperator(OpMul, func() Operator { return &mulOp{} })
	RegisterOperator(OpSub, func() Operator { return &subOp{} })
	RegisterOperator(OpTranspose, func() Operator { return &transposeOp{} })
	RegisterOperator(OpSlice, func() Operator { return &sliceOp{} })
	RegisterOperator(OpTanh, func() Operator { return &tanhOp{} })
	RegisterOperator(OpWhere, func() Operator { return &whereOp{} })
	RegisterOperator(OpSplit, func() Operator { return &splitOp{} })
	RegisterOperator(OpDiv, func() Operator { return &divOp{} })
	RegisterOperator(OpPow, func() Operator { return &powOp{} })
	RegisterOperator(OpIsNaN, func() Operator { return &isNaNOp{} })
	RegisterOperator(OpAnd, func() Operator { return &andOp{} })
	RegisterOperator(OpGlobalAvgPool2D, func() Operator { return &globalAvgPool2dOp{} })
	RegisterOperator(OpHardSigmoid, func() Operator { return &hardSigmoidOp{} })
	RegisterOperator(OpHardSwish, func() Operator { return &hardSwishOp{} })
}

// ════════════════════════════════════════════════════════════════════════════
// MatMul: C = A × B    (float32, 2D)
// ════════════════════════════════════════════════════════════════════════════

// matMulOp implements the MatMul operator with optional hardware acceleration.
// The matMulAccel struct is defined per-platform in matmul_{generic,wasm,simd}.go.
type matMulOp struct {
	accel matMulAccel
}

// Init satisfies the Initializable interface so the Engine calls it
// during model compilation. Delegates to the platform-specific
// matMulAccel.Init (which e.g. sets up the WebGPU pipeline on WASM)
// and copies the result to the package-level singleton so that
// denseOp, batchedMatMulOp, and standalone callers also benefit.
func (op *matMulOp) Init(arena *InferenceArena) error {
	return initMatMulAccel(&op.accel, arena)
}

func (op *matMulOp) Execute(inputs, outputs []*Tensor) error {
	a, b, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	bData := b.Float32s()
	cData := c.Float32s()

	m := a.shape[len(a.shape)-2]
	k := a.shape[len(a.shape)-1]
	n := b.shape[len(b.shape)-1]

	matMulF32(aData, bData, cData, m, k, n)
	return nil
}

func (op *matMulOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpMatMul, in, nil)
}

// matMulF32 is defined per-platform in:
//   matmul_generic.go — fallback i-p-j loop
//   matmul_wasm.go    — WebGPU compute shader dispatch
//   matmul_simd.go    — BCE-optimized tiled kernel + optional assembly

// ════════════════════════════════════════════════════════════════════════════
// Dense: C = A × W + bias  (fused MatMul + BiasAdd)
// ════════════════════════════════════════════════════════════════════════════

type denseOp struct{}

func (denseOp) Execute(inputs, outputs []*Tensor) error {
	// inputs[0] = A [m, k], inputs[1] = W [k, n], inputs[2] = bias [n] (optional)
	a, w, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	wData := w.Float32s()
	cData := c.Float32s()

	m := a.shape[len(a.shape)-2]
	k := a.shape[len(a.shape)-1]
	n := w.shape[len(w.shape)-1]

	matMulF32(aData, wData, cData, m, k, n)

	// Add bias if present
	if len(inputs) > 2 {
		bias := inputs[2].Float32s()
		for i := 0; i < m; i++ {
			row := cData[i*n : i*n+n]
			for j := 0; j < n; j++ {
				row[j] += bias[j]
			}
		}
	}
	return nil
}

func (denseOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpDense, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Add: C = A + B  (element-wise, with broadcasting)
// ════════════════════════════════════════════════════════════════════════════

type addOp struct{}

func (addOp) Execute(inputs, outputs []*Tensor) error {
	a, b, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	bData := b.Float32s()
	cData := c.Float32s()

	na := len(aData)
	nb := len(bData)

	if nb == na {
		// Same size: element-wise
		for i := 0; i < na; i++ {
			cData[i] = aData[i] + bData[i]
		}
	} else if nb == 1 {
		scalar := bData[0]
		for i := 0; i < na; i++ {
			cData[i] = aData[i] + scalar
		}
	} else if na == 1 {
		scalar := aData[0]
		for i := 0; i < nb; i++ {
			cData[i] = scalar + bData[i]
		}
	} else if nb < na && na%nb == 0 {
		// Broadcasting: b is smaller
		for i := 0; i < na; i++ {
			cData[i] = aData[i] + bData[i%nb]
		}
	} else if na < nb && nb%na == 0 {
		// Broadcasting: a is smaller — repeat a along trailing dims
		for i := 0; i < nb; i++ {
			cData[i] = aData[i%na] + bData[i]
		}
	} else {
		return fmt.Errorf("add: incompatible sizes %d and %d", na, nb)
	}
	return nil
}

func (addOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpAdd, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// ReLU: y = max(0, x)  (in-place capable)
// ════════════════════════════════════════════════════════════════════════════

type reluOp struct{}

func (reluOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	for i, v := range src {
		if v < 0 {
			dst[i] = 0
		} else {
			dst[i] = v
		}
	}
	return nil
}

func (reluOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpReLU, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Sigmoid: y = 1 / (1 + exp(-x))
// ════════════════════════════════════════════════════════════════════════════

type sigmoidOp struct{}

func (sigmoidOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	for i, v := range src {
		dst[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
	}
	return nil
}

func (sigmoidOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpSigmoid, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Softmax: row-wise softmax
// ════════════════════════════════════════════════════════════════════════════

type softmaxOp struct{}

func (softmaxOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	shape := inputs[0].shape

	if len(shape) < 2 {
		// 1D softmax
		softmaxRow(src, dst)
		return nil
	}

	rows := 1
	for i := 0; i < len(shape)-1; i++ {
		rows *= shape[i]
	}
	cols := shape[len(shape)-1]

	for r := 0; r < rows; r++ {
		start := r * cols
		softmaxRow(src[start:start+cols], dst[start:start+cols])
	}
	return nil
}

// softmaxRow computes softmax over a single row (zero-alloc).
//
//mem:hot
//mem:nogc
func softmaxRow(src, dst []float32) {
	// Find max for numerical stability
	maxVal := src[0]
	for _, v := range src[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	// Compute exp and sum
	var sum float32
	for i, v := range src {
		e := float32(math.Exp(float64(v - maxVal)))
		dst[i] = e
		sum += e
	}
	// Normalize
	invSum := 1.0 / sum
	for i := range dst {
		dst[i] *= invSum
	}
}

func (softmaxOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpSoftmax, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Conv2D: im2col + matmul approach
// Input: [N,C,H,W]  Kernel: [OutC,InC,KH,KW]  Output: [N,OutC,OH,OW]
// ════════════════════════════════════════════════════════════════════════════

type conv2dOp struct {
	group                                int
	strideH, strideW                     int
	padTop, padLeft, padBottom, padRight int
	dilH, dilW                           int
	im2colBuf                            []float32 // scratch for im2col (heap, reused)
}

func (op *conv2dOp) SetAttrs(attrs []byte) error {
	op.group = 1
	op.strideH, op.strideW = 1, 1
	op.dilH, op.dilW = 1, 1
	if len(attrs) >= 18 {
		op.group = int(binary.LittleEndian.Uint16(attrs[0:2]))
		op.strideH = int(binary.LittleEndian.Uint16(attrs[2:4]))
		op.strideW = int(binary.LittleEndian.Uint16(attrs[4:6]))
		op.padTop = int(binary.LittleEndian.Uint16(attrs[6:8]))
		op.padLeft = int(binary.LittleEndian.Uint16(attrs[8:10]))
		op.padBottom = int(binary.LittleEndian.Uint16(attrs[10:12]))
		op.padRight = int(binary.LittleEndian.Uint16(attrs[12:14]))
		op.dilH = int(binary.LittleEndian.Uint16(attrs[14:16]))
		op.dilW = int(binary.LittleEndian.Uint16(attrs[16:18]))
	}
	if op.group < 1 {
		op.group = 1
	}
	return nil
}

func (op *conv2dOp) Execute(inputs, outputs []*Tensor) error {
	input := inputs[0]  // [N,C,H,W]
	kernel := inputs[1] // [OutC, InC/groups, KH, KW]
	output := outputs[0]

	inData := input.Float32s()
	kData := kernel.Float32s()
	outData := output.Float32s()

	n := input.shape[0]
	inC := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	outC := kernel.shape[0]
	grpInC := kernel.shape[1] // inC per group
	kH := kernel.shape[2]
	kW := kernel.shape[3]

	group := op.group
	if group < 1 {
		group = 1
	}
	sH := op.strideH
	if sH < 1 {
		sH = 1
	}
	sW := op.strideW
	if sW < 1 {
		sW = 1
	}
	pT, pL := op.padTop, op.padLeft
	pB, pR := op.padBottom, op.padRight
	dH := op.dilH
	if dH < 1 {
		dH = 1
	}
	dW := op.dilW
	if dW < 1 {
		dW = 1
	}

	effKH := (kH-1)*dH + 1
	effKW := (kW-1)*dW + 1
	oH := (h+pT+pB-effKH)/sH + 1
	oW := (w+pL+pR-effKW)/sW + 1
	outCPerGroup := outC / group

	// Dispatch to specialized fast paths
	if kH == 1 && kW == 1 && sH == 1 && sW == 1 && group == 1 {
		// ── Fast path: 1×1 pointwise conv = pure MatMul ──────────────
		// kernel [OutC, InC, 1, 1] → reshaped as [OutC, InC]
		// input  [N, InC, H, W]    → reshaped as [InC, H*W] per batch
		// output [N, OutC, H, W]   → reshaped as [OutC, H*W] per batch
		hw := h * w
		for batch := 0; batch < n; batch++ {
			inOff := batch * inC * hw
			outOff := batch * outC * hw
			matMulF32(kData, inData[inOff:inOff+inC*hw], outData[outOff:outOff+outC*hw],
				outC, inC, hw)
		}
	} else if group == outC && grpInC == 1 {
		// ── Fast path: depthwise convolution ─────────────────────────
		// Each output channel depends on exactly one input channel.
		for batch := 0; batch < n; batch++ {
			for ch := 0; ch < outC; ch++ {
				inBase := batch*inC*h*w + ch*h*w
				kBase := ch * kH * kW
				outBase := batch*outC*oH*oW + ch*oH*oW
				for oh := 0; oh < oH; oh++ {
					ihBase := oh*sH - pT
					for ow := 0; ow < oW; ow++ {
						iwBase := ow*sW - pL
						var sum float32
						for kh := 0; kh < kH; kh++ {
							ih := ihBase + kh*dH
							if ih < 0 || ih >= h {
								continue
							}
							inRow := inBase + ih*w
							kRow := kBase + kh*kW
							for kw := 0; kw < kW; kw++ {
								iw := iwBase + kw*dW
								if iw >= 0 && iw < w {
									sum += inData[inRow+iw] * kData[kRow+kw]
								}
							}
						}
						outData[outBase+oh*oW+ow] = sum
					}
				}
			}
		}
	} else if group == 1 {
		// ── im2col + matmul for regular convolutions ─────────────────
		// Rearrange input patches into columns, then multiply with
		// the kernel matrix using the optimized SIMD matmul.
		colSize := grpInC * kH * kW // rows of the column matrix
		spatial := oH * oW          // columns of the column matrix

		// Ensure scratch buffer is large enough (one-time heap alloc, reused)
		need := colSize * spatial
		if cap(op.im2colBuf) < need {
			op.im2colBuf = make([]float32, need)
		}
		col := op.im2colBuf[:need]

		for batch := 0; batch < n; batch++ {
			inOff := batch * inC * h * w
			outOff := batch * outC * oH * oW

			// Build im2col matrix: col[ic*kH*kW + kh*kW + kw][oh*oW + ow]
			idx := 0
			for ic := 0; ic < inC; ic++ {
				for kh := 0; kh < kH; kh++ {
					for kw := 0; kw < kW; kw++ {
						for oh := 0; oh < oH; oh++ {
							ih := oh*sH - pT + kh*dH
							for ow := 0; ow < oW; ow++ {
								iw := ow*sW - pL + kw*dW
								if ih >= 0 && ih < h && iw >= 0 && iw < w {
									col[idx] = inData[inOff+ic*h*w+ih*w+iw]
								} else {
									col[idx] = 0
								}
								idx++
							}
						}
					}
				}
			}

			// kernel is [OutC, InC*kH*kW] (already in the right layout)
			// col is    [InC*kH*kW, oH*oW]
			// output is [OutC, oH*oW]
			matMulF32(kData, col, outData[outOff:outOff+outC*spatial],
				outC, colSize, spatial)
		}
	} else {
		// ── General grouped convolution (fallback) ───────────────────
		for batch := 0; batch < n; batch++ {
			for g := 0; g < group; g++ {
				for oc := 0; oc < outCPerGroup; oc++ {
					absOC := g*outCPerGroup + oc
					for oh := 0; oh < oH; oh++ {
						for ow := 0; ow < oW; ow++ {
							var sum float32
							for ic := 0; ic < grpInC; ic++ {
								absIC := g*grpInC + ic
								for kh := 0; kh < kH; kh++ {
									for kw := 0; kw < kW; kw++ {
										ih := oh*sH - pT + kh*dH
										iw := ow*sW - pL + kw*dW
										if ih >= 0 && ih < h && iw >= 0 && iw < w {
											inIdx := batch*inC*h*w + absIC*h*w + ih*w + iw
											kIdx := absOC*grpInC*kH*kW + ic*kH*kW + kh*kW + kw
											sum += inData[inIdx] * kData[kIdx]
										}
									}
								}
							}
							outIdx := batch*outC*oH*oW + absOC*oH*oW + oh*oW + ow
							outData[outIdx] = sum
						}
					}
				}
			}
		}
	}

	// Add bias if present
	if len(inputs) > 2 {
		bias := inputs[2].Float32s()
		for batch := 0; batch < n; batch++ {
			for oc := 0; oc < outC; oc++ {
				bv := bias[oc]
				base := batch*outC*oH*oW + oc*oH*oW
				for i := 0; i < oH*oW; i++ {
					outData[base+i] += bv
				}
			}
		}
	}
	return nil
}

func (conv2dOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpConv2D, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// MaxPool2D: 2×2 kernel, stride 2
// ════════════════════════════════════════════════════════════════════════════

type maxPool2dOp struct{}

func (maxPool2dOp) Execute(inputs, outputs []*Tensor) error {
	in := inputs[0]
	out := outputs[0]
	inData := in.Float32s()
	outData := out.Float32s()

	n := in.shape[0]
	c := in.shape[1]
	h := in.shape[2]
	w := in.shape[3]
	oH := h / 2
	oW := w / 2

	for batch := 0; batch < n; batch++ {
		for ch := 0; ch < c; ch++ {
			for oh := 0; oh < oH; oh++ {
				for ow := 0; ow < oW; ow++ {
					maxVal := float32(-math.MaxFloat32)
					for kh := 0; kh < 2; kh++ {
						for kw := 0; kw < 2; kw++ {
							ih := oh*2 + kh
							iw := ow*2 + kw
							idx := batch*c*h*w + ch*h*w + ih*w + iw
							if inData[idx] > maxVal {
								maxVal = inData[idx]
							}
						}
					}
					outIdx := batch*c*oH*oW + ch*oH*oW + oh*oW + ow
					outData[outIdx] = maxVal
				}
			}
		}
	}
	return nil
}

func (maxPool2dOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpMaxPool2D, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// AvgPool2D: 2×2 kernel, stride 2
// ════════════════════════════════════════════════════════════════════════════

type avgPool2dOp struct{}

func (avgPool2dOp) Execute(inputs, outputs []*Tensor) error {
	in := inputs[0]
	out := outputs[0]
	inData := in.Float32s()
	outData := out.Float32s()

	n := in.shape[0]
	c := in.shape[1]
	h := in.shape[2]
	w := in.shape[3]
	oH := h / 2
	oW := w / 2

	for batch := 0; batch < n; batch++ {
		for ch := 0; ch < c; ch++ {
			for oh := 0; oh < oH; oh++ {
				for ow := 0; ow < oW; ow++ {
					var sum float32
					for kh := 0; kh < 2; kh++ {
						for kw := 0; kw < 2; kw++ {
							ih := oh*2 + kh
							iw := ow*2 + kw
							sum += inData[batch*c*h*w+ch*h*w+ih*w+iw]
						}
					}
					outData[batch*c*oH*oW+ch*oH*oW+oh*oW+ow] = sum / 4.0
				}
			}
		}
	}
	return nil
}

func (avgPool2dOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpAvgPool2D, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// BatchNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// inputs: [x, gamma, beta, mean, var]
// ════════════════════════════════════════════════════════════════════════════

type batchNormOp struct{}

func (batchNormOp) Execute(inputs, outputs []*Tensor) error {
	x := inputs[0]
	gamma := inputs[1].Float32s()
	beta := inputs[2].Float32s()
	runMean := inputs[3].Float32s()
	runVar := inputs[4].Float32s()

	xData := x.Float32s()
	yData := outputs[0].Float32s()

	const eps = 1e-5

	if len(x.shape) == 4 {
		// NCHW layout
		n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		spatial := h * w
		for batch := 0; batch < n; batch++ {
			for ch := 0; ch < c; ch++ {
				scale := gamma[ch] / float32(math.Sqrt(float64(runVar[ch]+eps)))
				shift := beta[ch] - runMean[ch]*scale
				base := batch*c*spatial + ch*spatial
				for s := 0; s < spatial; s++ {
					yData[base+s] = xData[base+s]*scale + shift
				}
			}
		}
	} else {
		// 2D: [batch, features]
		features := x.shape[len(x.shape)-1]
		rows := len(xData) / features
		for r := 0; r < rows; r++ {
			for f := 0; f < features; f++ {
				idx := r*features + f
				scale := gamma[f] / float32(math.Sqrt(float64(runVar[f]+eps)))
				yData[idx] = (xData[idx]-runMean[f])*scale + beta[f]
			}
		}
	}
	return nil
}

func (batchNormOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpBatchNorm, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Flatten: [N, C, H, W] → [N, C*H*W]
// ════════════════════════════════════════════════════════════════════════════

type flattenOp struct{}

func (flattenOp) Execute(inputs, outputs []*Tensor) error {
	// Flatten is a view operation. Since input and output share arena memory
	// (wired by engine), we just copy if they differ.
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	if &src[0] != &dst[0] {
		copy(dst, src)
	}
	return nil
}

func (flattenOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpFlatten, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Reshape: arbitrary reshape (view)
// ════════════════════════════════════════════════════════════════════════════

type reshapeOp struct{}

func (reshapeOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	if &src[0] != &dst[0] {
		copy(dst, src)
	}
	return nil
}

func (reshapeOp) OutputShape(in []Shape) ([]Shape, error) {
	// For OutputShape, we'd need attrs — this is handled by InferShapes
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// Quantize: float32 → int8 (symmetric)
// ════════════════════════════════════════════════════════════════════════════

type quantizeOp struct{}

func (quantizeOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Int8s()

	// Find abs max for scale
	var absMax float32
	for _, v := range src {
		av := v
		if av < 0 {
			av = -av
		}
		if av > absMax {
			absMax = av
		}
	}

	scale := absMax / 127.0
	if scale == 0 {
		scale = 1.0
	}
	invScale := 1.0 / scale

	for i, v := range src {
		q := int32(v * invScale)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		dst[i] = int8(q)
	}
	return nil
}

func (quantizeOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// Dequantize: int8 → float32
// ════════════════════════════════════════════════════════════════════════════

type dequantizeOp struct{}

func (dequantizeOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Int8s()
	dst := outputs[0].Float32s()

	// Default scale = 1/127 if no attrs; real scale would come from model metadata
	var scale float32 = 1.0 / 127.0
	for i, v := range src {
		dst[i] = float32(v) * scale
	}
	return nil
}

func (dequantizeOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// INT8 MatMul: C_i32 = A_i8 × B_i8 (quantized)
// ════════════════════════════════════════════════════════════════════════════

// MatMulInt8 performs an INT8 matrix multiplication with INT32 accumulation
// and float32 dequantization. Zero-alloc on pre-allocated tensors.
//
//mem:hot
//mem:nogc
func MatMulInt8(a, b *Tensor, c *Tensor, scaleA, scaleB float32) {
	aData := a.Int8s()
	bData := b.Int8s()
	cData := c.Int32s()

	m := a.shape[len(a.shape)-2]
	k := a.shape[len(a.shape)-1]
	n := b.shape[len(b.shape)-1]

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc int32
			for p := 0; p < k; p++ {
				acc += int32(aData[i*k+p]) * int32(bData[p*n+j])
			}
			cData[i*n+j] = acc
		}
	}
}

// unused import guard
var _ = unsafe.Pointer(nil)

// sqrt2OverPi is precomputed √(2/π) for GELU.
var sqrt2OverPi = float32(math.Sqrt(2.0 / math.Pi))

// ════════════════════════════════════════════════════════════════════════════
// GELU: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// ════════════════════════════════════════════════════════════════════════════

type geluOp struct{}

func (geluOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	for i, x := range src {
		x3 := x * x * x
		inner := sqrt2OverPi * (x + 0.044715*x3)
		dst[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
	}
	return nil
}

func (geluOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpGELU, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// inputs: [x, gamma, beta]
// ════════════════════════════════════════════════════════════════════════════

type layerNormOp struct{}

func (layerNormOp) Execute(inputs, outputs []*Tensor) error {
	x := inputs[0]
	gamma := inputs[1].Float32s()
	beta := inputs[2].Float32s()

	xData := x.Float32s()
	yData := outputs[0].Float32s()

	const eps = 1e-5

	// cols = last dim (hidden size), rows = everything else
	shape := x.shape
	cols := shape[len(shape)-1]
	rows := 1
	for i := 0; i < len(shape)-1; i++ {
		rows *= shape[i]
	}

	invCols := 1.0 / float32(cols)

	for r := 0; r < rows; r++ {
		base := r * cols
		row := xData[base : base+cols]

		// Compute mean
		var sum float32
		for _, v := range row {
			sum += v
		}
		mean := sum * invCols

		// Compute variance
		var varSum float32
		for _, v := range row {
			d := v - mean
			varSum += d * d
		}
		variance := varSum * invCols
		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))

		// Normalize + affine transform
		out := yData[base : base+cols]
		for c := 0; c < cols; c++ {
			out[c] = (row[c]-mean)*invStd*gamma[c] + beta[c]
		}
	}
	return nil
}

func (layerNormOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpLayerNorm, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Gather: embedding lookup — out[i] = weights[indices[i]]
// inputs: [weights [V, D], indices [S] (Int32)]
// ════════════════════════════════════════════════════════════════════════════

type gatherOp struct{}

func (gatherOp) Execute(inputs, outputs []*Tensor) error {
	weightData := inputs[0].Float32s()
	indices := inputs[1].Int32s()
	outData := outputs[0].Float32s()

	embedDim := inputs[0].shape[len(inputs[0].shape)-1]

	for i, idx := range indices {
		srcOff := int(idx) * embedDim
		dstOff := i * embedDim
		copy(outData[dstOff:dstOff+embedDim], weightData[srcOff:srcOff+embedDim])
	}
	return nil
}

func (gatherOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpGather, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// BatchedMatMul: C[b] = A[b] × B[b]    (float32, 3D)
// ════════════════════════════════════════════════════════════════════════════

type batchedMatMulOp struct{}

func (batchedMatMulOp) Execute(inputs, outputs []*Tensor) error {
	a, b, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	bData := b.Float32s()
	cData := c.Float32s()

	rank := len(a.shape)
	m := a.shape[rank-2]
	k := a.shape[rank-1]
	n := b.shape[len(b.shape)-1]

	// Compute total batch size (product of all dims except last 2)
	batch := 1
	for i := 0; i < rank-2; i++ {
		batch *= a.shape[i]
	}

	aStride := m * k
	bStride := k * n
	cStride := m * n

	for bi := 0; bi < batch; bi++ {
		matMulF32(
			aData[bi*aStride:(bi+1)*aStride],
			bData[bi*bStride:(bi+1)*bStride],
			cData[bi*cStride:(bi+1)*cStride],
			m, k, n,
		)
	}
	return nil
}

func (batchedMatMulOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpBatchedMatMul, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Mul: C = A * B  (element-wise, with broadcasting)
// ════════════════════════════════════════════════════════════════════════════

type mulOp struct{}

func (mulOp) Execute(inputs, outputs []*Tensor) error {
	a, b, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	bData := b.Float32s()
	cData := c.Float32s()

	na := len(aData)
	nb := len(bData)

	if nb == na {
		for i := 0; i < na; i++ {
			cData[i] = aData[i] * bData[i]
		}
	} else if nb == 1 {
		// Scalar broadcast
		scalar := bData[0]
		for i := 0; i < na; i++ {
			cData[i] = aData[i] * scalar
		}
	} else if na == 1 {
		// Scalar broadcast (a is scalar)
		scalar := aData[0]
		for i := 0; i < nb; i++ {
			cData[i] = scalar * bData[i]
		}
	} else if nb < na && na%nb == 0 {
		// b is smaller: broadcast b over a
		for i := 0; i < na; i++ {
			cData[i] = aData[i] * bData[i%nb]
		}
	} else if na < nb && nb%na == 0 {
		// Broadcasting: a is smaller — repeat a along trailing dims
		for i := 0; i < nb; i++ {
			cData[i] = aData[i%na] * bData[i]
		}
	} else {
		return fmt.Errorf("mul: incompatible sizes %d and %d", na, nb)
	}
	return nil
}

func (mulOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpMul, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Sub: C = A - B  (element-wise, with broadcasting)
// ════════════════════════════════════════════════════════════════════════════

type subOp struct{}

func (subOp) Execute(inputs, outputs []*Tensor) error {
	a, b, c := inputs[0], inputs[1], outputs[0]
	aData := a.Float32s()
	bData := b.Float32s()
	cData := c.Float32s()

	na := len(aData)
	nb := len(bData)

	if nb == na {
		for i := 0; i < na; i++ {
			cData[i] = aData[i] - bData[i]
		}
	} else if nb == 1 {
		scalar := bData[0]
		for i := 0; i < na; i++ {
			cData[i] = aData[i] - scalar
		}
	} else if na == 1 {
		scalar := aData[0]
		for i := 0; i < nb; i++ {
			cData[i] = scalar - bData[i]
		}
	} else if nb < na && na%nb == 0 {
		for i := 0; i < na; i++ {
			cData[i] = aData[i] - bData[i%nb]
		}
	} else if na < nb && nb%na == 0 {
		// Broadcasting: a is smaller — repeat a along trailing dims
		for i := 0; i < nb; i++ {
			cData[i] = aData[i%na] - bData[i]
		}
	} else {
		return fmt.Errorf("sub: incompatible sizes %d and %d", na, nb)
	}
	return nil
}

func (subOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpSub, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Transpose: permute dimensions
// Attrs: [ndims u16, perm0 u16, perm1 u16, ...]
// ════════════════════════════════════════════════════════════════════════════

type transposeOp struct{}

func (transposeOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0]
	dst := outputs[0]
	srcData := src.Float32s()
	dstData := dst.Float32s()

	shape := src.shape
	ndims := len(shape)

	// Derive permutation from output shape comparison
	// Convention: perm is encoded such that dst.shape[i] = src.shape[perm[i]]
	dstShape := dst.shape

	// Compute source strides in elements
	srcStrides := [8]int{} // max 8 dims, zero-alloc
	srcStrides[ndims-1] = 1
	for i := ndims - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * shape[i+1]
	}

	// Compute destination strides in elements
	dstStrides := [8]int{}
	dstStrides[ndims-1] = 1
	for i := ndims - 2; i >= 0; i-- {
		dstStrides[i] = dstStrides[i+1] * dstShape[i+1]
	}

	// Build perm from shapes: find which src dim maps to each dst dim
	perm := [8]int{}
	used := [8]bool{}
	for di := 0; di < ndims; di++ {
		for si := 0; si < ndims; si++ {
			if !used[si] && dstShape[di] == shape[si] {
				perm[di] = si
				used[si] = true
				break
			}
		}
	}

	total := len(srcData)
	coords := [8]int{}
	for idx := 0; idx < total; idx++ {
		// Compute dst coordinates from linear index
		remain := idx
		for d := 0; d < ndims; d++ {
			coords[d] = remain / dstStrides[d]
			remain %= dstStrides[d]
		}
		// Compute source linear index
		srcIdx := 0
		for d := 0; d < ndims; d++ {
			srcIdx += coords[d] * srcStrides[perm[d]]
		}
		dstData[idx] = srcData[srcIdx]
	}
	return nil
}

func (transposeOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpTranspose, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Tanh: y = tanh(x)  (element-wise)
// ════════════════════════════════════════════════════════════════════════════

type tanhOp struct{}

func (tanhOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	for i, v := range src {
		dst[i] = float32(math.Tanh(float64(v)))
	}
	return nil
}

func (tanhOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpTanh, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Slice: extract sub-range along one axis
// Attrs: [axis u16, start i32, end i32]
// ════════════════════════════════════════════════════════════════════════════

type sliceOp struct{}

func (sliceOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0]
	dst := outputs[0]
	srcData := src.Float32s()
	dstData := dst.Float32s()

	srcShape := src.shape
	dstShape := dst.shape
	ndims := len(srcShape)

	// Find the axis that differs in size
	axis := 0
	sliceStart := 0
	for d := 0; d < ndims; d++ {
		if dstShape[d] != srcShape[d] {
			axis = d
			break
		}
	}
	_ = sliceStart

	// Compute element counts
	outerSize := 1
	for d := 0; d < axis; d++ {
		outerSize *= srcShape[d]
	}
	innerSize := 1
	for d := axis + 1; d < ndims; d++ {
		innerSize *= srcShape[d]
	}

	sliceLen := dstShape[axis]
	srcAxisStride := srcShape[axis] * innerSize
	dstAxisStride := sliceLen * innerSize

	for o := 0; o < outerSize; o++ {
		srcBase := o * srcAxisStride
		dstBase := o * dstAxisStride
		copy(dstData[dstBase:dstBase+dstAxisStride], srcData[srcBase:srcBase+dstAxisStride])
	}
	return nil
}

func (sliceOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpSlice, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Where: out[i] = cond[i] != 0 ? a[i] : b[i]
// inputs: [cond, a, b]
// ════════════════════════════════════════════════════════════════════════════

type whereOp struct{}

func (whereOp) Execute(inputs, outputs []*Tensor) error {
	condData := inputs[0].Float32s()
	aData := inputs[1].Float32s()
	bData := inputs[2].Float32s()
	outData := outputs[0].Float32s()

	n := len(outData)
	nCond := len(condData)
	nA := len(aData)
	nB := len(bData)

	for i := 0; i < n; i++ {
		cond := condData[i%nCond]
		if cond != 0 {
			outData[i] = aData[i%nA]
		} else {
			outData[i] = bData[i%nB]
		}
	}
	return nil
}

func (whereOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpWhere, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Split: split tensor along axis into equal parts
// Attrs: [axis u16, numSplits u16]
// ════════════════════════════════════════════════════════════════════════════

type splitOp struct{}

func (splitOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0]
	srcData := src.Float32s()
	srcShape := src.shape
	ndims := len(srcShape)
	numSplits := len(outputs)

	// Determine axis from output shape comparison
	axis := 0
	for d := 0; d < ndims; d++ {
		if outputs[0].shape[d] != srcShape[d] {
			axis = d
			break
		}
	}

	splitSize := srcShape[axis] / numSplits

	outerSize := 1
	for d := 0; d < axis; d++ {
		outerSize *= srcShape[d]
	}
	innerSize := 1
	for d := axis + 1; d < ndims; d++ {
		innerSize *= srcShape[d]
	}

	srcAxisStride := srcShape[axis] * innerSize
	chunkBytes := splitSize * innerSize

	for s := 0; s < numSplits; s++ {
		dstData := outputs[s].Float32s()
		for o := 0; o < outerSize; o++ {
			srcOff := o*srcAxisStride + s*chunkBytes
			dstOff := o * chunkBytes
			copy(dstData[dstOff:dstOff+chunkBytes], srcData[srcOff:srcOff+chunkBytes])
		}
	}
	return nil
}

func (splitOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpSplit, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// Div: element-wise divide (supports broadcasting)
// ════════════════════════════════════════════════════════════════════════════

type divOp struct{}

func (divOp) Execute(inputs, outputs []*Tensor) error {
	a := inputs[0].Float32s()
	b := inputs[1].Float32s()
	dst := outputs[0].Float32s()

	if len(a) == len(b) {
		// Same shape — element-wise
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
	} else if len(b) == 1 {
		// Scalar broadcast
		invB := 1.0 / b[0]
		for i := range dst {
			dst[i] = a[i] * invB
		}
	} else {
		// General broadcast (match addOp pattern)
		for i := range dst {
			dst[i] = a[i] / b[i%len(b)]
		}
	}
	return nil
}

func (divOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// Pow: element-wise power (supports broadcasting)
// ════════════════════════════════════════════════════════════════════════════

type powOp struct{}

func (powOp) Execute(inputs, outputs []*Tensor) error {
	a := inputs[0].Float32s()
	b := inputs[1].Float32s()
	dst := outputs[0].Float32s()

	if len(b) == 1 {
		// Scalar exponent (most common: x^3 for GELU)
		exp := b[0]
		if exp == 2.0 {
			for i := range dst {
				dst[i] = a[i] * a[i]
			}
		} else if exp == 3.0 {
			for i := range dst {
				dst[i] = a[i] * a[i] * a[i]
			}
		} else {
			for i := range dst {
				dst[i] = float32(math.Pow(float64(a[i]), float64(exp)))
			}
		}
	} else if len(a) == len(b) {
		for i := range dst {
			dst[i] = float32(math.Pow(float64(a[i]), float64(b[i])))
		}
	} else {
		for i := range dst {
			dst[i] = float32(math.Pow(float64(a[i]), float64(b[i%len(b)])))
		}
	}
	return nil
}

func (powOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// IsNaN: element-wise NaN check → outputs 1.0 if NaN, 0.0 otherwise
// (output is float32 bitmask that Where can consume)
// ════════════════════════════════════════════════════════════════════════════

type isNaNOp struct{}

func (isNaNOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()
	for i, v := range src {
		if v != v { // NaN != NaN
			dst[i] = 1.0
		} else {
			dst[i] = 0.0
		}
	}
	return nil
}

func (isNaNOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// And: element-wise logical AND on float32 bitmasks
// (treats 0.0 as false, nonzero as true; outputs 1.0 / 0.0)
// ════════════════════════════════════════════════════════════════════════════

type andOp struct{}

func (andOp) Execute(inputs, outputs []*Tensor) error {
	a := inputs[0].Float32s()
	b := inputs[1].Float32s()
	dst := outputs[0].Float32s()

	na := len(a)
	nb := len(b)

	for i := range dst {
		if a[i%na] != 0 && b[i%nb] != 0 {
			dst[i] = 1.0
		} else {
			dst[i] = 0.0
		}
	}
	return nil
}

func (andOp) OutputShape(in []Shape) ([]Shape, error) {
	return []Shape{in[0]}, nil
}

// ════════════════════════════════════════════════════════════════════════════
// GlobalAvgPool2D: reduces spatial dims to 1×1
// Input: [N,C,H,W]  Output: [N,C,1,1]
// ════════════════════════════════════════════════════════════════════════════

type globalAvgPool2dOp struct{}

func (globalAvgPool2dOp) Execute(inputs, outputs []*Tensor) error {
	in := inputs[0]
	out := outputs[0]
	inData := in.Float32s()
	outData := out.Float32s()

	n := in.shape[0]
	c := in.shape[1]
	h := in.shape[2]
	w := in.shape[3]
	spatial := h * w
	invSpatial := 1.0 / float32(spatial)

	for batch := 0; batch < n; batch++ {
		for ch := 0; ch < c; ch++ {
			var sum float32
			base := batch*c*h*w + ch*h*w
			for s := 0; s < spatial; s++ {
				sum += inData[base+s]
			}
			outData[batch*c+ch] = sum * invSpatial
		}
	}
	return nil
}

func (globalAvgPool2dOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpGlobalAvgPool2D, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// HardSigmoid: y = clip(alpha*x + beta, 0, 1)
// ONNX defaults: alpha=0.2, beta=0.5
// ════════════════════════════════════════════════════════════════════════════

type hardSigmoidOp struct {
	alpha float32
	beta  float32
}

func (op *hardSigmoidOp) SetAttrs(attrs []byte) error {
	op.alpha = 0.2
	op.beta = 0.5
	if len(attrs) >= 8 {
		op.alpha = math.Float32frombits(binary.LittleEndian.Uint32(attrs[0:4]))
		op.beta = math.Float32frombits(binary.LittleEndian.Uint32(attrs[4:8]))
	}
	return nil
}

func (op *hardSigmoidOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()

	alpha := op.alpha
	beta := op.beta
	if alpha == 0 {
		alpha = 0.2
	}
	if beta == 0 {
		beta = 0.5
	}

	for i, v := range src {
		y := alpha*v + beta
		if y < 0 {
			y = 0
		} else if y > 1 {
			y = 1
		}
		dst[i] = y
	}
	return nil
}

func (op *hardSigmoidOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpHardSigmoid, in, nil)
}

// ════════════════════════════════════════════════════════════════════════════
// HardSwish: y = x * HardSigmoid(x) = x * clip(alpha*x + beta, 0, 1)
// Standard def: y = x * clip(x/6 + 0.5, 0, 1)
// ════════════════════════════════════════════════════════════════════════════

type hardSwishOp struct{}

func (hardSwishOp) Execute(inputs, outputs []*Tensor) error {
	src := inputs[0].Float32s()
	dst := outputs[0].Float32s()

	for i, v := range src {
		hs := v/6.0 + 0.5 // HardSigmoid with alpha=1/6, beta=0.5
		if hs < 0 {
			hs = 0
		} else if hs > 1 {
			hs = 1
		}
		dst[i] = v * hs
	}
	return nil
}

func (hardSwishOp) OutputShape(in []Shape) ([]Shape, error) {
	return inferOpOutputShapes(OpHardSwish, in, nil)
}
