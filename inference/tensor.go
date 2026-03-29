package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// DType represents the data type of tensor elements.
type DType uint8

const (
	Float32 DType = iota
	Float16
	Int8
	Uint8
	Int32
)

// Size returns the byte size of one element of this dtype.
func (d DType) Size() int {
	switch d {
	case Float32:
		return 4
	case Float16:
		return 2
	case Int8, Uint8:
		return 1
	case Int32:
		return 4
	default:
		return 0
	}
}

// String returns the dtype name.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case Int8:
		return "int8"
	case Uint8:
		return "uint8"
	case Int32:
		return "int32"
	default:
		return "unknown"
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Tensor — a view into arena memory
// ────────────────────────────────────────────────────────────────────────────

// Tensor is a multi-dimensional view into a contiguous block of arena memory.
// Tensors never own memory — they are always views into an InferenceArena.
// All hot-path methods are zero-allocation.
type Tensor struct {
	data       unsafe.Pointer // pointer to first element in arena
	shape      []int          // dimensions (e.g. [batch, rows, cols])
	strides    []int          // byte strides per dimension
	dtype      DType          // element data type
	name       string         // tensor name (for debugging)
	size       int            // total byte size of data
	scratch    unsafe.Pointer // pre-allocated Float32 scratch for dequantization
	quantScale float32        // per-tensor quantization scale (0 = use default)
	quantZero  int32          // per-tensor quantization zero-point
}

// Name returns the tensor's name.
func (t *Tensor) Name() string { return t.name }

// Shape returns the tensor's dimensions.
func (t *Tensor) Shape() []int { return t.shape }

// SetShape updates the tensor's shape and strides in-place without
// reallocating memory. The new shape must require no more bytes than the
// tensor's original allocation (t.size). This enables dynamic sequence
// lengths: the arena is allocated for the maximum shape once, and callers
// shrink the shape before each inference pass so operators only process
// the active elements.
//
// Returns an error if the new shape would exceed the allocated byte budget.
// Zero-alloc: reuses the existing shape/strides slices when lengths match.
func (t *Tensor) SetShape(newShape []int) error {
	newElems := 1
	for _, d := range newShape {
		if d <= 0 {
			return fmt.Errorf("invalid dimension %d in shape for tensor %q", d, t.name)
		}
		newElems *= d
	}
	need := newElems * t.dtype.Size()
	if need > t.size {
		return fmt.Errorf("SetShape: new shape requires %d bytes but tensor %q has %d allocated",
			need, t.name, t.size)
	}
	// Reuse existing slices if lengths match (zero-alloc hot path).
	if len(t.shape) == len(newShape) {
		copy(t.shape, newShape)
	} else {
		t.shape = append(t.shape[:0], newShape...)
	}
	newStrides := computeStridesInto(t.strides, t.shape, t.dtype)
	t.strides = newStrides
	return nil
}

// Strides returns the byte strides per dimension.
func (t *Tensor) Strides() []int { return t.strides }

// DType returns the tensor's element data type.
func (t *Tensor) DType() DType { return t.dtype }

// DataPtr returns the raw pointer to the tensor's data in arena memory.
//
//mem:hot
func (t *Tensor) DataPtr() unsafe.Pointer { return t.data }

// ByteSize returns the total byte size of the tensor's data.
func (t *Tensor) ByteSize() int { return t.size }

// NumElements returns the total number of elements.
func (t *Tensor) NumElements() int {
	if len(t.shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range t.shape {
		n *= d
	}
	return n
}

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int { return len(t.shape) }

// ────────────────────────────────────────────────────────────────────────────
// Element access  (hot path, zero-alloc)
// ────────────────────────────────────────────────────────────────────────────

// AtF32 reads a float32 element at the given indices.
// Panics on out-of-bounds for hot-path performance.
//
//mem:hot
//mem:nogc
func (t *Tensor) AtF32(indices ...int) float32 {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	return *(*float32)(ptr)
}

// SetF32 writes a float32 element at the given indices.
//
//mem:hot
//mem:nogc
func (t *Tensor) SetF32(value float32, indices ...int) {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	*(*float32)(ptr) = value
}

// AtInt8 reads an int8 element.
//
//mem:hot
//mem:nogc
func (t *Tensor) AtInt8(indices ...int) int8 {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	return *(*int8)(ptr)
}

// SetInt8 writes an int8 element.
//
//mem:hot
//mem:nogc
func (t *Tensor) SetInt8(value int8, indices ...int) {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	*(*int8)(ptr) = value
}

// AtInt32 reads an int32 element.
//
//mem:hot
//mem:nogc
func (t *Tensor) AtInt32(indices ...int) int32 {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	return *(*int32)(ptr)
}

// SetInt32 writes an int32 element.
//
//mem:hot
//mem:nogc
func (t *Tensor) SetInt32(value int32, indices ...int) {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	*(*int32)(ptr) = value
}

// linearOffset computes the byte offset from indices using strides.
//
//mem:hot
//mem:nogc
func (t *Tensor) linearOffset(indices []int) int {
	off := 0
	for i, idx := range indices {
		off += idx * t.strides[i]
	}
	return off
}

// ────────────────────────────────────────────────────────────────────────────
// Float32 slice access  (bulk read/write, zero-copy)
// ────────────────────────────────────────────────────────────────────────────

// Float32s returns the tensor's data as a float32 slice.
// The returned slice is a direct view into arena memory — do NOT retain
// it beyond the engine's lifetime.
//
//mem:hot
func (t *Tensor) Float32s() []float32 {
	n := t.NumElements()
	return unsafe.Slice((*float32)(t.data), n)
}

// EnsureFloat32 returns the tensor's data as float32, performing on-the-fly
// dequantization for non-Float32 dtypes using the pre-allocated scratch buffer.
// For Float32 tensors this is zero-cost (returns Float32s directly).
// For Int8/FP16 tensors, the scratch buffer must have been allocated at
// compile time via SetScratch.
//
//mem:hot
func (t *Tensor) EnsureFloat32() []float32 {
	if t.dtype == Float32 {
		return t.Float32s()
	}
	if t.scratch == nil {
		// Fallback: should not happen in a properly compiled engine.
		return t.Float32s()
	}
	n := t.NumElements()
	return unsafe.Slice((*float32)(t.scratch), n)
}

// PopulateScratchFloat32 dequantizes the tensor data into the scratch buffer.
// Called once at compile time for weight tensors. Zero-alloc on the hot path
// since the scratch is reused across inference calls.
func (t *Tensor) PopulateScratchFloat32() {
	if t.dtype == Float32 || t.scratch == nil {
		return
	}
	n := t.NumElements()
	dst := unsafe.Slice((*float32)(t.scratch), n)
	switch t.dtype {
	case Int8:
		src := t.Int8s()
		scale := t.quantScale
		zero := t.quantZero
		if scale == 0 {
			scale = 1.0 / 127.0
		}
		for i, v := range src {
			dst[i] = float32(int32(v)-zero) * scale
		}
	case Uint8:
		src := unsafe.Slice((*uint8)(t.data), n)
		scale := t.quantScale
		zero := t.quantZero
		if scale == 0 {
			scale = 1.0 / 255.0
		}
		for i, v := range src {
			dst[i] = float32(int32(v)-zero) * scale
		}
	case Float16:
		src := t.Float16s()
		for i, v := range src {
			dst[i] = F16BitsToF32(v)
		}
	}
}

// SetScratch installs a pre-allocated scratch buffer for dequantization.
func (t *Tensor) SetScratch(ptr unsafe.Pointer) {
	t.scratch = ptr
}

// SetQuantParams sets per-tensor quantization scale and zero-point.
func (t *Tensor) SetQuantParams(scale float32, zero int32) {
	t.quantScale = scale
	t.quantZero = zero
}

// QuantScale returns the per-tensor quantization scale.
func (t *Tensor) QuantScale() float32 { return t.quantScale }

// QuantZero returns the per-tensor quantization zero-point.
func (t *Tensor) QuantZero() int32 { return t.quantZero }

// Int8s returns the tensor's data as an int8 slice (view into arena).
//
//mem:hot
func (t *Tensor) Int8s() []int8 {
	n := t.NumElements()
	return unsafe.Slice((*int8)(t.data), n)
}

// Int32s returns the tensor's data as an int32 slice (view into arena).
//
//mem:hot
func (t *Tensor) Int32s() []int32 {
	n := t.NumElements()
	return unsafe.Slice((*int32)(t.data), n)
}

// Float16s returns the tensor's data as a uint16 slice of raw IEEE 754
// half-precision bits (view into arena). Go has no native float16 type,
// so raw bits are exposed. Use F16BitsToF32 / F32ToF16Bits for conversion.
//
//mem:hot
func (t *Tensor) Float16s() []uint16 {
	n := t.NumElements()
	return unsafe.Slice((*uint16)(t.data), n)
}

// AtF16 reads a raw IEEE 754 half-precision value (uint16) at the given indices.
//
//mem:hot
//mem:nogc
func (t *Tensor) AtF16(indices ...int) uint16 {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	return *(*uint16)(ptr)
}

// SetF16 writes a raw IEEE 754 half-precision value (uint16) at the given indices.
//
//mem:hot
//mem:nogc
func (t *Tensor) SetF16(value uint16, indices ...int) {
	ptr := unsafe.Add(t.data, t.linearOffset(indices))
	*(*uint16)(ptr) = value
}

// Bytes returns the tensor's raw bytes (view into arena).
//
//mem:hot
func (t *Tensor) Bytes() []byte {
	return unsafe.Slice((*byte)(t.data), t.size)
}

// CopyFrom copies data from a byte slice into this tensor.
// The slice length must match the tensor's byte size.
func (t *Tensor) CopyFrom(src []byte) error {
	if len(src) != t.size {
		return fmt.Errorf("size mismatch: src=%d, tensor=%d", len(src), t.size)
	}
	dst := unsafe.Slice((*byte)(t.data), t.size)
	copy(dst, src)
	return nil
}

// CopyTo copies this tensor's data into a byte slice.
func (t *Tensor) CopyTo(dst []byte) error {
	if len(dst) < t.size {
		return fmt.Errorf("dst too small: %d < %d", len(dst), t.size)
	}
	src := unsafe.Slice((*byte)(t.data), t.size)
	copy(dst, src)
	return nil
}

// Zero fills the tensor with zeros.
//
//mem:hot
func (t *Tensor) Zero() {
	b := unsafe.Slice((*byte)(t.data), t.size)
	for i := range b {
		b[i] = 0
	}
}

// ────────────────────────────────────────────────────────────────────────────
// View operations (no copy, same arena memory)
// ────────────────────────────────────────────────────────────────────────────

// Reshape returns a new tensor view with different dimensions.
// The total number of elements must remain the same.
func (t *Tensor) Reshape(newShape ...int) (*Tensor, error) {
	newTotal := 1
	for _, d := range newShape {
		newTotal *= d
	}
	if newTotal != t.NumElements() {
		return nil, fmt.Errorf("reshape: element count mismatch: %d vs %d", newTotal, t.NumElements())
	}
	strides := computeStrides(newShape, t.dtype)
	return &Tensor{
		data:    t.data,
		shape:   newShape,
		strides: strides,
		dtype:   t.dtype,
		name:    t.name,
		size:    t.size,
	}, nil
}

// Slice returns a view into a sub-range along the first dimension.
func (t *Tensor) Slice(start, end int) (*Tensor, error) {
	if len(t.shape) == 0 {
		return nil, errors.New("cannot slice scalar tensor")
	}
	if start < 0 || end > t.shape[0] || start >= end {
		return nil, fmt.Errorf("invalid slice range [%d:%d] for dim size %d", start, end, t.shape[0])
	}
	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	newShape[0] = end - start

	offset := start * t.strides[0]
	newSize := (end - start) * t.strides[0]

	return &Tensor{
		data:    unsafe.Add(t.data, offset),
		shape:   newShape,
		strides: t.strides, // strides unchanged for slice
		dtype:   t.dtype,
		name:    t.name,
		size:    newSize,
	}, nil
}

// ────────────────────────────────────────────────────────────────────────────
// Inference Arena — single allocation for all tensors
// ────────────────────────────────────────────────────────────────────────────

// InferenceArena is a single contiguous memory block that holds all
// model weights, intermediate activations, and I/O buffers.
type InferenceArena struct {
	memory []byte // single allocation
	offset int    // current allocation cursor
	size   int    // total size
}

// NewInferenceArena allocates a single memory block of the given size.
//
//mem:allow(single_alloc) — THE one allocation
func NewInferenceArena(size int) *InferenceArena {
	// Align to 64 bytes
	size = int(align64(uint64(size)))
	return &InferenceArena{
		memory: make([]byte, size),
		size:   size,
	}
}

// AllocTensor reserves space in the arena for a tensor and returns it.
// This is a build-time operation — not called on the hot path.
func (a *InferenceArena) AllocTensor(name string, shape []int, dtype DType) (*Tensor, error) {
	numElems := 1
	for _, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("invalid dimension %d in shape for tensor %q", d, name)
		}
		numElems *= d
	}
	byteSize := numElems * dtype.Size()

	// Align offset to 64 bytes
	a.offset = int(align64(uint64(a.offset)))

	if a.offset+byteSize > a.size {
		return nil, fmt.Errorf("arena out of memory: need %d bytes at offset %d, arena size %d (tensor %q)",
			byteSize, a.offset, a.size, name)
	}

	ptr := unsafe.Pointer(&a.memory[a.offset])
	strides := computeStrides(shape, dtype)

	t := &Tensor{
		data:    ptr,
		shape:   shape,
		strides: strides,
		dtype:   dtype,
		name:    name,
		size:    byteSize,
	}

	a.offset += byteSize
	return t, nil
}

// AllocRaw reserves a raw block of bytes in the arena (64-byte aligned)
// and returns an unsafe.Pointer to its start. Used for scratch buffers
// that don't need a full Tensor wrapper (e.g. SIMD packing buffers).
func (a *InferenceArena) AllocRaw(byteSize int) (unsafe.Pointer, error) {
	a.offset = int(align64(uint64(a.offset)))
	if a.offset+byteSize > a.size {
		return nil, fmt.Errorf("arena out of memory: need %d bytes at offset %d, arena size %d",
			byteSize, a.offset, a.size)
	}
	ptr := unsafe.Pointer(&a.memory[a.offset])
	a.offset += byteSize
	return ptr, nil
}

// LoadWeights copies raw weight bytes into the arena at the current offset.
// Returns the base pointer of the loaded weights.
func (a *InferenceArena) LoadWeights(blob []byte) (unsafe.Pointer, error) {
	a.offset = int(align64(uint64(a.offset)))
	if a.offset+len(blob) > a.size {
		return nil, fmt.Errorf("arena too small for weights: need %d, available %d",
			len(blob), a.size-a.offset)
	}
	base := unsafe.Pointer(&a.memory[a.offset])
	copy(a.memory[a.offset:], blob)
	a.offset += len(blob)
	return base, nil
}

// UsedBytes returns the number of bytes allocated so far.
func (a *InferenceArena) UsedBytes() int { return a.offset }

// TotalBytes returns the total arena capacity.
func (a *InferenceArena) TotalBytes() int { return a.size }

// Reset resets the arena cursor to zero (reuse memory without re-allocating).
// WARNING: all previously allocated tensors become invalid.
func (a *InferenceArena) Reset() {
	a.offset = 0
}

// Zero clears all arena memory.
func (a *InferenceArena) Zero() {
	for i := range a.memory {
		a.memory[i] = 0
	}
	a.offset = 0
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

// computeStrides returns row-major byte strides for the given shape and dtype.
func computeStrides(shape []int, dtype DType) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	return computeStridesInto(strides, shape, dtype)
}

// computeStridesInto fills dst with the byte-strides for shape.
// If dst has sufficient capacity, no allocation occurs.
func computeStridesInto(dst []int, shape []int, dtype DType) []int {
	n := len(shape)
	if cap(dst) >= n {
		dst = dst[:n]
	} else {
		dst = make([]int, n)
	}
	stride := dtype.Size()
	for i := n - 1; i >= 0; i-- {
		dst[i] = stride
		stride *= shape[i]
	}
	return dst
}

// ────────────────────────────────────────────────────────────────────────────
// Shape inference — compute intermediate tensor shapes from graph
// ────────────────────────────────────────────────────────────────────────────

// InferShapes walks the graph nodes and computes output tensor shapes
// given the input shapes. Returns a map of tensor name → shape.
func InferShapes(graph []OpNode, tensorNames []string, inputShapes map[string]Shape) (map[string]Shape, error) {
	shapes := make(map[string]Shape, len(tensorNames))
	// Seed with known input shapes
	for name, s := range inputShapes {
		shapes[name] = s
	}

	for i, node := range graph {
		// Gather input shapes
		inShapes := make([]Shape, len(node.InputIndices))
		for j, idx := range node.InputIndices {
			name := tensorNames[idx]
			s, ok := shapes[name]
			if !ok {
				return nil, fmt.Errorf("node %d (%s): input %q shape unknown", i, node.Type, name)
			}
			inShapes[j] = s
		}

		// Compute output shape based on operator
		outShapes, err := inferOpOutputShapes(node.Type, inShapes, node.Attrs)
		if err != nil {
			return nil, fmt.Errorf("node %d (%s): %w", i, node.Type, err)
		}

		if len(outShapes) != len(node.OutputIndices) {
			return nil, fmt.Errorf("node %d (%s): expected %d outputs, got %d",
				i, node.Type, len(node.OutputIndices), len(outShapes))
		}

		for j, idx := range node.OutputIndices {
			shapes[tensorNames[idx]] = outShapes[j]
		}
	}

	return shapes, nil
}

// inferOpOutputShapes computes output shapes for a single operator.
func inferOpOutputShapes(op OpType, inputs []Shape, attrs []byte) ([]Shape, error) {
	switch op {
	case OpMatMul, OpDense:
		if len(inputs) < 2 {
			return nil, errors.New("MatMul requires at least 2 inputs")
		}
		a, b := inputs[0], inputs[1]
		if len(a.Dims) < 2 || len(b.Dims) < 2 {
			return nil, errors.New("MatMul inputs must be at least 2D")
		}
		m := a.Dims[len(a.Dims)-2]
		n := b.Dims[len(b.Dims)-1]
		outDims := make([]int, len(a.Dims))
		copy(outDims, a.Dims)
		outDims[len(outDims)-2] = m
		outDims[len(outDims)-1] = n
		return []Shape{{Dims: outDims}}, nil

	case OpAdd:
		if len(inputs) < 2 {
			return nil, errors.New("Add requires at least 2 inputs")
		}
		// Broadcasting — output shape is the larger input's shape
		if inputs[1].NumElements() > inputs[0].NumElements() {
			return []Shape{inputs[1]}, nil
		}
		return []Shape{inputs[0]}, nil

	case OpReLU, OpSigmoid:
		if len(inputs) < 1 {
			return nil, errors.New("activation requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpSoftmax:
		if len(inputs) < 1 {
			return nil, errors.New("Softmax requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpConv2D:
		if len(inputs) < 2 {
			return nil, errors.New("Conv2D requires at least 2 inputs")
		}
		// input [N,C,H,W], kernel [OutC,InC/groups,KH,KW]
		// attrs: [group u16, strideH u16, strideW u16, padTop u16, padLeft u16, padBottom u16, padRight u16, dilH u16, dilW u16]
		in := inputs[0]
		kernel := inputs[1]
		if len(in.Dims) != 4 || len(kernel.Dims) != 4 {
			return nil, errors.New("Conv2D expects 4D input and kernel")
		}
		strideH, strideW := 1, 1
		padTop, padLeft, padBottom, padRight := 0, 0, 0, 0
		dilH, dilW := 1, 1
		if len(attrs) >= 18 {
			// group at offset 0 (not needed for shape)
			strideH = int(binary.LittleEndian.Uint16(attrs[2:4]))
			strideW = int(binary.LittleEndian.Uint16(attrs[4:6]))
			padTop = int(binary.LittleEndian.Uint16(attrs[6:8]))
			padLeft = int(binary.LittleEndian.Uint16(attrs[8:10]))
			padBottom = int(binary.LittleEndian.Uint16(attrs[10:12]))
			padRight = int(binary.LittleEndian.Uint16(attrs[12:14]))
			dilH = int(binary.LittleEndian.Uint16(attrs[14:16]))
			dilW = int(binary.LittleEndian.Uint16(attrs[16:18]))
		}
		effKH := (kernel.Dims[2]-1)*dilH + 1
		effKW := (kernel.Dims[3]-1)*dilW + 1
		outH := (in.Dims[2]+padTop+padBottom-effKH)/strideH + 1
		outW := (in.Dims[3]+padLeft+padRight-effKW)/strideW + 1
		return []Shape{{Dims: []int{in.Dims[0], kernel.Dims[0], outH, outW}}}, nil

	case OpMaxPool2D, OpAvgPool2D:
		if len(inputs) < 1 {
			return nil, errors.New("Pool requires 1 input")
		}
		// Default 2×2 pool with stride 2
		in := inputs[0]
		if len(in.Dims) != 4 {
			return nil, errors.New("Pool expects 4D input")
		}
		return []Shape{{Dims: []int{in.Dims[0], in.Dims[1], in.Dims[2] / 2, in.Dims[3] / 2}}}, nil

	case OpBatchNorm:
		if len(inputs) < 1 {
			return nil, errors.New("BatchNorm requires at least 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpFlatten:
		if len(inputs) < 1 {
			return nil, errors.New("Flatten requires 1 input")
		}
		in := inputs[0]
		flat := 1
		for i := 1; i < len(in.Dims); i++ {
			flat *= in.Dims[i]
		}
		return []Shape{{Dims: []int{in.Dims[0], flat}}}, nil

	case OpReshape:
		if len(inputs) < 1 {
			return nil, errors.New("Reshape requires 1 input")
		}
		// Target shape in attrs: [ndims u16, dim0 i32, dim1 i32, ...]
		if len(attrs) < 2 {
			return nil, errors.New("Reshape requires attrs with target shape")
		}
		ndims := int(binary.LittleEndian.Uint16(attrs[0:2]))
		if len(attrs) < 2+ndims*4 {
			return nil, errors.New("Reshape attrs truncated")
		}
		dims := make([]int, ndims)
		for i := range dims {
			dims[i] = int(int32(binary.LittleEndian.Uint32(attrs[2+i*4:])))
		}

		// Support -1 dimension (infer from input element count, like ONNX).
		// Also handle dynamic reshaping: if element counts don't match and
		// there's no explicit -1, find the dimension that changed (e.g.
		// seq_len) and adjust it proportionally. This enables ReshapeInputs
		// to resize tensors without patching every Reshape node's attrs.
		inputElems := inputs[0].NumElements()
		neg1Idx := -1
		known := 1
		for i, d := range dims {
			if d == -1 {
				neg1Idx = i
			} else if d == 0 {
				// 0 means "copy from input" (ONNX convention)
				if i < len(inputs[0].Dims) {
					dims[i] = inputs[0].Dims[i]
				}
				known *= dims[i]
			} else {
				known *= d
			}
		}
		if neg1Idx >= 0 && known > 0 {
			dims[neg1Idx] = inputElems / known
		} else {
			// No explicit -1. Check if element counts match.
			attrElems := 1
			for _, d := range dims {
				attrElems *= d
			}
			if attrElems != inputElems && known > 0 {
				// Attrs were baked for a different size (e.g. max seq_len).
				// Find the dimension that, when adjusted, makes total equal
				// inputElems. For each candidate dim, compute the product of
				// ALL other dims; if inputElems is evenly divisible by that
				// product, that dim gets adjusted.
				// Example: [1,512,12,32] with input [1,8,384] (3072 elems):
				//   i=1: other prod = 1*12*32=384, 3072/384=8 ✓ → [1,8,12,32]
				fixed := false
				for i := 0; i < len(dims) && !fixed; i++ {
					prod := 1
					for j, d := range dims {
						if j != i {
							prod *= d
						}
					}
					if prod > 0 && inputElems%prod == 0 {
						newVal := inputElems / prod
						if newVal != dims[i] {
							dims[i] = newVal
							fixed = true
						}
					}
				}
			}
		}

		return []Shape{{Dims: dims}}, nil

	case OpConcat:
		if len(inputs) < 2 {
			return nil, errors.New("Concat requires at least 2 inputs")
		}
		// Concatenate along axis 1 by default
		outDims := make([]int, len(inputs[0].Dims))
		copy(outDims, inputs[0].Dims)
		axis := 1
		if len(attrs) >= 2 {
			axis = int(binary.LittleEndian.Uint16(attrs[0:2]))
		}
		for i := 1; i < len(inputs); i++ {
			outDims[axis] += inputs[i].Dims[axis]
		}
		return []Shape{{Dims: outDims}}, nil

	case OpQuantize, OpDequantize:
		if len(inputs) < 1 {
			return nil, errors.New("Quantize/Dequantize requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpGELU, OpTanh:
		if len(inputs) < 1 {
			return nil, errors.New("activation requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpLayerNorm:
		if len(inputs) < 1 {
			return nil, errors.New("LayerNorm requires at least 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpMul, OpSub:
		if len(inputs) < 2 {
			return nil, errors.New("element-wise op requires 2 inputs")
		}
		// Broadcasting — output shape is the larger input's shape
		if inputs[1].NumElements() > inputs[0].NumElements() {
			return []Shape{inputs[1]}, nil
		}
		return []Shape{inputs[0]}, nil

	case OpGather:
		if len(inputs) < 2 {
			return nil, errors.New("Gather requires 2 inputs (weights, indices)")
		}
		// weights [V, D], indices [...] → output = indices_shape + [D]
		embedDim := inputs[0].Dims[len(inputs[0].Dims)-1]
		outDims := make([]int, len(inputs[1].Dims)+1)
		copy(outDims, inputs[1].Dims)
		outDims[len(outDims)-1] = embedDim
		return []Shape{{Dims: outDims}}, nil

	case OpBatchedMatMul:
		if len(inputs) < 2 {
			return nil, errors.New("BatchedMatMul requires 2 inputs")
		}
		a, b := inputs[0], inputs[1]
		if len(a.Dims) < 2 || len(b.Dims) < 2 {
			return nil, errors.New("BatchedMatMul inputs must be at least 2D")
		}
		// Last 2 dims are the matrix: [..., M, K] x [..., K, N] → [..., M, N]
		m := a.Dims[len(a.Dims)-2]
		n := b.Dims[len(b.Dims)-1]
		// Batch dims = all dims except last 2
		outDims := make([]int, len(a.Dims))
		copy(outDims, a.Dims[:len(a.Dims)-2])
		outDims[len(outDims)-2] = m
		outDims[len(outDims)-1] = n
		return []Shape{{Dims: outDims}}, nil

	case OpTranspose:
		if len(inputs) < 1 {
			return nil, errors.New("Transpose requires 1 input")
		}
		// Permutation from attrs: [ndims u16, perm0 u16, ...]
		inDims := inputs[0].Dims
		ndims := len(inDims)
		outDims := make([]int, ndims)
		if len(attrs) >= 2+ndims*2 {
			// Read perm from attrs
			for i := 0; i < ndims; i++ {
				p := int(binary.LittleEndian.Uint16(attrs[2+i*2:]))
				outDims[i] = inDims[p]
			}
		} else {
			// Default: reverse dims
			for i := 0; i < ndims; i++ {
				outDims[i] = inDims[ndims-1-i]
			}
		}
		return []Shape{{Dims: outDims}}, nil

	case OpSlice:
		if len(inputs) < 1 {
			return nil, errors.New("Slice requires 1 input")
		}
		inDims := inputs[0].Dims
		outDims := make([]int, len(inDims))
		copy(outDims, inDims)
		if len(attrs) >= 10 {
			axis := int(binary.LittleEndian.Uint16(attrs[0:2]))
			start := int(int32(binary.LittleEndian.Uint32(attrs[2:6])))
			end := int(int32(binary.LittleEndian.Uint32(attrs[6:10])))
			outDims[axis] = end - start
		}
		return []Shape{{Dims: outDims}}, nil

	case OpWhere:
		if len(inputs) < 3 {
			return nil, errors.New("Where requires 3 inputs (cond, a, b)")
		}
		// Output shape is the broadcast of all three inputs — pick the one with the most elements
		best := inputs[0]
		bestN := best.NumElements()
		for _, s := range inputs[1:] {
			if n := s.NumElements(); n > bestN {
				best = s
				bestN = n
			}
		}
		return []Shape{best}, nil

	case OpDiv:
		if len(inputs) < 2 {
			return nil, errors.New("Div requires 2 inputs")
		}
		return []Shape{inputs[0]}, nil

	case OpPow:
		if len(inputs) < 2 {
			return nil, errors.New("Pow requires 2 inputs")
		}
		return []Shape{inputs[0]}, nil

	case OpIsNaN:
		if len(inputs) < 1 {
			return nil, errors.New("IsNaN requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpAnd:
		if len(inputs) < 2 {
			return nil, errors.New("And requires 2 inputs")
		}
		// Broadcast: pick the larger shape
		if inputs[1].NumElements() > inputs[0].NumElements() {
			return []Shape{inputs[1]}, nil
		}
		return []Shape{inputs[0]}, nil

	case OpGlobalAvgPool2D:
		if len(inputs) < 1 {
			return nil, errors.New("GlobalAvgPool2D requires 1 input")
		}
		in := inputs[0]
		if len(in.Dims) != 4 {
			return nil, errors.New("GlobalAvgPool2D expects 4D input [N,C,H,W]")
		}
		return []Shape{{Dims: []int{in.Dims[0], in.Dims[1], 1, 1}}}, nil

	case OpHardSigmoid, OpHardSwish:
		if len(inputs) < 1 {
			return nil, errors.New("activation requires 1 input")
		}
		return []Shape{inputs[0]}, nil

	case OpSplit:
		if len(inputs) < 1 {
			return nil, errors.New("Split requires 1 input")
		}
		inDims := inputs[0].Dims
		axis := 0
		numSplits := 2
		if len(attrs) >= 4 {
			axis = int(binary.LittleEndian.Uint16(attrs[0:2]))
			numSplits = int(binary.LittleEndian.Uint16(attrs[2:4]))
		}
		splitDim := inDims[axis] / numSplits
		results := make([]Shape, numSplits)
		for i := 0; i < numSplits; i++ {
			dims := make([]int, len(inDims))
			copy(dims, inDims)
			dims[axis] = splitDim
			results[i] = Shape{Dims: dims}
		}
		return results, nil

	default:
		return nil, fmt.Errorf("unsupported op for shape inference: %s", op)
	}
}

// unused import guard
var _ = math.Float32frombits
