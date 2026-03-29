// Package inference provides arena-backed AI model inference for MemPipe.
//
// The inference engine loads custom .mpmodel files, allocates all memory
// (weights + activations + I/O) in a single arena, and executes neural
// network graphs with zero allocations on the hot path.
package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

// Format constants for .mpmodel binary files.
const (
	// MagicBytes is the 4-byte magic number at the start of every .mpmodel file.
	MagicBytes = "MPMD"

	// FormatVersion is the current .mpmodel format version.
	FormatVersion uint16 = 1

	// HeaderSize is the fixed size of the file header in bytes.
	HeaderSize = 64

	// WeightAlignment is the byte alignment for the weights section.
	WeightAlignment = 64
)

// Header flags.
const (
	FlagQuantizedInt8 uint16 = 1 << iota
	FlagQuantizedFP16
	FlagHasCalibration
)

// ────────────────────────────────────────────────────────────────────────────
// File Header (64 bytes, little-endian)
// ────────────────────────────────────────────────────────────────────────────
//
//	Offset  Size  Field
//	  0       4   magic ("MPMD")
//	  4       2   version (uint16)
//	  6       2   flags (uint16)
//	  8       8   metadata_offset (uint64)
//	 16       8   metadata_size (uint64)
//	 24       8   graph_offset (uint64)
//	 32       8   graph_size (uint64)
//	 40       8   weights_offset (uint64)
//	 48       8   weights_size (uint64)
//	 56       8   total_size (uint64)
// ────────────────────────────────────────────────────────────────────────────

// fileHeader is the raw on-disk header structure.
type fileHeader struct {
	Magic          [4]byte
	Version        uint16
	Flags          uint16
	MetadataOffset uint64
	MetadataSize   uint64
	GraphOffset    uint64
	GraphSize      uint64
	WeightsOffset  uint64
	WeightsSize    uint64
	TotalSize      uint64
}

// ────────────────────────────────────────────────────────────────────────────
// Metadata
// ────────────────────────────────────────────────────────────────────────────

// Metadata describes the model's top-level properties.
type Metadata struct {
	Name          string             // model name (null-terminated in file)
	InputShapes   []Shape            // input tensor shapes
	OutputShapes  []Shape            // output tensor shapes
	QuantMethod   string             // "", "int8_symmetric", "int8_asymmetric", "fp16"
	QuantScale    float32            // global quantization scale (fallback)
	QuantZero     int32              // global quantization zero-point (fallback)
	PlatformHints string             // e.g. "wasm", "arm64", ""
	TensorScales  map[string]float32 // per-tensor quantization scales (overrides QuantScale)
	TensorZeros   map[string]int32   // per-tensor quantization zero-points (overrides QuantZero)
}

// Shape represents a tensor's dimensions.
type Shape struct {
	Dims []int
}

// NumElements returns the total number of elements in this shape.
func (s Shape) NumElements() int {
	if len(s.Dims) == 0 {
		return 0
	}
	n := 1
	for _, d := range s.Dims {
		n *= d
	}
	return n
}

// Equal returns true if two shapes have identical dimensions.
func (s Shape) Equal(other Shape) bool {
	if len(s.Dims) != len(other.Dims) {
		return false
	}
	for i, d := range s.Dims {
		if d != other.Dims[i] {
			return false
		}
	}
	return true
}

// ────────────────────────────────────────────────────────────────────────────
// Graph / Operator nodes
// ────────────────────────────────────────────────────────────────────────────

// OpType enumerates the supported neural-network operators.
type OpType uint16

const (
	OpMatMul OpType = iota
	OpAdd
	OpReLU
	OpSigmoid
	OpSoftmax
	OpConv2D
	OpMaxPool2D
	OpAvgPool2D
	OpBatchNorm
	OpFlatten
	OpReshape
	OpConcat
	OpDense // MatMul + BiasAdd fused
	OpQuantize
	OpDequantize
	OpGELU            // Transformer: GELU activation
	OpLayerNorm       // Transformer: Layer Normalization
	OpGather          // Transformer: Embedding lookup
	OpBatchedMatMul   // Transformer: 3D batched matmul for MHA
	OpMul             // Element-wise multiply
	OpSub             // Element-wise subtract
	OpTranspose       // Dim permutation
	OpSlice           // Sub-range extraction
	OpTanh            // Element-wise tanh
	OpWhere           // Conditional select
	OpSplit           // Split along axis
	OpDiv             // Element-wise divide
	OpPow             // Element-wise power
	OpIsNaN           // Element-wise NaN check
	OpAnd             // Element-wise logical AND
	OpGlobalAvgPool2D // Global average pooling → [N,C,1,1]
	OpHardSigmoid     // HardSigmoid activation
	OpHardSwish       // HardSwish activation
	OpRoPE            // Rotary Positional Embedding
)

// String returns the operator name.
func (o OpType) String() string {
	names := [...]string{
		"MatMul", "Add", "ReLU", "Sigmoid", "Softmax",
		"Conv2D", "MaxPool2D", "AvgPool2D", "BatchNorm",
		"Flatten", "Reshape", "Concat", "Dense",
		"Quantize", "Dequantize",
		"GELU", "LayerNorm", "Gather", "BatchedMatMul",
		"Mul", "Sub", "Transpose", "Slice",
		"Tanh", "Where", "Split", "Div",
		"Pow", "IsNaN", "And",
		"GlobalAvgPool2D", "HardSigmoid", "HardSwish",
		"RoPE",
	}
	if int(o) < len(names) {
		return names[o]
	}
	return fmt.Sprintf("Unknown(%d)", o)
}

// OpNode is one node in the computation graph.
type OpNode struct {
	Type          OpType // operator type
	InputIndices  []int  // indices into the tensor name list
	OutputIndices []int  // indices into the tensor name list
	Attrs         []byte // operator-specific attributes (encoded)
}

// ────────────────────────────────────────────────────────────────────────────
// Model
// ────────────────────────────────────────────────────────────────────────────

// Model is an in-memory representation of a loaded .mpmodel file.
type Model struct {
	Metadata     Metadata
	Graph        []OpNode
	TensorNames  []string         // interned tensor names referenced by graph indices
	TensorShapes map[string]Shape // known shapes for weight/intermediate tensors
	WeightsBlob  []byte           // raw weight bytes (64-byte aligned)
}

// WeightDType returns the DType used for weight tensors based on the model's
// quantization method. Returns Float32 if no quantization is specified.
func (m *Model) WeightDType() DType {
	switch m.Metadata.QuantMethod {
	case "int8_symmetric", "int8_asymmetric":
		return Int8
	case "fp16":
		return Float16
	default:
		return Float32
	}
}

// InputSize returns the total number of float32 input elements.
func (m *Model) InputSize() int {
	total := 0
	for _, s := range m.Metadata.InputShapes {
		total += s.NumElements()
	}
	return total
}

// OutputSize returns the total number of float32 output elements.
func (m *Model) OutputSize() int {
	total := 0
	for _, s := range m.Metadata.OutputShapes {
		total += s.NumElements()
	}
	return total
}

// WeightsSize returns the byte size of the weights blob.
func (m *Model) WeightsSize() int {
	return len(m.WeightsBlob)
}

// Validate checks the model for internal consistency.
func (m *Model) Validate() error {
	if len(m.Metadata.InputShapes) == 0 {
		return errors.New("model has no input shapes")
	}
	if len(m.Metadata.OutputShapes) == 0 {
		return errors.New("model has no output shapes")
	}
	if len(m.Graph) == 0 {
		return errors.New("model has no graph nodes")
	}
	for _, s := range m.Metadata.InputShapes {
		if s.NumElements() == 0 {
			return errors.New("input shape has zero elements")
		}
	}
	for _, s := range m.Metadata.OutputShapes {
		if s.NumElements() == 0 {
			return errors.New("output shape has zero elements")
		}
	}
	// Validate graph indices
	maxIdx := len(m.TensorNames)
	for i, node := range m.Graph {
		for _, idx := range node.InputIndices {
			if idx < 0 || idx >= maxIdx {
				return fmt.Errorf("node %d input index %d out of range [0,%d)", i, idx, maxIdx)
			}
		}
		for _, idx := range node.OutputIndices {
			if idx < 0 || idx >= maxIdx {
				return fmt.Errorf("node %d output index %d out of range [0,%d)", i, idx, maxIdx)
			}
		}
	}
	return nil
}

// ────────────────────────────────────────────────────────────────────────────
// Loaders
// ────────────────────────────────────────────────────────────────────────────

// LoadModel reads a .mpmodel file from disk and returns a Model.
// This is for native targets — not available on WASM/embedded (use LoadModelFromBytes).
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model file: %w", err)
	}
	return LoadModelFromBytes(data)
}

// LoadModelFromBytes parses a .mpmodel from a byte slice.
// Works everywhere: native, WASM, embedded.
func LoadModelFromBytes(data []byte) (*Model, error) {
	if len(data) < HeaderSize {
		return nil, fmt.Errorf("data too short for header: %d < %d", len(data), HeaderSize)
	}

	// ── Parse header ──────────────────────────────────────────────────
	var hdr fileHeader
	hdr.Magic = [4]byte(data[0:4])
	if string(hdr.Magic[:]) != MagicBytes {
		return nil, fmt.Errorf("invalid magic: %q (expected %q)", hdr.Magic[:], MagicBytes)
	}
	hdr.Version = binary.LittleEndian.Uint16(data[4:6])
	if hdr.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported version: %d (expected %d)", hdr.Version, FormatVersion)
	}
	hdr.Flags = binary.LittleEndian.Uint16(data[6:8])
	hdr.MetadataOffset = binary.LittleEndian.Uint64(data[8:16])
	hdr.MetadataSize = binary.LittleEndian.Uint64(data[16:24])
	hdr.GraphOffset = binary.LittleEndian.Uint64(data[24:32])
	hdr.GraphSize = binary.LittleEndian.Uint64(data[32:40])
	hdr.WeightsOffset = binary.LittleEndian.Uint64(data[40:48])
	hdr.WeightsSize = binary.LittleEndian.Uint64(data[48:56])
	hdr.TotalSize = binary.LittleEndian.Uint64(data[56:64])

	if uint64(len(data)) < hdr.TotalSize {
		return nil, fmt.Errorf("data truncated: %d < %d", len(data), hdr.TotalSize)
	}

	// ── Parse metadata ────────────────────────────────────────────────
	metaEnd := hdr.MetadataOffset + hdr.MetadataSize
	if metaEnd > uint64(len(data)) {
		return nil, errors.New("metadata section exceeds file bounds")
	}
	meta, err := decodeMetadata(data[hdr.MetadataOffset:metaEnd])
	if err != nil {
		return nil, fmt.Errorf("decode metadata: %w", err)
	}
	meta.applyFlags(hdr.Flags)

	// ── Parse graph ───────────────────────────────────────────────────
	graphEnd := hdr.GraphOffset + hdr.GraphSize
	if graphEnd > uint64(len(data)) {
		return nil, errors.New("graph section exceeds file bounds")
	}
	graph, tensorNames, tensorShapes, err := decodeGraph(data[hdr.GraphOffset:graphEnd])
	if err != nil {
		return nil, fmt.Errorf("decode graph: %w", err)
	}

	// ── Extract weights ──────────────────────────────────────────────
	weightsEnd := hdr.WeightsOffset + hdr.WeightsSize
	if weightsEnd > uint64(len(data)) {
		return nil, errors.New("weights section exceeds file bounds")
	}
	weights := make([]byte, hdr.WeightsSize)
	copy(weights, data[hdr.WeightsOffset:weightsEnd])

	model := &Model{
		Metadata:     meta,
		Graph:        graph,
		TensorNames:  tensorNames,
		TensorShapes: tensorShapes,
		WeightsBlob:  weights,
	}
	return model, nil
}

// ────────────────────────────────────────────────────────────────────────────
// Serialization  (write path — used by the Python converter & tests)
// ────────────────────────────────────────────────────────────────────────────

// SerializeModel encodes a Model to .mpmodel bytes.
func SerializeModel(m *Model) ([]byte, error) {
	metaBytes, err := encodeMetadata(&m.Metadata)
	if err != nil {
		return nil, fmt.Errorf("encode metadata: %w", err)
	}
	graphBytes, err := encodeGraph(m.Graph, m.TensorNames, m.TensorShapes)
	if err != nil {
		return nil, fmt.Errorf("encode graph: %w", err)
	}

	// Compute section offsets with alignment.
	metaOff := uint64(HeaderSize)
	graphOff := metaOff + uint64(len(metaBytes))
	weightsOff := align64(graphOff + uint64(len(graphBytes)))
	totalSize := weightsOff + uint64(len(m.WeightsBlob))

	flags := flagsFromMetadata(&m.Metadata)

	// Build output buffer.
	buf := make([]byte, totalSize)

	// Header
	copy(buf[0:4], MagicBytes)
	binary.LittleEndian.PutUint16(buf[4:6], FormatVersion)
	binary.LittleEndian.PutUint16(buf[6:8], flags)
	binary.LittleEndian.PutUint64(buf[8:16], metaOff)
	binary.LittleEndian.PutUint64(buf[16:24], uint64(len(metaBytes)))
	binary.LittleEndian.PutUint64(buf[24:32], graphOff)
	binary.LittleEndian.PutUint64(buf[32:40], uint64(len(graphBytes)))
	binary.LittleEndian.PutUint64(buf[40:48], weightsOff)
	binary.LittleEndian.PutUint64(buf[48:56], uint64(len(m.WeightsBlob)))
	binary.LittleEndian.PutUint64(buf[56:64], totalSize)

	// Sections
	copy(buf[metaOff:], metaBytes)
	copy(buf[graphOff:], graphBytes)
	copy(buf[weightsOff:], m.WeightsBlob)

	return buf, nil
}

// align64 rounds v up to the next multiple of 64.
func align64(v uint64) uint64 {
	return (v + 63) &^ 63
}

// ────────────────────────────────────────────────────────────────────────────
// Metadata codec  (compact binary, no JSON/msgpack dependency)
// ────────────────────────────────────────────────────────────────────────────

func encodeMetadata(md *Metadata) ([]byte, error) {
	// Simple TLV-style encoding:
	//   [nameLen u16][name bytes]
	//   [numInputs u16][ for each: ndims u16, dims... i32 ]
	//   [numOutputs u16][ for each: ndims u16, dims... i32 ]
	//   [quantMethodLen u16][quantMethod bytes]
	//   [quantScale f32][quantZero i32]
	//   [platformHintsLen u16][platformHints bytes]

	est := 2 + len(md.Name) + 2 + 2 + 2 + len(md.QuantMethod) + 4 + 4 + 2 + len(md.PlatformHints)
	for _, s := range md.InputShapes {
		est += 2 + 4*len(s.Dims)
	}
	for _, s := range md.OutputShapes {
		est += 2 + 4*len(s.Dims)
	}
	buf := make([]byte, 0, est)

	buf = appendU16(buf, uint16(len(md.Name)))
	buf = append(buf, md.Name...)

	buf = appendU16(buf, uint16(len(md.InputShapes)))
	for _, s := range md.InputShapes {
		buf = appendU16(buf, uint16(len(s.Dims)))
		for _, d := range s.Dims {
			buf = appendI32(buf, int32(d))
		}
	}
	buf = appendU16(buf, uint16(len(md.OutputShapes)))
	for _, s := range md.OutputShapes {
		buf = appendU16(buf, uint16(len(s.Dims)))
		for _, d := range s.Dims {
			buf = appendI32(buf, int32(d))
		}
	}

	buf = appendU16(buf, uint16(len(md.QuantMethod)))
	buf = append(buf, md.QuantMethod...)
	buf = appendF32(buf, md.QuantScale)
	buf = appendI32(buf, md.QuantZero)
	buf = appendU16(buf, uint16(len(md.PlatformHints)))
	buf = append(buf, md.PlatformHints...)

	// Per-tensor quantization parameters (v1.1 extension)
	buf = appendU16(buf, uint16(len(md.TensorScales)))
	for name, scale := range md.TensorScales {
		buf = appendU16(buf, uint16(len(name)))
		buf = append(buf, name...)
		buf = appendF32(buf, scale)
	}
	buf = appendU16(buf, uint16(len(md.TensorZeros)))
	for name, zero := range md.TensorZeros {
		buf = appendU16(buf, uint16(len(name)))
		buf = append(buf, name...)
		buf = appendI32(buf, zero)
	}

	return buf, nil
}

func decodeMetadata(data []byte) (Metadata, error) {
	var md Metadata
	off := 0

	nameLen, off, err := readU16(data, off)
	if err != nil {
		return md, err
	}
	if off+int(nameLen) > len(data) {
		return md, errors.New("metadata name truncated")
	}
	md.Name = string(data[off : off+int(nameLen)])
	off += int(nameLen)

	numInputs, off, err := readU16(data, off)
	if err != nil {
		return md, err
	}
	md.InputShapes = make([]Shape, numInputs)
	for i := range md.InputShapes {
		ndims, newOff, e := readU16(data, off)
		if e != nil {
			return md, e
		}
		off = newOff
		dims := make([]int, ndims)
		for j := range dims {
			v, newOff2, e2 := readI32(data, off)
			if e2 != nil {
				return md, e2
			}
			dims[j] = int(v)
			off = newOff2
		}
		md.InputShapes[i] = Shape{Dims: dims}
	}

	numOutputs, off, err := readU16(data, off)
	if err != nil {
		return md, err
	}
	md.OutputShapes = make([]Shape, numOutputs)
	for i := range md.OutputShapes {
		ndims, newOff, e := readU16(data, off)
		if e != nil {
			return md, e
		}
		off = newOff
		dims := make([]int, ndims)
		for j := range dims {
			v, newOff2, e2 := readI32(data, off)
			if e2 != nil {
				return md, e2
			}
			dims[j] = int(v)
			off = newOff2
		}
		md.OutputShapes[i] = Shape{Dims: dims}
	}

	qmLen, off, err := readU16(data, off)
	if err != nil {
		return md, err
	}
	if off+int(qmLen) > len(data) {
		return md, errors.New("metadata quantMethod truncated")
	}
	md.QuantMethod = string(data[off : off+int(qmLen)])
	off += int(qmLen)

	md.QuantScale, off, err = readF32(data, off)
	if err != nil {
		return md, err
	}
	md.QuantZero, off, err = readI32(data, off)
	if err != nil {
		return md, err
	}

	phLen, off, err := readU16(data, off)
	if err != nil {
		return md, err
	}
	if off+int(phLen) > len(data) {
		return md, errors.New("metadata platformHints truncated")
	}
	md.PlatformHints = string(data[off : off+int(phLen)])
	off += int(phLen)

	// Per-tensor quantization parameters (backward-compatible extension)
	if off < len(data) {
		numScales, newOff, e := readU16(data, off)
		if e == nil {
			off = newOff
			md.TensorScales = make(map[string]float32, numScales)
			for i := 0; i < int(numScales); i++ {
				nl, o, e2 := readU16(data, off)
				if e2 != nil {
					break
				}
				off = o
				if off+int(nl) > len(data) {
					break
				}
				name := string(data[off : off+int(nl)])
				off += int(nl)
				scale, o2, e3 := readF32(data, off)
				if e3 != nil {
					break
				}
				off = o2
				md.TensorScales[name] = scale
			}
		}
	}
	if off < len(data) {
		numZeros, newOff, e := readU16(data, off)
		if e == nil {
			off = newOff
			md.TensorZeros = make(map[string]int32, numZeros)
			for i := 0; i < int(numZeros); i++ {
				nl, o, e2 := readU16(data, off)
				if e2 != nil {
					break
				}
				off = o
				if off+int(nl) > len(data) {
					break
				}
				name := string(data[off : off+int(nl)])
				off += int(nl)
				zero, o2, e3 := readI32(data, off)
				if e3 != nil {
					break
				}
				off = o2
				md.TensorZeros[name] = zero
			}
		}
	}

	return md, nil
}

func (md *Metadata) applyFlags(flags uint16) {
	if flags&FlagQuantizedInt8 != 0 && md.QuantMethod == "" {
		md.QuantMethod = "int8_symmetric"
	}
	if flags&FlagQuantizedFP16 != 0 && md.QuantMethod == "" {
		md.QuantMethod = "fp16"
	}
}

func flagsFromMetadata(md *Metadata) uint16 {
	var f uint16
	switch md.QuantMethod {
	case "int8_symmetric", "int8_asymmetric":
		f |= FlagQuantizedInt8
	case "fp16":
		f |= FlagQuantizedFP16
	}
	return f
}

// ────────────────────────────────────────────────────────────────────────────
// Graph codec
// ────────────────────────────────────────────────────────────────────────────
//
// Format:
//   [numTensorNames u16][ for each: len u16, bytes... ]
//   [numNodes u16][ for each node:
//     opType u16,
//     numInputs u16, inputIndices... u16,
//     numOutputs u16, outputIndices... u16,
//     attrsLen u16, attrs bytes
//   ]

func encodeGraph(nodes []OpNode, tensorNames []string, tensorShapes map[string]Shape) ([]byte, error) {
	buf := make([]byte, 0, 256)

	buf = appendU16(buf, uint16(len(tensorNames)))
	for _, name := range tensorNames {
		buf = appendU16(buf, uint16(len(name)))
		buf = append(buf, name...)
	}

	buf = appendU16(buf, uint16(len(nodes)))
	for _, n := range nodes {
		buf = appendU16(buf, uint16(n.Type))
		buf = appendU16(buf, uint16(len(n.InputIndices)))
		for _, idx := range n.InputIndices {
			buf = appendU16(buf, uint16(idx))
		}
		buf = appendU16(buf, uint16(len(n.OutputIndices)))
		for _, idx := range n.OutputIndices {
			buf = appendU16(buf, uint16(idx))
		}
		buf = appendU16(buf, uint16(len(n.Attrs)))
		buf = append(buf, n.Attrs...)
	}

	// Encode tensor shapes (name -> dims)
	buf = appendU16(buf, uint16(len(tensorShapes)))
	for name, s := range tensorShapes {
		buf = appendU16(buf, uint16(len(name)))
		buf = append(buf, name...)
		buf = appendU16(buf, uint16(len(s.Dims)))
		for _, d := range s.Dims {
			buf = appendI32(buf, int32(d))
		}
	}

	return buf, nil
}

func decodeGraph(data []byte) ([]OpNode, []string, map[string]Shape, error) {
	off := 0

	numNames, off2, err := readU16(data, off)
	if err != nil {
		return nil, nil, nil, err
	}
	off = off2
	names := make([]string, numNames)
	for i := range names {
		nLen, o, e := readU16(data, off)
		if e != nil {
			return nil, nil, nil, e
		}
		off = o
		if off+int(nLen) > len(data) {
			return nil, nil, nil, errors.New("tensor name truncated")
		}
		names[i] = string(data[off : off+int(nLen)])
		off += int(nLen)
	}

	numNodes, off2, err := readU16(data, off)
	if err != nil {
		return nil, nil, nil, err
	}
	off = off2
	nodes := make([]OpNode, numNodes)
	for i := range nodes {
		opType, o, e := readU16(data, off)
		if e != nil {
			return nil, nil, nil, e
		}
		off = o
		nodes[i].Type = OpType(opType)

		numIn, o2, e2 := readU16(data, off)
		if e2 != nil {
			return nil, nil, nil, e2
		}
		off = o2
		nodes[i].InputIndices = make([]int, numIn)
		for j := range nodes[i].InputIndices {
			idx, o3, e3 := readU16(data, off)
			if e3 != nil {
				return nil, nil, nil, e3
			}
			off = o3
			nodes[i].InputIndices[j] = int(idx)
		}

		numOut, o4, e4 := readU16(data, off)
		if e4 != nil {
			return nil, nil, nil, e4
		}
		off = o4
		nodes[i].OutputIndices = make([]int, numOut)
		for j := range nodes[i].OutputIndices {
			idx, o5, e5 := readU16(data, off)
			if e5 != nil {
				return nil, nil, nil, e5
			}
			off = o5
			nodes[i].OutputIndices[j] = int(idx)
		}

		attrLen, o6, e6 := readU16(data, off)
		if e6 != nil {
			return nil, nil, nil, e6
		}
		off = o6
		if off+int(attrLen) > len(data) {
			return nil, nil, nil, errors.New("node attrs truncated")
		}
		nodes[i].Attrs = make([]byte, attrLen)
		copy(nodes[i].Attrs, data[off:off+int(attrLen)])
		off += int(attrLen)
	}

	// Decode tensor shapes (may be absent in older files)
	tensorShapes := make(map[string]Shape)
	if off < len(data) {
		numShapes, o7, e7 := readU16(data, off)
		if e7 == nil {
			off = o7
			for i := 0; i < int(numShapes); i++ {
				nLen, o, e := readU16(data, off)
				if e != nil {
					return nil, nil, nil, e
				}
				off = o
				if off+int(nLen) > len(data) {
					return nil, nil, nil, errors.New("tensor shape name truncated")
				}
				name := string(data[off : off+int(nLen)])
				off += int(nLen)

				ndims, o2, e2 := readU16(data, off)
				if e2 != nil {
					return nil, nil, nil, e2
				}
				off = o2
				dims := make([]int, ndims)
				for j := range dims {
					d, o3, e3 := readI32(data, off)
					if e3 != nil {
						return nil, nil, nil, e3
					}
					off = o3
					dims[j] = int(d)
				}
				tensorShapes[name] = Shape{Dims: dims}
			}
		}
	}

	return nodes, names, tensorShapes, nil
}

// ────────────────────────────────────────────────────────────────────────────
// Binary helpers  (little-endian, no allocation on read)
// ────────────────────────────────────────────────────────────────────────────

func appendU16(buf []byte, v uint16) []byte {
	return append(buf, byte(v), byte(v>>8))
}

func appendI32(buf []byte, v int32) []byte {
	return append(buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

func appendF32(buf []byte, v float32) []byte {
	bits := math.Float32bits(v)
	return append(buf, byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24))
}

func readU16(data []byte, off int) (uint16, int, error) {
	if off+2 > len(data) {
		return 0, off, errors.New("unexpected end of data reading u16")
	}
	v := binary.LittleEndian.Uint16(data[off:])
	return v, off + 2, nil
}

func readI32(data []byte, off int) (int32, int, error) {
	if off+4 > len(data) {
		return 0, off, errors.New("unexpected end of data reading i32")
	}
	v := int32(binary.LittleEndian.Uint32(data[off:]))
	return v, off + 4, nil
}

func readF32(data []byte, off int) (float32, int, error) {
	if off+4 > len(data) {
		return 0, off, errors.New("unexpected end of data reading f32")
	}
	bits := binary.LittleEndian.Uint32(data[off:])
	return math.Float32frombits(bits), off + 4, nil
}
