package inference

import (
	"errors"
	"fmt"
	"unsafe"
)

// ────────────────────────────────────────────────────────────────────────────
// Engine — zero-alloc inference runtime
// ────────────────────────────────────────────────────────────────────────────

// compiledOp is a pre-resolved operator with its input/output tensor pointers.
type compiledOp struct {
	op      Operator
	inputs  []*Tensor
	outputs []*Tensor
}

// Engine loads a Model, allocates all memory (weights + activations + I/O)
// in a single InferenceArena, and executes the graph with zero allocations
// on the hot path.
type Engine struct {
	model           *Model
	arena           *InferenceArena
	tensors         map[string]*Tensor // all named tensors
	opOrder         []compiledOp       // topologically sorted op list
	inputs          []*Tensor          // model input tensors (ordered)
	outputs         []*Tensor          // model output tensors (ordered)
	weightNames     map[string]bool    // pre-computed set of weight tensor names
	overrides       *EngineOverrides   // per-engine operator overrides (nil = use global)
	inferShapeOpts  *InferShapeOptions // optional reshape hints etc. (nil = defaults)
}

// EngineOption configures Engine construction.
type EngineOption func(*engineConfig)

type engineConfig struct {
	extraArenaBytes  int
	inferShapeOpts   *InferShapeOptions
}

// WithExtraArena adds extra bytes to the inference arena.
func WithExtraArena(bytes int) EngineOption {
	return func(c *engineConfig) { c.extraArenaBytes = bytes }
}

// WithInferShapeOptions passes options into graph shape inference (e.g. Reshape
// shape hints from the converter). Used during compile and ReshapeInputs.
func WithInferShapeOptions(opt *InferShapeOptions) EngineOption {
	return func(c *engineConfig) { c.inferShapeOpts = opt }
}

// NewEngine creates an inference engine from a model.
// It allocates all memory up front and compiles the execution graph.
func NewEngine(model *Model, opts ...EngineOption) (*Engine, error) {
	return newEngine(model, nil, opts...)
}

// NewEngineWithOverrides creates an inference engine with per-engine operator
// overrides. This avoids mutating the global operator registry. Overrides
// take precedence over the global registry for the lifetime of this engine.
func NewEngineWithOverrides(model *Model, overrides *EngineOverrides, opts ...EngineOption) (*Engine, error) {
	return newEngine(model, overrides, opts...)
}

func newEngine(model *Model, overrides *EngineOverrides, opts ...EngineOption) (*Engine, error) {
	if model == nil {
		return nil, errors.New("model is nil")
	}
	if err := model.Validate(); err != nil {
		return nil, fmt.Errorf("invalid model: %w", err)
	}

	cfg := engineConfig{}
	for _, o := range opts {
		o(&cfg)
	}

	e := &Engine{
		model:          model,
		tensors:        make(map[string]*Tensor, len(model.TensorNames)),
		overrides:      overrides,
		inferShapeOpts: cfg.inferShapeOpts,
	}

	if err := e.compile(cfg); err != nil {
		return nil, fmt.Errorf("compile: %w", err)
	}

	return e, nil
}

// compile sets up the arena, loads weights, allocates intermediate tensors,
// and resolves operators.
func (e *Engine) compile(cfg engineConfig) error {
	model := e.model

	// 1. Infer all tensor shapes
	inputShapes := make(map[string]Shape, len(model.Metadata.InputShapes)+len(model.TensorShapes))
	// Convention: first N tensor names are inputs, in order
	for i, s := range model.Metadata.InputShapes {
		if i >= len(model.TensorNames) {
			return errors.New("more input shapes than tensor names")
		}
		inputShapes[model.TensorNames[i]] = s
	}
	// Seed weight/parameter tensor shapes from model's TensorShapes map.
	for name, s := range model.TensorShapes {
		if _, exists := inputShapes[name]; !exists {
			inputShapes[name] = s
		}
	}

	allShapes, err := InferShapes(model.Graph, model.TensorNames, inputShapes, cfg.inferShapeOpts)
	if err != nil {
		return fmt.Errorf("shape inference: %w", err)
	}

	// 2. Identify which tensors are weights packed in the blob.
	//    Convention: _pack_weights iterates tensor_names in order and packs
	//    every tensor that is an initializer (weight / constant).  Graph
	//    inputs (first N tensor names) are never packed.
	//    Weight dtype is derived from the model's quantization metadata.
	numInputs := len(model.Metadata.InputShapes)
	weightDType := model.WeightDType()
	isWeight := make(map[string]bool)
	{
		cursor := 0
		blobLen := len(model.WeightsBlob)
		for i, name := range model.TensorNames {
			if i < numInputs {
				continue // graph inputs are not in the blob
			}
			s, ok := allShapes[name]
			if !ok {
				continue
			}
			byteSize := s.NumElements() * weightDType.Size()
			if cursor+byteSize <= blobLen {
				isWeight[name] = true
				cursor += byteSize
			}
		}
	}

	// 3. Compute total arena size using int64 to avoid overflow for large models.
	//    = weights (64B aligned) + all non-weight tensors (64B aligned each)
	//    + scratch buffers for non-Float32 weight tensors (for EnsureFloat32)
	arenaSize64 := int64(align64(uint64(len(model.WeightsBlob))))

	// Store the weight set on the engine for ReshapeInputs.
	e.weightNames = isWeight

	for _, name := range model.TensorNames {
		if isWeight[name] {
			continue // already counted in the blob
		}
		s, ok := allShapes[name]
		if !ok {
			continue
		}
		byteSize := int64(s.NumElements()) * int64(Float32.Size())
		arenaSize64 += int64(align64(uint64(byteSize)))
	}

	// Reserve scratch space for non-Float32 weight dequantization
	if weightDType != Float32 {
		for name := range isWeight {
			s, ok := allShapes[name]
			if !ok {
				continue
			}
			scratchSize := int64(s.NumElements()) * int64(Float32.Size())
			arenaSize64 += int64(align64(uint64(scratchSize)))
		}
	}

	arenaSize64 += int64(cfg.extraArenaBytes)
	arenaSize := int(arenaSize64)

	// 4. Allocate single arena
	e.arena = NewInferenceArena(arenaSize)

	// 5. Load weights into arena (copies blob into arena[0..blobSize])
	weightsBase, err := e.arena.LoadWeights(model.WeightsBlob)
	if err != nil {
		return fmt.Errorf("load weights: %w", err)
	}

	// 6. Create Tensor objects.
	//    Weight tensors → point into the loaded weights region using native dtype.
	//    Non-weight tensors → allocate fresh arena memory after the blob.
	{
		cursor := uintptr(0) // offset within the weights region
		for _, name := range model.TensorNames {
			s, ok := allShapes[name]
			if !ok {
				continue
			}
			if isWeight[name] {
				// Map directly into the loaded weights region
				byteSize := s.NumElements() * weightDType.Size()
				ptr := unsafe.Pointer(uintptr(weightsBase) + cursor)
				strides := computeStrides(s.Dims, weightDType)
				t := &Tensor{
					data:    ptr,
					shape:   s.Dims,
					strides: strides,
					dtype:   weightDType,
					name:    name,
					size:    byteSize,
				}
				e.tensors[name] = t
				cursor += uintptr(byteSize)
			} else {
				// Allocate fresh memory in the arena (after the blob)
				t, err := e.arena.AllocTensor(name, s.Dims, Float32)
				if err != nil {
					return fmt.Errorf("alloc tensor %q: %w", name, err)
				}
				e.tensors[name] = t
			}
		}
	}

	// 6a. Set per-tensor quantization params and allocate scratch buffers
	//     for non-Float32 weight tensors so EnsureFloat32() works on the hot path.
	for name, t := range e.tensors {
		if t.dtype == Float32 {
			continue
		}
		// Set quantization parameters: per-tensor first, global fallback
		if model.Metadata.TensorScales != nil {
			if s, ok := model.Metadata.TensorScales[name]; ok {
				t.quantScale = s
			}
		}
		if t.quantScale == 0 {
			t.quantScale = model.Metadata.QuantScale
		}
		if model.Metadata.TensorZeros != nil {
			if z, ok := model.Metadata.TensorZeros[name]; ok {
				t.quantZero = z
			}
		}
		if t.quantZero == 0 {
			t.quantZero = model.Metadata.QuantZero
		}
		// Allocate scratch buffer and pre-populate with dequantized data
		scratchSize := t.NumElements() * Float32.Size()
		scratch, err := e.arena.AllocRaw(scratchSize)
		if err != nil {
			return fmt.Errorf("alloc scratch for %q: %w", name, err)
		}
		t.SetScratch(scratch)
		t.PopulateScratchFloat32()
	}

	// 6. Record input/output tensors
	e.inputs = make([]*Tensor, len(model.Metadata.InputShapes))
	for i := range model.Metadata.InputShapes {
		name := model.TensorNames[i]
		t, ok := e.tensors[name]
		if !ok {
			return fmt.Errorf("input tensor %q not allocated", name)
		}
		e.inputs[i] = t
	}

	// Outputs: last M tensor names
	numOut := len(model.Metadata.OutputShapes)
	e.outputs = make([]*Tensor, numOut)
	outStart := len(model.TensorNames) - numOut
	for i := 0; i < numOut; i++ {
		name := model.TensorNames[outStart+i]
		t, ok := e.tensors[name]
		if !ok {
			return fmt.Errorf("output tensor %q not allocated", name)
		}
		e.outputs[i] = t
	}

	// 7. Compile operators in graph order (already topological)
	e.opOrder = make([]compiledOp, len(model.Graph))
	for i, node := range model.Graph {
		op, err := getOperatorWithOverrides(node.Type, e.overrides)
		if err != nil {
			return fmt.Errorf("node %d: %w", i, err)
		}

		ins := make([]*Tensor, len(node.InputIndices))
		inNames := make([]string, len(node.InputIndices))
		inDTypes := make([]DType, len(node.InputIndices))
		for j, idx := range node.InputIndices {
			name := model.TensorNames[idx]
			t, ok := e.tensors[name]
			if !ok {
				return fmt.Errorf("node %d input tensor %q not found", i, name)
			}
			ins[j] = t
			inNames[j] = name
			inDTypes[j] = t.dtype
		}

		outs := make([]*Tensor, len(node.OutputIndices))
		outNames := make([]string, len(node.OutputIndices))
		outDTypes := make([]DType, len(node.OutputIndices))
		for j, idx := range node.OutputIndices {
			name := model.TensorNames[idx]
			t, ok := e.tensors[name]
			if !ok {
				return fmt.Errorf("node %d output tensor %q not found", i, name)
			}
			outs[j] = t
			outNames[j] = name
			outDTypes[j] = t.dtype
		}

		e.opOrder[i] = compiledOp{op: op, inputs: ins, outputs: outs}

		// If the operator accepts encoded attributes, pass them now.
		if attr, ok := op.(Attributable); ok && len(node.Attrs) > 0 {
			if err := attr.SetAttrs(node.Attrs); err != nil {
				return fmt.Errorf("node %d set attrs: %w", i, err)
			}
		}

		// DType negotiation: if the operator implements DTypeAware,
		// provide dtype and quantization metadata so it can select
		// optimized execution paths (e.g. SIMD INT8 matmul).
		if dtAware, ok := op.(DTypeAware); ok {
			info := &DTypeInfo{
				InputDTypes:  inDTypes,
				OutputDTypes: outDTypes,
				InputNames:   inNames,
				OutputNames:  outNames,
				TensorScales: model.Metadata.TensorScales,
				TensorZeros:  model.Metadata.TensorZeros,
				GlobalScale:  model.Metadata.QuantScale,
				GlobalZero:   model.Metadata.QuantZero,
			}
			if err := dtAware.SetDTypeInfo(info); err != nil {
				return fmt.Errorf("node %d set dtype info: %w", i, err)
			}
		}

		// If the operator supports one-time initialization (e.g. GPU
		// pipeline setup, scratch buffer allocation), call Init now.
		if init, ok := op.(Initializable); ok {
			if err := init.Init(e.arena); err != nil {
				return fmt.Errorf("node %d init: %w", i, err)
			}
		}
	}

	return nil
}

// ────────────────────────────────────────────────────────────────────────────
// Inference execution
// ────────────────────────────────────────────────────────────────────────────

// Infer runs the model on the given input bytes and returns output bytes.
// The hot loop is zero-allocation — input/output copies use pre-allocated arena memory.
//
//mem:hot
func (e *Engine) Infer(input []byte) ([]byte, error) {
	// Copy input into arena input tensor
	expectedSize := 0
	for _, t := range e.inputs {
		expectedSize += t.ByteSize()
	}
	if len(input) != expectedSize {
		return nil, fmt.Errorf("input size mismatch: got %d, expected %d", len(input), expectedSize)
	}

	off := 0
	for _, t := range e.inputs {
		if err := t.CopyFrom(input[off : off+t.ByteSize()]); err != nil {
			return nil, err
		}
		off += t.ByteSize()
	}

	// Execute compiled ops — ZERO ALLOC HOT PATH
	for _, cop := range e.opOrder {
		if err := cop.op.Execute(cop.inputs, cop.outputs); err != nil {
			return nil, fmt.Errorf("op execute: %w", err)
		}
	}

	// Copy output from arena
	outSize := 0
	for _, t := range e.outputs {
		outSize += t.ByteSize()
	}
	output := make([]byte, outSize)
	off = 0
	for _, t := range e.outputs {
		if err := t.CopyTo(output[off : off+t.ByteSize()]); err != nil {
			return nil, err
		}
		off += t.ByteSize()
	}
	return output, nil
}

// InferTensor runs inference with direct tensor access (zero-copy I/O).
// The caller must have written input data into the input tensors beforehand.
// Returns the output tensors (views into arena memory).
//
//mem:hot
//mem:nogc
func (e *Engine) InferTensor() ([]*Tensor, error) {
	for _, cop := range e.opOrder {
		if err := cop.op.Execute(cop.inputs, cop.outputs); err != nil {
			return nil, fmt.Errorf("op execute: %w", err)
		}
	}
	return e.outputs, nil
}

// InferBatch runs inference on multiple inputs.
// Each input/output is a separate byte slice.
func (e *Engine) InferBatch(inputs [][]byte) ([][]byte, error) {
	results := make([][]byte, len(inputs))
	for i, input := range inputs {
		out, err := e.Infer(input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = out
	}
	return results, nil
}

// ────────────────────────────────────────────────────────────────────────────
// Accessors
// ────────────────────────────────────────────────────────────────────────────

// InputTensors returns the model's input tensors (arena-backed).
func (e *Engine) InputTensors() []*Tensor { return e.inputs }

// OutputTensors returns the model's output tensors (arena-backed).
func (e *Engine) OutputTensors() []*Tensor { return e.outputs }

// Tensor returns a named tensor from the engine.
func (e *Engine) Tensor(name string) (*Tensor, bool) {
	t, ok := e.tensors[name]
	return t, ok
}

// Model returns the loaded model.
func (e *Engine) Model() *Model { return e.model }

// TensorNames returns the full list of tensor names (same order as the model).
func (e *Engine) TensorNames() []string { return e.model.TensorNames }

// IsWeight reports whether the named tensor is a weight (stored in WeightsBlob).
func (e *Engine) IsWeight(name string) bool { return e.weightNames[name] }

// ArenaUsed returns the number of arena bytes used.
func (e *Engine) ArenaUsed() int { return e.arena.UsedBytes() }

// ArenaTotal returns the total arena capacity.
func (e *Engine) ArenaTotal() int { return e.arena.TotalBytes() }

// ReshapeInputs updates input (and optional non-input) tensor shapes and
// propagates new shapes through the entire graph. All tensors remain in
// their original arena memory — only the shape/strides metadata is updated.
//
// This enables dynamic sequence lengths: allocate once for the maximum length
// (e.g. 8192 for long-context models like nomic-embed-text-v1.5), then call
// ReshapeInputs with actual length before each inference pass so operators
// only process active elements.
//
// inputShapes maps tensor name → new Shape. Typically these are the model's
// named inputs, but you may also include weight/constant tensor names
// (e.g. position embeddings) that have a seq_len dimension.
//
// Zero-alloc on the hot path when the number of dimensions doesn't change
// (which is the common case for variable-length sequences).
func (e *Engine) ReshapeInputs(inputShapes map[string]Shape) error {
	model := e.model

	// Build the full seed map: known shapes for weight tensors + caller overrides.
	seedShapes := make(map[string]Shape, len(model.TensorNames))

	// 1. Seed ALL tensors with their current shapes.
	for name, t := range e.tensors {
		seedShapes[name] = Shape{Dims: t.shape}
	}

	// 2. Override with caller's shapes (inputs + any constants like position embeds).
	for name, s := range inputShapes {
		seedShapes[name] = s
	}

	// 3. Remove intermediate (non-weight, non-input, non-override) tensors
	//    so InferShapes recomputes them.
	numInputs := len(model.Metadata.InputShapes)
	for i, name := range model.TensorNames {
		if i < numInputs {
			continue // keep inputs
		}
		if e.weightNames[name] {
			// Weight: keep it unless caller explicitly overrode it.
			if _, overridden := inputShapes[name]; !overridden {
				continue
			}
			// Caller overrode this weight — keep the override in seeds.
			continue
		}
		// Intermediate tensor — remove so InferShapes recomputes it.
		delete(seedShapes, name)
	}

	// 4. Re-infer all intermediate shapes.
	allShapes, err := InferShapes(model.Graph, model.TensorNames, seedShapes, e.inferShapeOpts)
	if err != nil {
		return fmt.Errorf("reshape: shape inference: %w", err)
	}

	// 5. Update every tensor's shape/strides in-place.
	for name, s := range allShapes {
		t, ok := e.tensors[name]
		if !ok {
			continue
		}
		if err := t.SetShape(s.Dims); err != nil {
			return fmt.Errorf("reshape tensor %q: %w", name, err)
		}
	}

	return nil
}
