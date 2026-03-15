//go:build js && wasm

// Package wasm provides the WASM entry point for MemPipe.
// Build with: GOOS=js GOARCH=wasm go build -o mempipe.wasm ./platform/wasm/
//
// Exported JS functions:
//
//	mempipe.version()                    → string
//	mempipe.platform()                   → string
//	mempipe.loadModel(Uint8Array)        → modelHandle (int)
//	mempipe.newEngine(modelHandle)       → engineHandle (int)
//	mempipe.infer(engineHandle, Float32Array) → Float32Array
//	mempipe.inferZeroCopy(engineHandle)  → void (uses shared memory)
//	mempipe.getInputPtr(engineHandle, index) → pointer (int) for direct memory access
//	mempipe.getOutputPtr(engineHandle, index) → pointer (int)
//	mempipe.getInputShape(engineHandle, index) → []int
//	mempipe.getOutputShape(engineHandle, index) → []int
//	mempipe.freeEngine(engineHandle)
//	mempipe.freeModel(modelHandle)
//	mempipe.arenaPtr(engineHandle)       → pointer (int)
//	mempipe.arenaSize(engineHandle)      → int
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"syscall/js"
	"unsafe"

	"github.com/GoMemPipe/mempipe/build"
	"github.com/GoMemPipe/mempipe/inference"
)

// ────────────────────────────────────────────────────────────────────────────
// Handle registry — maps integer handles to Go objects
// ────────────────────────────────────────────────────────────────────────────

var (
	mu         sync.Mutex
	nextHandle int = 1
	models         = make(map[int]*inference.Model)
	engines        = make(map[int]*inference.Engine)
)

func allocModelHandle(m *inference.Model) int {
	mu.Lock()
	defer mu.Unlock()
	h := nextHandle
	nextHandle++
	models[h] = m
	return h
}

func allocEngineHandle(e *inference.Engine) int {
	mu.Lock()
	defer mu.Unlock()
	h := nextHandle
	nextHandle++
	engines[h] = e
	return h
}

func getModel(h int) (*inference.Model, error) {
	mu.Lock()
	defer mu.Unlock()
	m, ok := models[h]
	if !ok {
		return nil, fmt.Errorf("invalid model handle: %d", h)
	}
	return m, nil
}

func getEngine(h int) (*inference.Engine, error) {
	mu.Lock()
	defer mu.Unlock()
	e, ok := engines[h]
	if !ok {
		return nil, fmt.Errorf("invalid engine handle: %d", h)
	}
	return e, nil
}

// ────────────────────────────────────────────────────────────────────────────
// JS function wrappers
// ────────────────────────────────────────────────────────────────────────────

// jsVersion returns the MemPipe version string.
func jsVersion(_ js.Value, _ []js.Value) interface{} {
	return build.Version
}

// jsPlatform returns the current build platform ("wasm").
func jsPlatform(_ js.Value, _ []js.Value) interface{} {
	return build.Platform()
}

// jsLoadModel loads a .mpmodel from a Uint8Array and returns a handle.
func jsLoadModel(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("loadModel requires a Uint8Array argument")
	}

	arr := args[0]
	length := arr.Get("byteLength").Int()
	data := make([]byte, length)
	js.CopyBytesToGo(data, arr)

	model, err := inference.LoadModelFromBytes(data)
	if err != nil {
		return jsError("loadModel: " + err.Error())
	}

	return allocModelHandle(model)
}

// jsNewEngine creates an inference engine from a model handle.
func jsNewEngine(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("newEngine requires a model handle")
	}

	model, err := getModel(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	engine, err := inference.NewEngine(model)
	if err != nil {
		return jsError("newEngine: " + err.Error())
	}

	return allocEngineHandle(engine)
}

// jsInfer runs inference with a Float32Array input and returns a Float32Array output.
func jsInfer(_ js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return jsError("infer requires (engineHandle, Float32Array)")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	// Read input Float32Array → byte slice
	inputArr := args[1]
	inputLen := inputArr.Get("length").Int()
	inputBytes := make([]byte, inputLen*4)
	for i := 0; i < inputLen; i++ {
		bits := math.Float32bits(float32(inputArr.Index(i).Float()))
		binary.LittleEndian.PutUint32(inputBytes[i*4:], bits)
	}

	// Copy input into engine input tensors
	inputs := engine.InputTensors()
	off := 0
	for _, t := range inputs {
		bs := t.ByteSize()
		if off+bs > len(inputBytes) {
			return jsError("input data too short")
		}
		if err := t.CopyFrom(inputBytes[off : off+bs]); err != nil {
			return jsError("copy input: " + err.Error())
		}
		off += bs
	}

	// Run inference (zero-alloc hot path)
	outputs, err := engine.InferTensor()
	if err != nil {
		return jsError("infer: " + err.Error())
	}

	// Build output Float32Array
	totalFloats := 0
	for _, t := range outputs {
		totalFloats += t.NumElements()
	}

	result := js.Global().Get("Float32Array").New(totalFloats)
	idx := 0
	for _, t := range outputs {
		n := t.NumElements()
		outBytes := make([]byte, t.ByteSize())
		if err := t.CopyTo(outBytes); err != nil {
			return jsError("copy output: " + err.Error())
		}
		for j := 0; j < n; j++ {
			bits := binary.LittleEndian.Uint32(outBytes[j*4:])
			result.SetIndex(idx, math.Float32frombits(bits))
			idx++
		}
	}

	return result
}

// jsInferZeroCopy runs inference in-place. The caller must have written
// input data directly into the arena (via getInputPtr) before calling this.
func jsInferZeroCopy(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("inferZeroCopy requires an engine handle")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	if _, err := engine.InferTensor(); err != nil {
		return jsError("inferZeroCopy: " + err.Error())
	}

	return js.Undefined()
}

// jsGetInputPtr returns the WASM linear memory pointer of an input tensor.
// JS can create a Float32Array view: new Float32Array(wasmMemory.buffer, ptr, len)
func jsGetInputPtr(_ js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return jsError("getInputPtr requires (engineHandle, index)")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	idx := args[1].Int()
	inputs := engine.InputTensors()
	if idx < 0 || idx >= len(inputs) {
		return jsError(fmt.Sprintf("input index %d out of range [0, %d)", idx, len(inputs)))
	}

	return int(uintptr(inputs[idx].DataPtr()))
}

// jsGetOutputPtr returns the WASM linear memory pointer of an output tensor.
func jsGetOutputPtr(_ js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return jsError("getOutputPtr requires (engineHandle, index)")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	idx := args[1].Int()
	outputs := engine.OutputTensors()
	if idx < 0 || idx >= len(outputs) {
		return jsError(fmt.Sprintf("output index %d out of range [0, %d)", idx, len(outputs)))
	}

	return int(uintptr(outputs[idx].DataPtr()))
}

// jsGetInputShape returns the shape array of an input tensor.
func jsGetInputShape(_ js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return jsError("getInputShape requires (engineHandle, index)")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	idx := args[1].Int()
	inputs := engine.InputTensors()
	if idx < 0 || idx >= len(inputs) {
		return jsError(fmt.Sprintf("input index %d out of range [0, %d)", idx, len(inputs)))
	}

	return intsToJSArray(inputs[idx].Shape())
}

// jsGetOutputShape returns the shape array of an output tensor.
func jsGetOutputShape(_ js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return jsError("getOutputShape requires (engineHandle, index)")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	idx := args[1].Int()
	outputs := engine.OutputTensors()
	if idx < 0 || idx >= len(outputs) {
		return jsError(fmt.Sprintf("output index %d out of range [0, %d)", idx, len(outputs)))
	}

	return intsToJSArray(outputs[idx].Shape())
}

// jsFreeModel releases a model handle.
func jsFreeModel(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("freeModel requires a handle")
	}
	mu.Lock()
	defer mu.Unlock()
	delete(models, args[0].Int())
	return js.Undefined()
}

// jsFreeEngine releases an engine handle.
func jsFreeEngine(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("freeEngine requires a handle")
	}
	mu.Lock()
	defer mu.Unlock()
	delete(engines, args[0].Int())
	return js.Undefined()
}

// jsArenaPtr returns the base pointer of the engine's arena in WASM linear memory.
func jsArenaPtr(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("arenaPtr requires an engine handle")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	// Access the arena memory via the first input tensor's data pointer
	// and subtract its offset. For zero-copy, JS can map a typed array
	// starting from the input tensor pointer directly.
	inputs := engine.InputTensors()
	if len(inputs) == 0 {
		return 0
	}
	return int(uintptr(inputs[0].DataPtr()))
}

// jsArenaSize returns the total arena size in bytes.
func jsArenaSize(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return jsError("arenaSize requires an engine handle")
	}

	engine, err := getEngine(args[0].Int())
	if err != nil {
		return jsError(err.Error())
	}

	return engine.ArenaTotal()
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

func jsError(msg string) js.Value {
	return js.Global().Get("Error").New(msg)
}

func intsToJSArray(vals []int) js.Value {
	arr := js.Global().Get("Array").New(len(vals))
	for i, v := range vals {
		arr.SetIndex(i, v)
	}
	return arr
}

// ────────────────────────────────────────────────────────────────────────────
// Memory helpers for zero-copy JS ↔ WASM data sharing
// ────────────────────────────────────────────────────────────────────────────

// GetArenaPtr returns the base pointer of an InferenceArena's memory in
// WASM linear memory. JS can create Float32Array views directly:
//
//	const ptr = mempipe.getArenaPtr(engineHandle);
//	const view = new Float32Array(wasmMemory.buffer, ptr, size/4);
func GetArenaPtr(engine *inference.Engine) uintptr {
	inputs := engine.InputTensors()
	if len(inputs) == 0 {
		return 0
	}
	return uintptr(inputs[0].DataPtr())
}

// GetArenaSize returns the total arena capacity in bytes.
func GetArenaSize(engine *inference.Engine) int {
	return engine.ArenaTotal()
}

// WASMMemorySize returns the current WASM linear memory size in bytes.
// This is useful for JS wrappers to know the valid address range.
func WASMMemorySize() int {
	// In Go's WASM runtime, the heap is within the linear memory.
	// We can approximate the usable size via a dummy allocation check.
	// For practical purposes, JS accesses memory via wasmMemory.buffer.
	dummy := make([]byte, 1)
	return int(uintptr(unsafe.Pointer(&dummy[0])))
}

// ────────────────────────────────────────────────────────────────────────────
// Main — register exports and block
// ────────────────────────────────────────────────────────────────────────────

func main() {
	mp := js.Global().Get("Object").New()

	// Core info
	mp.Set("version", js.FuncOf(jsVersion))
	mp.Set("platform", js.FuncOf(jsPlatform))

	// Model lifecycle
	mp.Set("loadModel", js.FuncOf(jsLoadModel))
	mp.Set("freeModel", js.FuncOf(jsFreeModel))

	// Engine lifecycle
	mp.Set("newEngine", js.FuncOf(jsNewEngine))
	mp.Set("freeEngine", js.FuncOf(jsFreeEngine))

	// Inference
	mp.Set("infer", js.FuncOf(jsInfer))
	mp.Set("inferZeroCopy", js.FuncOf(jsInferZeroCopy))

	// Zero-copy memory access
	mp.Set("getInputPtr", js.FuncOf(jsGetInputPtr))
	mp.Set("getOutputPtr", js.FuncOf(jsGetOutputPtr))
	mp.Set("getInputShape", js.FuncOf(jsGetInputShape))
	mp.Set("getOutputShape", js.FuncOf(jsGetOutputShape))

	// Arena introspection
	mp.Set("arenaPtr", js.FuncOf(jsArenaPtr))
	mp.Set("arenaSize", js.FuncOf(jsArenaSize))

	// Export to global
	js.Global().Set("mempipe", mp)

	// Block forever — WASM module stays alive
	select {}
}
