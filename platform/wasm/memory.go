//go:build js && wasm

package main

import (
	"unsafe"

	"github.com/GoMemPipe/mempipe/inference"
)

// ────────────────────────────────────────────────────────────────────────────
// WASM Memory Helpers
//
// These functions allow JavaScript to create zero-copy views over
// WASM linear memory holding arena-backed tensor data.
//
// Usage from JS:
//
//	const ptr = mempipe.getInputPtr(engine, 0);
//	const shape = mempipe.getInputShape(engine, 0);
//	const numFloats = shape.reduce((a, b) => a * b, 1);
//	const view = new Float32Array(wasmMemory.buffer, ptr, numFloats);
//	// Write directly into view — zero copy!
//	view.set(myInputData);
//	mempipe.inferZeroCopy(engine);
//	const outPtr = mempipe.getOutputPtr(engine, 0);
//	const outShape = mempipe.getOutputShape(engine, 0);
//	const outView = new Float32Array(wasmMemory.buffer, outPtr, outShape.reduce((a, b) => a * b, 1));
//
// ────────────────────────────────────────────────────────────────────────────

// TensorPtr returns the raw WASM linear memory pointer for a tensor.
// The pointer is a byte offset into the WASM linear memory (wasmMemory.buffer).
func TensorPtr(t *inference.Tensor) uintptr {
	return uintptr(t.DataPtr())
}

// TensorByteSize returns the total byte size of a tensor's data.
func TensorByteSize(t *inference.Tensor) int {
	return t.ByteSize()
}

// TensorFloat32Count returns the number of float32 elements in a tensor.
// For use with Float32Array views: new Float32Array(buffer, ptr, count).
func TensorFloat32Count(t *inference.Tensor) int {
	return t.ByteSize() / 4
}

// SlicePtr returns the WASM linear memory pointer for a Go byte slice.
// This enables JS to create typed array views over arbitrary Go memory.
//
// WARNING: The slice must not be garbage collected while the pointer is in use.
// Keep a reference to the slice on the Go side.
func SlicePtr(b []byte) uintptr {
	if len(b) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&b[0]))
}

// PtrToFloat32Slice creates a Float32 slice view over a raw pointer.
// This is the inverse of SlicePtr — used for reading back results.
//
//go:nosplit
func PtrToFloat32Slice(ptr uintptr, count int) []float32 {
	if count <= 0 || ptr == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count)
}

// CopyFloat32ToWASM copies a Go float32 slice directly into WASM memory
// at the specified pointer. Used for bulk input data transfer.
func CopyFloat32ToWASM(dst uintptr, src []float32) {
	if len(src) == 0 || dst == 0 {
		return
	}
	dstSlice := unsafe.Slice((*float32)(unsafe.Pointer(dst)), len(src))
	copy(dstSlice, src)
}

// CopyFloat32FromWASM copies float32 data from WASM memory into a Go slice.
func CopyFloat32FromWASM(src uintptr, count int) []float32 {
	if count <= 0 || src == 0 {
		return nil
	}
	srcSlice := unsafe.Slice((*float32)(unsafe.Pointer(src)), count)
	dst := make([]float32, count)
	copy(dst, srcSlice)
	return dst
}

// EngineInputPtrs returns all input tensor pointers and their float32 counts.
// Useful for bulk setup of zero-copy input views.
func EngineInputPtrs(engine *inference.Engine) (ptrs []uintptr, counts []int) {
	inputs := engine.InputTensors()
	ptrs = make([]uintptr, len(inputs))
	counts = make([]int, len(inputs))
	for i, t := range inputs {
		ptrs[i] = uintptr(t.DataPtr())
		counts[i] = t.ByteSize() / 4
	}
	return
}

// EngineOutputPtrs returns all output tensor pointers and their float32 counts.
func EngineOutputPtrs(engine *inference.Engine) (ptrs []uintptr, counts []int) {
	outputs := engine.OutputTensors()
	ptrs = make([]uintptr, len(outputs))
	counts = make([]int, len(outputs))
	for i, t := range outputs {
		ptrs[i] = uintptr(t.DataPtr())
		counts[i] = t.ByteSize() / 4
	}
	return
}
