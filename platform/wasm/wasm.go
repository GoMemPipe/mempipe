//go:build js && wasm

// Package main provides WASM-specific entry points and helpers for MemPipe.
//
// Build with: GOOS=js GOARCH=wasm go build -o mempipe.wasm ./platform/wasm/
//
// This package contains:
//   - main.go: WASM entry point that exports functions to JavaScript via syscall/js
//   - memory.go: zero-copy memory helpers for sharing arena data with JS
//   - js/mempipe.js: idiomatic JS wrapper with Pipeline, Model, InferenceEngine classes
//   - js/mempipe.d.ts: TypeScript type declarations
//
// Browser usage:
//
//	<script src="wasm_exec.js"></script>
//	<script src="mempipe.js"></script>
//	<script>
//	  const mp = await MemPipeJS.MemPipe.load('mempipe.wasm');
//	  const model = mp.loadModel(modelBytes);
//	  const engine = mp.newEngine(model);
//	  const output = engine.infer(inputData);
//	</script>
//
// Node.js usage:
//
//	require('./wasm_exec.js');
//	const { MemPipe } = require('./mempipe.js');
//	const mp = await MemPipe.load('mempipe.wasm');
package main
