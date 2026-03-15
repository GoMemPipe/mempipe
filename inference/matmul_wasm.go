//go:build js && wasm

package inference

import (
	"syscall/js"
	"unsafe"
)

// ════════════════════════════════════════════════════════════════════════════
// matMulAccel — Async WebGPU compute-shader accelerated MatMul (WASM)
//
// Architecture overview (solves the Go WASM event-loop deadlock):
//
//   Init() — called during main() or Engine.Compile(), before select{}:
//     • await(requestAdapter/requestDevice) — safe here, main goroutine
//     • Create WGSL shader module, compute pipeline, bind group layout
//     • Pre-allocate a done-channel and a persistent js.Func callback
//     • Install a JS-side async orchestrator that does the GPU dispatch
//     • Cache all js.Value references for zero-alloc hot path
//
//   matMulF32() — hot path, called from js.FuncOf handlers:
//     • Invoke the cached JS dispatch function with raw int args (zero alloc)
//     • Block on <-doneChan — Go WASM scheduler yields to JS event loop
//     • JS orchestrator runs async: writeBuffer → compute → mapAsync →
//       copy-back → invokes Go done-callback → doneChan unblocks
//
// This is zero-alloc on the Go hot path: no js.ValueOf, no js.FuncOf,
// no closures, no slice/map allocation. All JS values are pre-cached.
// ════════════════════════════════════════════════════════════════════════════

// wgslMatMulShader is the WGSL compute shader for tiled MatMul.
// Workgroup size 16×16 with shared-memory tiling for good GPU occupancy.
const wgslMatMulShader = `
struct Dims {
    M : u32,
    K : u32,
    N : u32,
    pad : u32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> dims : Dims;

const TILE : u32 = 16u;
var<workgroup> tileA : array<f32, 256>;
var<workgroup> tileB : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id)  lid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let localRow = lid.y;
    let localCol = lid.x;

    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    var acc : f32 = 0.0;

    let numTiles = (K + TILE - 1u) / TILE;

    for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
        let aRow = row;
        let aCol = t * TILE + localCol;
        if (aRow < M && aCol < K) {
            tileA[localRow * TILE + localCol] = A[aRow * K + aCol];
        } else {
            tileA[localRow * TILE + localCol] = 0.0;
        }

        let bRow = t * TILE + localRow;
        let bCol = col;
        if (bRow < K && bCol < N) {
            tileB[localRow * TILE + localCol] = B[bRow * N + bCol];
        } else {
            tileB[localRow * TILE + localCol] = 0.0;
        }

        workgroupBarrier();

        for (var p : u32 = 0u; p < TILE; p = p + 1u) {
            acc = acc + tileA[localRow * TILE + p] * tileB[p * TILE + localCol];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
`

// matMulAccel holds the cached WebGPU pipeline state and the async
// dispatch machinery. All fields are set once in Init and reused.
type matMulAccel struct {
	ready bool

	// doneChan is signalled by the persistent JS callback when GPU
	// work completes. Pre-allocated in Init, never re-created.
	doneChan chan struct{}

	// dispatchFn is a cached reference to the JS-side async
	// orchestrator function: self.__mempipe_gpu_dispatch(aPtr, bPtr, cPtr, M, K, N)
	dispatchFn js.Value

	// doneCallback is the persistent js.Func that JS calls when the
	// GPU readback is complete. It sends on doneChan. Created once in
	// Init, never released (lives for the program's lifetime).
	doneCallback js.Func
}

// Init sets up the full WebGPU pipeline and installs the JS-side async
// dispatch orchestrator. Called once during main() via EnsureWebGPU()
// or during Engine.Compile() via the Initializable interface.
//
// All js.FuncOf / js.ValueOf allocations happen here, NOT on the hot path.
func (acc *matMulAccel) Init(arena *InferenceArena) error {
	gpu := js.Global().Get("navigator").Get("gpu")
	if gpu.IsUndefined() || gpu.IsNull() {
		acc.ready = false
		return nil
	}

	// ── Request adapter & device ────────────────────────────────────
	adapter := await(gpu.Call("requestAdapter"))
	if adapter.IsUndefined() || adapter.IsNull() {
		acc.ready = false
		return nil
	}

	// Request maximum buffer limits from the adapter so large weight
	// matrices (e.g. GPT-2 LM head at ~154 MB) don't exceed the
	// default 128 MB maxStorageBufferBindingSize.
	deviceDesc := js.Global().Get("Object").New()
	adapterLimits := adapter.Get("limits")
	if !adapterLimits.IsUndefined() && !adapterLimits.IsNull() {
		reqLimits := js.Global().Get("Object").New()
		for _, key := range []string{
			"maxStorageBufferBindingSize",
			"maxBufferSize",
			"maxComputeWorkgroupStorageSize",
		} {
			v := adapterLimits.Get(key)
			if !v.IsUndefined() && !v.IsNull() {
				reqLimits.Set(key, v)
			}
		}
		deviceDesc.Set("requiredLimits", reqLimits)
	}

	device := await(adapter.Call("requestDevice", deviceDesc))
	if device.IsUndefined() || device.IsNull() {
		acc.ready = false
		return nil
	}

	queue := device.Get("queue")

	// ── Compile shader module ───────────────────────────────────────
	shaderDesc := js.Global().Get("Object").New()
	shaderDesc.Set("code", wgslMatMulShader)
	shaderModule := device.Call("createShaderModule", shaderDesc)

	// ── Bind group layout: 3 storage + 1 uniform ────────────────────
	entries := js.Global().Get("Array").New()
	for i := 0; i < 3; i++ {
		entry := js.Global().Get("Object").New()
		entry.Set("binding", i)
		entry.Set("visibility", 4) // GPUShaderStage.COMPUTE
		bufLayout := js.Global().Get("Object").New()
		if i < 2 {
			bufLayout.Set("type", "read-only-storage")
		} else {
			bufLayout.Set("type", "storage")
		}
		entry.Set("buffer", bufLayout)
		entries.Call("push", entry)
	}
	{
		entry := js.Global().Get("Object").New()
		entry.Set("binding", 3)
		entry.Set("visibility", 4)
		bufLayout := js.Global().Get("Object").New()
		bufLayout.Set("type", "uniform")
		entry.Set("buffer", bufLayout)
		entries.Call("push", entry)
	}
	bglDesc := js.Global().Get("Object").New()
	bglDesc.Set("entries", entries)
	bindGroupLayout := device.Call("createBindGroupLayout", bglDesc)

	// ── Pipeline ────────────────────────────────────────────────────
	plLayoutDesc := js.Global().Get("Object").New()
	bgls := js.Global().Get("Array").New()
	bgls.Call("push", bindGroupLayout)
	plLayoutDesc.Set("bindGroupLayouts", bgls)
	pipelineLayout := device.Call("createPipelineLayout", plLayoutDesc)

	pipelineDesc := js.Global().Get("Object").New()
	pipelineDesc.Set("layout", pipelineLayout)
	compute := js.Global().Get("Object").New()
	compute.Set("module", shaderModule)
	compute.Set("entryPoint", "main")
	pipelineDesc.Set("compute", compute)
	pipeline := device.Call("createComputePipeline", pipelineDesc)

	// ── Resolve WASM linear memory ──────────────────────────────────
	var wasmMemory js.Value
	// Path 1: worker.js sets self.__mempipe_wasm_memory
	wasmMemory = js.Global().Get("__mempipe_wasm_memory")
	if wasmMemory.IsUndefined() || wasmMemory.IsNull() {
		// Path 2: Go runtime _inst.exports.mem
		goObj := js.Global().Get("Go")
		if !goObj.IsUndefined() && !goObj.IsNull() {
			inst := goObj.Get("_inst")
			if !inst.IsUndefined() && !inst.IsNull() {
				exports := inst.Get("exports")
				if !exports.IsUndefined() && !exports.IsNull() {
					for _, name := range []string{"mem", "memory"} {
						m := exports.Get(name)
						if !m.IsUndefined() && !m.IsNull() {
							wasmMemory = m
							break
						}
					}
				}
			}
		}
	}
	if wasmMemory.IsUndefined() || wasmMemory.IsNull() {
		// No WASM memory reference — can't do zero-copy buffer writes.
		acc.ready = false
		return nil
	}

	// ── Pre-allocate done channel & persistent callback ─────────────
	acc.doneChan = make(chan struct{}, 1)

	acc.doneCallback = js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Non-blocking send: if channel already has a value (shouldn't
		// happen), we don't block the JS thread.
		select {
		case acc.doneChan <- struct{}{}:
		default:
		}
		return nil
	})

	// ── Install the JS-side async dispatch orchestrator ──────────────
	// This JS function receives (aPtr, bPtr, cPtr, M, K, N) as plain
	// integers, runs the full GPU dispatch asynchronously, then calls
	// the Go done-callback when the result has been copied back into
	// WASM linear memory.
	//
	// We use js.Global().Call("eval", ...) to define the function once.
	// The function captures device, queue, pipeline, bindGroupLayout,
	// wasmMemory, and the Go done-callback via closure in the JS scope.
	//
	// We store these objects on a global JS namespace so the function
	// body can reference them.
	ctx := js.Global().Get("Object").New()
	ctx.Set("device", device)
	ctx.Set("queue", queue)
	ctx.Set("pipeline", pipeline)
	ctx.Set("bindGroupLayout", bindGroupLayout)
	ctx.Set("wasmMemory", wasmMemory)
	ctx.Set("doneCallback", acc.doneCallback)
	js.Global().Set("__mempipe_gpu_ctx", ctx)

	// Define the async dispatch function. It must never throw without
	// calling doneCallback, or the Go side will hang forever.
	js.Global().Call("eval", `
self.__mempipe_gpu_dispatch = async function(aPtr, bPtr, cPtr, M, K, N) {
  const ctx = self.__mempipe_gpu_ctx;
  const device = ctx.device;
  const queue = ctx.queue;
  try {
    const aBytes = M * K * 4;
    const bBytes = K * N * 4;
    const cBytes = M * N * 4;

    // GPU buffer usage flags
    const COPY_DST = 0x0008, COPY_SRC = 0x0004;
    const STORAGE  = 0x0080, UNIFORM  = 0x0040;
    const MAP_READ = 0x0001;

    const bufA    = device.createBuffer({ size: aBytes, usage: STORAGE | COPY_DST });
    const bufB    = device.createBuffer({ size: bBytes, usage: STORAGE | COPY_DST });
    const bufC    = device.createBuffer({ size: cBytes, usage: STORAGE | COPY_SRC });
    const bufDims = device.createBuffer({ size: 16,     usage: UNIFORM | COPY_DST });

    // Zero-copy views into WASM linear memory
    const mem = ctx.wasmMemory.buffer;
    queue.writeBuffer(bufA, 0, new Float32Array(mem, aPtr, M * K));
    queue.writeBuffer(bufB, 0, new Float32Array(mem, bPtr, K * N));
    queue.writeBuffer(bufDims, 0, new Uint32Array([M, K, N, 0]));

    // Bind group
    const bindGroup = device.createBindGroup({
      layout: ctx.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufDims } },
      ],
    });

    // Encode & submit
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(ctx.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(((N + 15) >>> 4), ((M + 15) >>> 4));
    pass.end();

    // Staging buffer for readback
    const staging = device.createBuffer({ size: cBytes, usage: MAP_READ | COPY_DST });
    encoder.copyBufferToBuffer(bufC, 0, staging, 0, cBytes);

    queue.submit([encoder.finish()]);

    // Async readback — this is the key: the await yields to the JS
    // event loop, which is exactly what we need.
    await staging.mapAsync(1); // GPUMapMode.READ = 1
    const mapped = new Float32Array(staging.getMappedRange());

    // Copy result back into WASM linear memory (arena-backed C tensor)
    new Float32Array(ctx.wasmMemory.buffer, cPtr, M * N).set(mapped);
    staging.unmap();

    // Cleanup GPU buffers
    bufA.destroy();
    bufB.destroy();
    bufC.destroy();
    bufDims.destroy();
    staging.destroy();
  } catch(e) {
    // Log but don't re-throw — we must always signal done.
    console.error('[mempipe WebGPU dispatch error]', e);
  }
  // Signal Go that the result is ready.
  ctx.doneCallback();
};
`)

	acc.dispatchFn = js.Global().Get("__mempipe_gpu_dispatch")
	if acc.dispatchFn.IsUndefined() || acc.dispatchFn.IsNull() {
		acc.ready = false
		return nil
	}

	acc.ready = true
	return nil
}

// ════════════════════════════════════════════════════════════════════════════
// Hot path — zero allocations
// ════════════════════════════════════════════════════════════════════════════

// matMulF32 dispatches MatMul to WebGPU if the pipeline is ready,
// otherwise falls back to the generic i-p-j loop.
//
// When using the GPU path:
//  1. Invoke the pre-cached JS dispatch function with raw int args.
//     js.Value.Invoke(int...) does NOT allocate in Go WASM — integer
//     args are passed directly without js.ValueOf boxing.
//  2. Block on <-doneChan. The Go WASM scheduler sees the goroutine
//     is parked on a channel and yields control to the JS event loop.
//  3. The JS async function runs the GPU work, then calls the pre-
//     allocated Go callback, which sends on doneChan.
//  4. The Go goroutine wakes up with the result already written into
//     the C tensor's arena memory (zero-copy via WASM linear memory).
//
//mem:hot
//mem:nogc
func matMulF32(a, b, c []float32, m, k, n int) {
	if !wasmAccel.ready {
		matMulF32Generic(a, b, c, m, k, n)
		return
	}

	// Pointer offsets into WASM linear memory — plain int arithmetic,
	// no allocation. These are byte offsets that JS uses to create
	// Float32Array views directly over the arena-backed slices.
	aPtr := int(uintptr(unsafe.Pointer(&a[0])))
	bPtr := int(uintptr(unsafe.Pointer(&b[0])))
	cPtr := int(uintptr(unsafe.Pointer(&c[0])))

	// Dispatch to JS — all args are plain ints, no js.ValueOf needed.
	// The JS function is async: it returns a Promise immediately, but
	// we don't touch the return value. We wait on the channel instead.
	wasmAccel.dispatchFn.Invoke(aPtr, bPtr, cPtr, m, k, n)

	// Yield to JS event loop. The Go scheduler parks this goroutine
	// on the channel receive. When the JS async function completes,
	// it calls our pre-allocated callback which sends on doneChan.
	<-wasmAccel.doneChan
}

// matMulF32Generic is the portable fallback kernel.
//
//mem:hot
//mem:nogc
func matMulF32Generic(a, b, c []float32, m, k, n int) {
	for i := 0; i < m; i++ {
		rowA := a[i*k : i*k+k]
		rowC := c[i*n : i*n+n]
		for j := range rowC {
			rowC[j] = 0
		}
		for p := 0; p < k; p++ {
			aVal := rowA[p]
			rowB := b[p*n : p*n+n]
			for j := 0; j < n; j++ {
				rowC[j] += aVal * rowB[j]
			}
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Package-level singleton — shared by matMulOp, denseOp, batchedMatMulOp
// ────────────────────────────────────────────────────────────────────────────

var wasmAccel matMulAccel

// initMatMulAccel initializes the given matMulAccel and promotes it
// to the package-level singleton so all callers (denseOp, batchedMatMulOp,
// standalone benchmarks) share the same GPU pipeline.
func initMatMulAccel(acc *matMulAccel, arena *InferenceArena) error {
	// Only init once — if the singleton is already ready, copy it.
	if wasmAccel.ready {
		*acc = wasmAccel
		return nil
	}
	if err := acc.Init(arena); err != nil {
		return err
	}
	// Promote to singleton so matMulF32 (called by denseOp etc.) uses it.
	wasmAccel = *acc
	return nil
}

// EnsureWebGPU initialises the WebGPU MatMul pipeline if it hasn't been
// done yet. Call this before standalone GetOperator + Execute sequences
// (e.g. benchmarks) that bypass the Engine's Initializable flow.
func EnsureWebGPU() {
	if wasmAccel.ready {
		return
	}
	arena := NewInferenceArena(64)
	_ = wasmAccel.Init(arena)
}

// IsWebGPUReady reports whether the WebGPU MatMul pipeline is initialised.
func IsWebGPUReady() bool {
	return wasmAccel.ready
}

// ────────────────────────────────────────────────────────────────────────────
// JS helper — used only during Init (not on hot path)
// ────────────────────────────────────────────────────────────────────────────

// await blocks the Go goroutine until the JS Promise resolves.
// Only safe to call during main() init, NOT from js.FuncOf handlers.
func await(promise js.Value) js.Value {
	ch := make(chan js.Value, 1)
	var then, catch js.Func
	then = js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) > 0 {
			ch <- args[0]
		} else {
			ch <- js.Undefined()
		}
		then.Release()
		catch.Release()
		return nil
	})
	catch = js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		ch <- js.Undefined()
		then.Release()
		catch.Release()
		return nil
	})
	promise.Call("then", then).Call("catch", catch)
	return <-ch
}
