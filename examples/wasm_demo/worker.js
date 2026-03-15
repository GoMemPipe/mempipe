// worker.js — Runs Go WASM in a dedicated Web Worker so the main thread
// stays responsive during heavy operations (model loading, inference, benchmarks).
//
// Protocol:
//   Main → Worker:  { id, method, args }
//   Worker → Main:  { id, type:'result', result }
//                   { id, type:'error', error }
//                   { id, type:'progress', step, total, token, timeMs }
//   Worker → Main:  { type:'init', ok, error?, loadTimeMs? }

importScripts('wasm_exec.js');

// ── WASM Initialization ────────────────────────────────────────────────

(async () => {
  try {
    const t0 = performance.now();
    const go = new Go();
    const resp = await fetch('demo.wasm');
    if (!resp.ok) throw new Error(`fetch demo.wasm: ${resp.status}`);
    const wasmBytes = await resp.arrayBuffer();
    const result = await WebAssembly.instantiate(wasmBytes, go.importObject);

    // Expose the WASM linear memory so Go's matmul_wasm.go can find it
    // for zero-copy GPU buffer writes. The Go runtime needs this to create
    // Float32Array views over arena memory without copying.
    const wasmMemory = result.instance.exports.mem || result.instance.exports.memory;
    if (wasmMemory) {
      self.__mempipe_wasm_memory = wasmMemory;
    }

    go.run(result.instance);

    // Wait for Go main() to finish EnsureWebGPU() (which does async
    // requestAdapter + requestDevice via await()) and register the
    // mempipeDemo global. Each iteration yields to the JS event loop
    // so Go's Promise callbacks (_resume cycles) can fire.
    let waited = 0;
    while (!self.mempipeDemo && waited < 10000) {
      await new Promise(r => setTimeout(r, 50));
      waited += 50;
    }
    if (!self.mempipeDemo) throw new Error('mempipeDemo global not registered after ' + waited + 'ms');

    const loadTimeMs = performance.now() - t0;
    postMessage({
      type: 'init', ok: true,
      loadTimeMs: loadTimeMs,
      wasmBytes: wasmBytes.byteLength,
    });
  } catch (err) {
    postMessage({ type: 'init', ok: false, error: err.message });
  }
})();

// ── Message Handler ────────────────────────────────────────────────────

self.onmessage = async function (e) {
  const { id, method, args } = e.data;

  try {
    switch (method) {
    // ── Simple pass-through calls ─────────────────────────────────
    // All Go callbacks now return Promises (via goAsync wrapper),
    // so we must await them to get the resolved value.
    case 'info':
    case 'testArena':
    case 'testZeroCopy':
    case 'benchOperators':
    case 'loadGPT2':
    case 'benchMatMul': {
      const fn = self.mempipeDemo[method];
      if (!fn) throw new Error(`unknown method: ${method}`);
      const result = await fn.apply(null, args || []);
      postMessage({ id, type: 'result', result: toPlain(result) });
      break;
    }

    // ── Per-token generation with progress ────────────────────────
    case 'generateGPT2': {
      const [tokenIDs, maxTokens, temperature] = args;
      let tokens = Array.from(tokenIDs);
      const generated = [];
      const stepTimings = [];
      const totalStart = performance.now();

      for (let step = 0; step < maxTokens; step++) {
        // generateStep returns a Promise (goAsync wrapper) — must await.
        const stepResult = await self.mempipeDemo.generateStep(tokens, temperature);
        const plain = toPlain(stepResult);

        if (plain.error && plain.error !== null) {
          postMessage({ id, type: 'error', error: String(plain.error) });
          return;
        }

        const token = plain.token;
        const timeMs = plain.timeMs;
        tokens.push(token);
        generated.push(token);
        stepTimings.push({ step, token, timeMs });

        // Post progress so the main thread can update UI
        postMessage({
          id, type: 'progress',
          step: step + 1, total: maxTokens,
          token, timeMs,
        });

        // Yield the worker event loop so posted messages get delivered
        await new Promise(r => setTimeout(r, 0));
      }

      const totalMs = performance.now() - totalStart;
      postMessage({
        id, type: 'result',
        result: {
          tokens: generated,
          stepTimings,
          totalMs,
          tokensPerSec: generated.length > 0 ? generated.length / (totalMs / 1000) : 0,
          inputTokens: tokenIDs.length,
          generatedCount: generated.length,
          error: null,
        },
      });
      break;
    }

    default:
      postMessage({ id, type: 'error', error: `unknown method: ${method}` });
    }
  } catch (err) {
    postMessage({ id, type: 'error', error: err.message || String(err) });
  }
};

// ── Helpers ────────────────────────────────────────────────────────────

// Convert Go WASM js.Value objects to plain structured-clonable values.
function toPlain(val) {
  if (val instanceof Error) return { error: val.message };
  if (val == null) return val;
  if (typeof val === 'number' || typeof val === 'string' || typeof val === 'boolean') return val;

  if (Array.isArray(val)) {
    const arr = new Array(val.length);
    for (let i = 0; i < val.length; i++) arr[i] = toPlain(val[i]);
    return arr;
  }

  // Plain object — recursively convert
  const obj = {};
  const keys = Object.keys(val);
  for (let i = 0; i < keys.length; i++) {
    obj[keys[i]] = toPlain(val[keys[i]]);
  }
  return obj;
}
