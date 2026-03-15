/**
 * MemPipe — JavaScript/TypeScript wrapper for the WASM inference engine.
 *
 * Usage (browser):
 *
 *   import { MemPipe } from './mempipe.js';
 *
 *   const mp = await MemPipe.load('mempipe.wasm');
 *   const model = mp.loadModel(modelBytes);
 *   const engine = mp.newEngine(model);
 *
 *   // High-level API (copies data)
 *   const output = engine.infer(new Float32Array([...]));
 *
 *   // Zero-copy API (direct WASM memory access)
 *   const input = engine.inputView(0);   // Float32Array into WASM memory
 *   input.set(myData);
 *   engine.inferZeroCopy();
 *   const result = engine.outputView(0); // Float32Array into WASM memory
 *
 *   engine.free();
 *   model.free();
 *
 * Usage (Node.js):
 *
 *   const { MemPipe } = require('./mempipe.js');
 *   const fs = require('fs');
 *   const mp = await MemPipe.load(fs.readFileSync('mempipe.wasm'));
 *
 * @module mempipe
 */

'use strict';

/**
 * MemPipe WASM runtime wrapper.
 */
class MemPipe {
  /**
   * @param {WebAssembly.Instance} instance
   * @param {object} go - Go runtime instance
   */
  constructor(instance, go) {
    this._instance = instance;
    this._go = go;
    this._mp = globalThis.mempipe;
    if (!this._mp) {
      throw new Error('MemPipe WASM module did not register global mempipe object');
    }
  }

  /**
   * Load a MemPipe WASM module.
   *
   * @param {string|ArrayBuffer|Uint8Array} wasmSource
   *   - string: URL to fetch the .wasm file (browser)
   *   - ArrayBuffer/Uint8Array: raw WASM bytes (Node.js or pre-fetched)
   * @returns {Promise<MemPipe>}
   */
  static async load(wasmSource) {
    // Go runtime from wasm_exec.js must be loaded first
    if (typeof Go === 'undefined') {
      throw new Error(
        'Go WASM runtime not found. Load wasm_exec.js before mempipe.js:\n' +
        '  <script src="wasm_exec.js"></script>'
      );
    }

    const go = new Go();

    let wasmBytes;
    if (typeof wasmSource === 'string') {
      // URL — fetch it
      if (typeof fetch === 'function') {
        const resp = await fetch(wasmSource);
        wasmBytes = await resp.arrayBuffer();
      } else {
        // Node.js fallback
        const fs = await import('fs');
        const path = await import('path');
        wasmBytes = fs.readFileSync(wasmSource);
      }
    } else if (wasmSource instanceof ArrayBuffer) {
      wasmBytes = wasmSource;
    } else if (wasmSource instanceof Uint8Array) {
      wasmBytes = wasmSource.buffer;
    } else {
      throw new Error('wasmSource must be a URL string, ArrayBuffer, or Uint8Array');
    }

    const result = await WebAssembly.instantiate(wasmBytes, go.importObject);
    // Run the Go main() — it registers the global `mempipe` object and blocks
    go.run(result.instance);

    // Small delay to let Go's main() register the global
    await new Promise(r => setTimeout(r, 0));

    return new MemPipe(result.instance, go);
  }

  /**
   * Get the WASM linear memory buffer.
   * @returns {ArrayBuffer}
   */
  get memory() {
    return this._instance.exports.mem.buffer;
  }

  /** @returns {string} MemPipe version */
  get version() {
    return this._mp.version();
  }

  /** @returns {string} Platform identifier ("wasm") */
  get platform() {
    return this._mp.platform();
  }

  /**
   * Load a .mpmodel from raw bytes.
   *
   * @param {Uint8Array} bytes - Raw .mpmodel file bytes
   * @returns {Model}
   */
  loadModel(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      throw new Error('loadModel requires a Uint8Array');
    }
    const handle = this._mp.loadModel(bytes);
    if (handle instanceof Error) throw handle;
    return new Model(this, handle);
  }

  /**
   * Create an inference engine from a loaded model.
   *
   * @param {Model} model
   * @returns {InferenceEngine}
   */
  newEngine(model) {
    if (!(model instanceof Model)) {
      throw new Error('newEngine requires a Model instance');
    }
    const handle = this._mp.newEngine(model._handle);
    if (handle instanceof Error) throw handle;
    return new InferenceEngine(this, handle);
  }
}

/**
 * Represents a loaded .mpmodel neural network model.
 */
class Model {
  /**
   * @param {MemPipe} mp
   * @param {number} handle
   */
  constructor(mp, handle) {
    this._mp = mp;
    this._handle = handle;
  }

  /** Release the model. Must be called when done. */
  free() {
    if (this._handle !== null) {
      this._mp._mp.freeModel(this._handle);
      this._handle = null;
    }
  }
}

/**
 * Inference engine backed by a single pre-allocated arena.
 * Supports both high-level (copy) and zero-copy inference.
 */
class InferenceEngine {
  /**
   * @param {MemPipe} mp
   * @param {number} handle
   */
  constructor(mp, handle) {
    this._mp = mp;
    this._handle = handle;
  }

  /**
   * Run inference with input data (copies in and out).
   *
   * @param {Float32Array} input - Input tensor data (flattened, row-major)
   * @returns {Float32Array} Output tensor data
   */
  infer(input) {
    if (!(input instanceof Float32Array)) {
      throw new Error('infer requires a Float32Array');
    }
    const result = this._mp._mp.infer(this._handle, input);
    if (result instanceof Error) throw result;
    return result;
  }

  /**
   * Run zero-copy inference. Input data must be written to inputView()
   * before calling this. Output is readable from outputView() after.
   */
  inferZeroCopy() {
    const result = this._mp._mp.inferZeroCopy(this._handle);
    if (result instanceof Error) throw result;
  }

  /**
   * Get a Float32Array view directly into the WASM linear memory
   * for input tensor at the given index. Writing to this view writes
   * directly into the inference arena — zero copy.
   *
   * @param {number} index - Input tensor index (0-based)
   * @returns {Float32Array}
   */
  inputView(index) {
    const ptr = this._mp._mp.getInputPtr(this._handle, index);
    if (ptr instanceof Error) throw ptr;
    const shape = this._mp._mp.getInputShape(this._handle, index);
    if (shape instanceof Error) throw shape;
    const count = Array.from(shape).reduce((a, b) => a * b, 1);
    return new Float32Array(this._mp.memory, ptr, count);
  }

  /**
   * Get a Float32Array view directly into the WASM linear memory
   * for output tensor at the given index. Read-only after inference.
   *
   * @param {number} index - Output tensor index (0-based)
   * @returns {Float32Array}
   */
  outputView(index) {
    const ptr = this._mp._mp.getOutputPtr(this._handle, index);
    if (ptr instanceof Error) throw ptr;
    const shape = this._mp._mp.getOutputShape(this._handle, index);
    if (shape instanceof Error) throw shape;
    const count = Array.from(shape).reduce((a, b) => a * b, 1);
    return new Float32Array(this._mp.memory, ptr, count);
  }

  /**
   * Get the shape of input tensor at the given index.
   *
   * @param {number} index
   * @returns {number[]}
   */
  inputShape(index) {
    const shape = this._mp._mp.getInputShape(this._handle, index);
    if (shape instanceof Error) throw shape;
    return Array.from(shape);
  }

  /**
   * Get the shape of output tensor at the given index.
   *
   * @param {number} index
   * @returns {number[]}
   */
  outputShape(index) {
    const shape = this._mp._mp.getOutputShape(this._handle, index);
    if (shape instanceof Error) throw shape;
    return Array.from(shape);
  }

  /**
   * Get arena memory usage information.
   *
   * @returns {{ ptr: number, size: number }}
   */
  arenaInfo() {
    const ptr = this._mp._mp.arenaPtr(this._handle);
    const size = this._mp._mp.arenaSize(this._handle);
    return { ptr: ptr instanceof Error ? 0 : ptr, size: size instanceof Error ? 0 : size };
  }

  /** Release the engine. Must be called when done. */
  free() {
    if (this._handle !== null) {
      this._mp._mp.freeEngine(this._handle);
      this._handle = null;
    }
  }
}

// Export for both ESM and CJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MemPipe, Model, InferenceEngine };
}
if (typeof globalThis !== 'undefined') {
  globalThis.MemPipeJS = { MemPipe, Model, InferenceEngine };
}
