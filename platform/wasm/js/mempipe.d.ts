/**
 * MemPipe WASM inference engine — TypeScript declarations.
 * @module mempipe
 */

/**
 * Main MemPipe WASM runtime wrapper.
 */
export declare class MemPipe {
  /**
   * Load a MemPipe WASM module from a URL, ArrayBuffer, or Uint8Array.
   */
  static load(wasmSource: string | ArrayBuffer | Uint8Array): Promise<MemPipe>;

  /** WASM linear memory buffer. */
  readonly memory: ArrayBuffer;

  /** MemPipe library version. */
  readonly version: string;

  /** Build platform identifier ("wasm"). */
  readonly platform: string;

  /**
   * Load a .mpmodel from raw bytes.
   */
  loadModel(bytes: Uint8Array): Model;

  /**
   * Create an inference engine from a loaded model.
   */
  newEngine(model: Model): InferenceEngine;
}

/**
 * A loaded .mpmodel neural network model.
 */
export declare class Model {
  /** Release the model memory. */
  free(): void;
}

/**
 * Inference engine with zero-copy WASM memory access.
 */
export declare class InferenceEngine {
  /**
   * Run inference (copies input/output).
   */
  infer(input: Float32Array): Float32Array;

  /**
   * Run zero-copy inference. Write to inputView() first,
   * read from outputView() after.
   */
  inferZeroCopy(): void;

  /**
   * Get a writable Float32Array view into the input tensor's
   * WASM linear memory. Zero-copy.
   */
  inputView(index: number): Float32Array;

  /**
   * Get a readable Float32Array view into the output tensor's
   * WASM linear memory. Zero-copy.
   */
  outputView(index: number): Float32Array;

  /** Get the shape of an input tensor. */
  inputShape(index: number): number[];

  /** Get the shape of an output tensor. */
  outputShape(index: number): number[];

  /** Get arena memory info. */
  arenaInfo(): { ptr: number; size: number };

  /** Release the engine. */
  free(): void;
}
