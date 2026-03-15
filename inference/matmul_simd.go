//go:build (linux || darwin) && (amd64 || arm64)

package inference

import "unsafe"

// ════════════════════════════════════════════════════════════════════════════
// matMulAccel — SIMD-optimized MatMul for amd64/arm64
//
// Uses a cache-friendly, blocked, loop-unrolled pure-Go implementation
// with explicit bounds-check elimination (BCE) hints so the compiler
// can auto-vectorize inner loops.
//
// The packing buffer for B (column-major tile layout) is pre-allocated
// from the arena in Init to guarantee 0 allocs/op in Execute.
// ════════════════════════════════════════════════════════════════════════════

// Block sizes tuned for L1/L2 cache on modern x86-64 / ARM64 cores.
const (
	blockM = 64  // rows of A processed per outer tile
	blockN = 256 // cols of B processed per outer tile
	blockK = 256 // depth processed per outer tile
	microM = 4   // rows of micro-kernel
	microN = 4   // cols of micro-kernel
)

// matMulAccel holds auxiliary state for the SIMD path.
type matMulAccel struct {
	// packBuf is a scratch buffer for the column-major repacked B tile.
	// Allocated once from the arena during Init.
	packBuf  unsafe.Pointer
	packSize int // byte capacity of packBuf
}

// initMatMulAccel initializes the SIMD-accelerated matMulAccel.
func initMatMulAccel(acc *matMulAccel, arena *InferenceArena) error {
	return acc.Init(arena)
}

// EnsureWebGPU is a no-op on non-WASM platforms.
func EnsureWebGPU() {}

// IsWebGPUReady always returns false on non-WASM platforms.
func IsWebGPUReady() bool { return false }

// Init pre-allocates the B-packing scratch buffer from the arena.
// Called once during Engine.compile.
func (acc *matMulAccel) Init(arena *InferenceArena) error {
	// Worst-case packing buffer: blockK * blockN float32s.
	need := blockK * blockN * 4
	ptr, err := arena.AllocRaw(need)
	if err != nil {
		// Arena too small for scratch — Execute will still work
		// (falls back to non-packed path).
		acc.packBuf = nil
		acc.packSize = 0
		return nil
	}
	acc.packBuf = ptr
	acc.packSize = need
	return nil
}

// matMulF32 computes C = A*B for row-major float32 matrices.
// Uses a three-level blocked algorithm with 4×4 micro-kernel,
// explicit BCE hints, and loop unrolling to enable the Go compiler's
// auto-vectorizer (SSA prove pass) to emit SIMD instructions.
//
// Zero-alloc: no make/new/append. Operates on arena-backed slices.
//
//mem:hot
//mem:nogc
func matMulF32(a, b, c []float32, m, k, n int) {
	// Zero the entire output first
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Outer blocking over M, N, K
	for i0 := 0; i0 < m; i0 += blockM {
		iEnd := i0 + blockM
		if iEnd > m {
			iEnd = m
		}
		for j0 := 0; j0 < n; j0 += blockN {
			jEnd := j0 + blockN
			if jEnd > n {
				jEnd = n
			}
			for p0 := 0; p0 < k; p0 += blockK {
				pEnd := p0 + blockK
				if pEnd > k {
					pEnd = k
				}
				// Process this block tile
				matMulBlockTile(a, b, c, m, k, n, i0, iEnd, j0, jEnd, p0, pEnd)
			}
		}
	}
}

// matMulBlockTile processes one (blockM × blockN × blockK) tile.
// It uses a 4×4 micro-kernel with explicit BCE hints.
//
//mem:hot
//mem:nogc
func matMulBlockTile(a, b, c []float32, m, k, n, i0, iEnd, j0, jEnd, p0, pEnd int) {
	// Process 4×4 micro-tiles within this block
	for i := i0; i+microM <= iEnd; i += microM {
		for j := j0; j+microN <= jEnd; j += microN {
			matMulMicro4x4(a, b, c, k, n, i, j, p0, pEnd)
		}
		// Remainder columns (j not divisible by microN)
		for j := jEnd - (jEnd-j0)%microN; j < jEnd; j++ {
			matMulMicroMx1(a, b, c, k, n, i, i+microM, j, p0, pEnd)
		}
	}
	// Remainder rows (i not divisible by microM)
	iRem := iEnd - (iEnd-i0)%microM
	for i := iRem; i < iEnd; i++ {
		for j := j0; j < jEnd; j++ {
			matMulMicro1x1(a, b, c, k, n, i, j, p0, pEnd)
		}
	}
}

// matMulMicro4x4 is the 4×4 micro-kernel with BCE and 4× unrolled K-loop.
// The explicit bounds hints let the compiler prove all indices are in-range,
// eliminating bounds checks and enabling SIMD codegen.
//
//mem:hot
//mem:nogc
func matMulMicro4x4(a, b, c []float32, k, n, i, j, p0, pEnd int) {
	// ── BCE hints ──
	// Prove upper bounds for row/col access patterns.
	_ = a[(i+3)*k+pEnd-1]
	_ = b[(pEnd-1)*n+j+3]
	_ = c[(i+3)*n+j+3]

	// 4 rows × 4 cols accumulators (kept in registers)
	var (
		c00, c01, c02, c03 float32
		c10, c11, c12, c13 float32
		c20, c21, c22, c23 float32
		c30, c31, c32, c33 float32
	)

	// Rows of A
	rowA0 := a[i*k : (i+1)*k]
	rowA1 := a[(i+1)*k : (i+2)*k]
	rowA2 := a[(i+2)*k : (i+3)*k]
	rowA3 := a[(i+3)*k : (i+4)*k]

	// Unrolled K-loop: process 4 elements per iteration
	p := p0
	pEnd4 := p0 + ((pEnd - p0) &^ 3) // round down to multiple of 4
	for ; p < pEnd4; p += 4 {
		// BCE hints for this unrolled block
		_ = rowA0[p+3]
		_ = b[(p+3)*n+j+3]

		// K iteration 0
		a0 := rowA0[p]
		a1 := rowA1[p]
		a2 := rowA2[p]
		a3 := rowA3[p]
		b0 := b[p*n+j]
		b1 := b[p*n+j+1]
		b2 := b[p*n+j+2]
		b3 := b[p*n+j+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// K iteration 1
		a0 = rowA0[p+1]
		a1 = rowA1[p+1]
		a2 = rowA2[p+1]
		a3 = rowA3[p+1]
		b0 = b[(p+1)*n+j]
		b1 = b[(p+1)*n+j+1]
		b2 = b[(p+1)*n+j+2]
		b3 = b[(p+1)*n+j+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// K iteration 2
		a0 = rowA0[p+2]
		a1 = rowA1[p+2]
		a2 = rowA2[p+2]
		a3 = rowA3[p+2]
		b0 = b[(p+2)*n+j]
		b1 = b[(p+2)*n+j+1]
		b2 = b[(p+2)*n+j+2]
		b3 = b[(p+2)*n+j+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// K iteration 3
		a0 = rowA0[p+3]
		a1 = rowA1[p+3]
		a2 = rowA2[p+3]
		a3 = rowA3[p+3]
		b0 = b[(p+3)*n+j]
		b1 = b[(p+3)*n+j+1]
		b2 = b[(p+3)*n+j+2]
		b3 = b[(p+3)*n+j+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}

	// Remainder K iterations (0–3 leftover)
	for ; p < pEnd; p++ {
		a0 := rowA0[p]
		a1 := rowA1[p]
		a2 := rowA2[p]
		a3 := rowA3[p]
		b0 := b[p*n+j]
		b1 := b[p*n+j+1]
		b2 := b[p*n+j+2]
		b3 := b[p*n+j+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}

	// Write accumulators back to C (accumulate — C was zeroed before)
	c[i*n+j] += c00
	c[i*n+j+1] += c01
	c[i*n+j+2] += c02
	c[i*n+j+3] += c03
	c[(i+1)*n+j] += c10
	c[(i+1)*n+j+1] += c11
	c[(i+1)*n+j+2] += c12
	c[(i+1)*n+j+3] += c13
	c[(i+2)*n+j] += c20
	c[(i+2)*n+j+1] += c21
	c[(i+2)*n+j+2] += c22
	c[(i+2)*n+j+3] += c23
	c[(i+3)*n+j] += c30
	c[(i+3)*n+j+1] += c31
	c[(i+3)*n+j+2] += c32
	c[(i+3)*n+j+3] += c33
}

// matMulMicroMx1 handles a Mx1 column remainder for rows i..iEnd.
//
//mem:hot
//mem:nogc
func matMulMicroMx1(a, b, c []float32, k, n, i, iEnd, j, p0, pEnd int) {
	for ii := i; ii < iEnd; ii++ {
		var acc float32
		rowA := a[ii*k : ii*k+k]
		for p := p0; p < pEnd; p++ {
			acc += rowA[p] * b[p*n+j]
		}
		c[ii*n+j] += acc
	}
}

// matMulMicro1x1 handles a single element remainder.
//
//mem:hot
//mem:nogc
func matMulMicro1x1(a, b, c []float32, k, n, i, j, p0, pEnd int) {
	var acc float32
	rowA := a[i*k : i*k+k]
	for p := p0; p < pEnd; p++ {
		acc += rowA[p] * b[p*n+j]
	}
	c[i*n+j] += acc
}
