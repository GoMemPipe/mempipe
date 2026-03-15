//go:build !(js && wasm) && !((linux || darwin) && (amd64 || arm64))

package inference

// matMulAccel is a no-op on unsupported architectures.
// The generic path is stateless — no Init required.
type matMulAccel struct{}

// initMatMulAccel is a no-op on the generic platform.
func initMatMulAccel(acc *matMulAccel, arena *InferenceArena) error { return nil }

// EnsureWebGPU is a no-op on non-WASM platforms.
func EnsureWebGPU() {}

// IsWebGPUReady always returns false on non-WASM platforms.
func IsWebGPUReady() bool { return false }

// matMulF32 computes C = A*B for row-major float32 matrices.
// This is the portable fallback using a simple i-p-j loop order
// (A-row × B-col, unrolled over the inner dimension p).
// Zero-alloc: operates directly on arena-backed slices.
//
//mem:hot
//mem:nogc
func matMulF32(a, b, c []float32, m, k, n int) {
	for i := 0; i < m; i++ {
		rowA := a[i*k : i*k+k]
		rowC := c[i*n : i*n+n]
		// Zero the output row
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
