// Package math provides zero-alloc mathematical functions for MemPipe.
//
// All functions are package-level with concrete types — no interface{} boxing.
// The MathModule wrapper exists solely for registry discovery.
package math

import (
	"math"

	"github.com/GoMemPipe/mempipe/module"
)

// --- Module wrapper for registry ---

// MathModule registers the math package with the module registry.
type MathModule struct{}

func NewMathModule() *MathModule   { return &MathModule{} }
func (m *MathModule) Name() string { return "math" }
func (m *MathModule) Init() error  { return nil }

func init() {
	module.Register(NewMathModule())
}

// --- Package-level typed functions (zero-alloc) ---

// Sqrt returns the square root of x.
func Sqrt(x float64) float64 { return math.Sqrt(x) }

// Pow returns base raised to the power of exp.
func Pow(base, exp float64) float64 { return math.Pow(base, exp) }

// Sin returns the sine of x (radians).
func Sin(x float64) float64 { return math.Sin(x) }

// Cos returns the cosine of x (radians).
func Cos(x float64) float64 { return math.Cos(x) }

// Tan returns the tangent of x (radians).
func Tan(x float64) float64 { return math.Tan(x) }

// Floor returns the greatest integer ≤ x.
func Floor(x float64) float64 { return math.Floor(x) }

// Ceil returns the smallest integer ≥ x.
func Ceil(x float64) float64 { return math.Ceil(x) }

// Round returns x rounded to the nearest integer.
func Round(x float64) float64 { return math.Round(x) }

// Log returns the natural logarithm of x.
func Log(x float64) float64 { return math.Log(x) }

// Log10 returns the base-10 logarithm of x.
func Log10(x float64) float64 { return math.Log10(x) }

// Exp returns e raised to the power x.
func Exp(x float64) float64 { return math.Exp(x) }

// Factorial returns n! for non-negative n.
func Factorial(n int64) int64 {
	if n <= 1 {
		return 1
	}
	result := int64(1)
	for i := int64(2); i <= n; i++ {
		result *= i
	}
	return result
}

// GCD returns the greatest common divisor of a and b (Euclidean algorithm).
func GCD(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// LCM returns the least common multiple of a and b.
func LCM(a, b int64) int64 {
	g := GCD(a, b)
	if g == 0 {
		return 0
	}
	return (a * b) / g
}

// Abs returns the absolute value of x.
func Abs(x float64) float64 { return math.Abs(x) }

// Min returns the smaller of x or y.
func Min(x, y float64) float64 { return math.Min(x, y) }

// Max returns the larger of x or y.
func Max(x, y float64) float64 { return math.Max(x, y) }

// Clamp returns x clamped to the range [lo, hi].
func Clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
