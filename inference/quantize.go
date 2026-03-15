package inference

import (
	"errors"
	"math"
)

// ────────────────────────────────────────────────────────────────────────────
// Quantization methods
// ────────────────────────────────────────────────────────────────────────────

// QuantizeSymmetric performs symmetric INT8 quantization on a float32 tensor.
// Returns the computed scale factor. Zero-alloc on pre-allocated tensors.
//
//mem:hot
func QuantizeSymmetric(src *Tensor, dst *Tensor) (scale float32, err error) {
	if src.dtype != Float32 {
		return 0, errors.New("QuantizeSymmetric: src must be float32")
	}
	if dst.dtype != Int8 {
		return 0, errors.New("QuantizeSymmetric: dst must be int8")
	}
	if src.NumElements() != dst.NumElements() {
		return 0, errors.New("QuantizeSymmetric: element count mismatch")
	}

	srcData := src.Float32s()

	// Find abs max
	var absMax float32
	for _, v := range srcData {
		av := v
		if av < 0 {
			av = -av
		}
		if av > absMax {
			absMax = av
		}
	}

	scale = absMax / 127.0
	if scale == 0 {
		scale = 1.0
	}
	invScale := 1.0 / scale

	dstData := dst.Int8s()
	for i, v := range srcData {
		q := int32(math.RoundToEven(float64(v * invScale)))
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		dstData[i] = int8(q)
	}

	return scale, nil
}

// QuantizeAsymmetric performs asymmetric INT8 quantization.
// Returns scale and zero-point.
//
//mem:hot
func QuantizeAsymmetric(src *Tensor, dst *Tensor) (scale float32, zeroPoint int32, err error) {
	if src.dtype != Float32 {
		return 0, 0, errors.New("QuantizeAsymmetric: src must be float32")
	}
	if dst.dtype != Int8 {
		return 0, 0, errors.New("QuantizeAsymmetric: dst must be int8")
	}
	if src.NumElements() != dst.NumElements() {
		return 0, 0, errors.New("QuantizeAsymmetric: element count mismatch")
	}

	srcData := src.Float32s()

	// Find min/max
	minVal := srcData[0]
	maxVal := srcData[0]
	for _, v := range srcData[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute scale and zero-point
	scale = (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}
	zeroPoint = int32(math.RoundToEven(float64(-minVal / scale)))
	if zeroPoint > 127 {
		zeroPoint = 127
	} else if zeroPoint < -128 {
		zeroPoint = -128
	}

	invScale := 1.0 / scale
	dstData := dst.Int8s()
	for i, v := range srcData {
		q := int32(math.RoundToEven(float64(v*invScale))) + zeroPoint
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		dstData[i] = int8(q)
	}

	return scale, zeroPoint, nil
}

// DequantizeInt8ToFloat32 converts an INT8 tensor to float32.
//
//mem:hot
func DequantizeInt8ToFloat32(src *Tensor, dst *Tensor, scale float32, zeroPoint int32) error {
	if src.dtype != Int8 {
		return errors.New("DequantizeInt8ToFloat32: src must be int8")
	}
	if dst.dtype != Float32 {
		return errors.New("DequantizeInt8ToFloat32: dst must be float32")
	}
	if src.NumElements() != dst.NumElements() {
		return errors.New("DequantizeInt8ToFloat32: element count mismatch")
	}

	srcData := src.Int8s()
	dstData := dst.Float32s()
	for i, v := range srcData {
		dstData[i] = float32(int32(v)-zeroPoint) * scale
	}
	return nil
}

// ────────────────────────────────────────────────────────────────────────────
// FP16 support (manual bit manipulation — no external dependency)
// ────────────────────────────────────────────────────────────────────────────

// F32ToF16Bits converts a float32 to IEEE 754 half-precision bits (uint16).
func F32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xFF) - 127
	mant := bits & 0x007FFFFF

	switch {
	case exp > 15:
		// Overflow → Inf
		return uint16(sign | 0x7C00)
	case exp > -15:
		// Normalized
		return uint16(sign | uint32(exp+15)<<10 | (mant >> 13))
	case exp > -25:
		// Denormalized
		mant |= 0x00800000
		shift := uint(-14 - exp)
		return uint16(sign | (mant >> (shift + 13)))
	default:
		return uint16(sign) // too small → zero
	}
}

// F16BitsToF32 converts IEEE 754 half-precision bits to float32.
func F16BitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1F)
	mant := uint32(h & 0x03FF)

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// Denormalized
		for mant&0x0400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
		return math.Float32frombits(sign | uint32(exp+127-15)<<23 | mant<<13)
	case exp == 0x1F:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000) // ±Inf
		}
		return math.Float32frombits(sign | 0x7FC00000) // NaN
	default:
		return math.Float32frombits(sign | uint32(exp+127-15)<<23 | mant<<13)
	}
}

// F32ToF16 converts a float32 tensor to float16 (stored as uint16 pairs in dst).
//
//mem:hot
func F32ToF16(src *Tensor, dst *Tensor) error {
	if src.dtype != Float32 {
		return errors.New("F32ToF16: src must be float32")
	}
	if dst.dtype != Float16 {
		return errors.New("F32ToF16: dst must be float16")
	}
	if src.NumElements() != dst.NumElements() {
		return errors.New("F32ToF16: element count mismatch")
	}

	srcData := src.Float32s()
	// dst is Float16 — access raw bytes as uint16 pairs
	n := dst.NumElements()
	dstBytes := dst.Bytes()
	for i := 0; i < n; i++ {
		h := F32ToF16Bits(srcData[i])
		dstBytes[i*2] = byte(h)
		dstBytes[i*2+1] = byte(h >> 8)
	}
	return nil
}

// F16ToF32 converts a float16 tensor to float32.
//
//mem:hot
func F16ToF32(src *Tensor, dst *Tensor) error {
	if src.dtype != Float16 {
		return errors.New("F16ToF32: src must be float16")
	}
	if dst.dtype != Float32 {
		return errors.New("F16ToF32: dst must be float32")
	}
	if src.NumElements() != dst.NumElements() {
		return errors.New("F16ToF32: element count mismatch")
	}

	srcBytes := src.Bytes()
	dstData := dst.Float32s()
	n := src.NumElements()
	for i := 0; i < n; i++ {
		h := uint16(srcBytes[i*2]) | uint16(srcBytes[i*2+1])<<8
		dstData[i] = F16BitsToF32(h)
	}
	return nil
}
