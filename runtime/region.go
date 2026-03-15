// Package runtime provides zero-allocation typed memory access via Region
package runtime

import (
	"fmt"
	"math"
	"unsafe"
)

// Region provides typed, zero-allocation access to region memory
// All accessor methods use unsafe pointer arithmetic for direct memory access
type Region struct {
	name   string         // Region name (for error messages)
	base   unsafe.Pointer // Base pointer to region memory
	size   int64          // Region size in bytes
	mode   RegionMode     // Region mode (stream, ring, etc.)
	layout *RegionLayout  // Field layout (from Phase 2)
}

// Name returns the region name
func (v *Region) Name() string {
	return v.name
}

// Size returns the region size in bytes
func (v *Region) Size() int64 {
	return v.size
}

// Mode returns the region mode
func (v *Region) Mode() RegionMode {
	return v.mode
}

// Layout returns the region layout
func (v *Region) Layout() *RegionLayout {
	return v.layout
}

// Base returns the base pointer (for advanced usage)
func (v *Region) Base() unsafe.Pointer {
	return v.base
}

// ============================================================================
// TYPED GETTERS (Zero-Allocation Hot Path)
// ============================================================================

// U8 reads a uint8 field
// Zero allocations - direct memory read via unsafe pointer
//
//mem:hot
//mem:nogc
func (v *Region) U8(field string) (uint8, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	// Validate type
	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU8 {
		return 0, fmt.Errorf("field %s is not u8 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*uint8)(ptr), nil
}

// U16 reads a uint16 field
//
//mem:hot
//mem:nogc
func (v *Region) U16(field string) (uint16, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU16 {
		return 0, fmt.Errorf("field %s is not u16 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*uint16)(ptr), nil
}

// U32 reads a uint32 field
//
//mem:hot
//mem:nogc
func (v *Region) U32(field string) (uint32, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU32 {
		return 0, fmt.Errorf("field %s is not u32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*uint32)(ptr), nil
}

// U64 reads a uint64 field
//
//mem:hot
//mem:nogc
func (v *Region) U64(field string) (uint64, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU64 {
		return 0, fmt.Errorf("field %s is not u64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*uint64)(ptr), nil
}

// I8 reads an int8 field
//
//mem:hot
//mem:nogc
func (v *Region) I8(field string) (int8, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI8 {
		return 0, fmt.Errorf("field %s is not i8 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*int8)(ptr), nil
}

// I16 reads an int16 field
//
//mem:hot
//mem:nogc
func (v *Region) I16(field string) (int16, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI16 {
		return 0, fmt.Errorf("field %s is not i16 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*int16)(ptr), nil
}

// I32 reads an int32 field
//
//mem:hot
//mem:nogc
func (v *Region) I32(field string) (int32, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI32 {
		return 0, fmt.Errorf("field %s is not i32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*int32)(ptr), nil
}

// I64 reads an int64 field
//
//mem:hot
//mem:nogc
func (v *Region) I64(field string) (int64, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI64 {
		return 0, fmt.Errorf("field %s is not i64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*int64)(ptr), nil
}

// F32 reads a float32 field
//
//mem:hot
//mem:nogc
func (v *Region) F32(field string) (float32, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeF32 {
		return 0, fmt.Errorf("field %s is not f32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*float32)(ptr), nil
}

// F64 reads a float64 field
//
//mem:hot
//mem:nogc
func (v *Region) F64(field string) (float64, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeF64 {
		return 0, fmt.Errorf("field %s is not f64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	return *(*float64)(ptr), nil
}

// Bool reads a boolean field (stored as uint8: 0=false, 1=true)
//
//mem:hot
//mem:nogc
func (v *Region) Bool(field string) (bool, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return false, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeBool {
		return false, fmt.Errorf("field %s is not bool (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	val := *(*uint8)(ptr)
	return val != 0, nil
}

// ============================================================================
// TYPED SETTERS (Zero-Allocation Hot Path)
// ============================================================================

// SetU8 writes a uint8 field
//
//mem:hot
//mem:nogc
func (v *Region) SetU8(field string, value uint8) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU8 {
		return fmt.Errorf("field %s is not u8 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*uint8)(ptr) = value
	return nil
}

// SetU16 writes a uint16 field
//
//mem:hot
//mem:nogc
func (v *Region) SetU16(field string, value uint16) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU16 {
		return fmt.Errorf("field %s is not u16 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*uint16)(ptr) = value
	return nil
}

// SetU32 writes a uint32 field
//
//mem:hot
//mem:nogc
func (v *Region) SetU32(field string, value uint32) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU32 {
		return fmt.Errorf("field %s is not u32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*uint32)(ptr) = value
	return nil
}

// SetU64 writes a uint64 field
//
//mem:hot
//mem:nogc
func (v *Region) SetU64(field string, value uint64) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeU64 {
		return fmt.Errorf("field %s is not u64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*uint64)(ptr) = value
	return nil
}

// SetI8 writes an int8 field
//
//mem:hot
//mem:nogc
func (v *Region) SetI8(field string, value int8) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI8 {
		return fmt.Errorf("field %s is not i8 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*int8)(ptr) = value
	return nil
}

// SetI16 writes an int16 field
//
//mem:hot
//mem:nogc
func (v *Region) SetI16(field string, value int16) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI16 {
		return fmt.Errorf("field %s is not i16 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*int16)(ptr) = value
	return nil
}

// SetI32 writes an int32 field
//
//mem:hot
//mem:nogc
func (v *Region) SetI32(field string, value int32) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI32 {
		return fmt.Errorf("field %s is not i32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*int32)(ptr) = value
	return nil
}

// SetI64 writes an int64 field
//
//mem:hot
//mem:nogc
func (v *Region) SetI64(field string, value int64) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeI64 {
		return fmt.Errorf("field %s is not i64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*int64)(ptr) = value
	return nil
}

// SetF32 writes a float32 field
//
//mem:hot
//mem:nogc
func (v *Region) SetF32(field string, value float32) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeF32 {
		return fmt.Errorf("field %s is not f32 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*float32)(ptr) = value
	return nil
}

// SetF64 writes a float64 field
//
//mem:hot
//mem:nogc
func (v *Region) SetF64(field string, value float64) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeF64 {
		return fmt.Errorf("field %s is not f64 (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	*(*float64)(ptr) = value
	return nil
}

// SetBool writes a boolean field (stored as uint8: 0=false, 1=true)
//
//mem:hot
//mem:nogc
func (v *Region) SetBool(field string, value bool) error {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeBool {
		return fmt.Errorf("field %s is not bool (got %s)", field, ftype.String())
	}

	ptr := unsafe.Add(v.base, offset)
	var val uint8
	if value {
		val = 1
	}
	*(*uint8)(ptr) = val
	return nil
}

// ============================================================================
// VECTOR FIELD ACCESS (VecF32)
// ============================================================================

// VecF32Ptr returns a pointer and length for a vecf32 field
// This enables zero-copy access to float32 arrays without allocation
//
//mem:hot
//mem:nogc
func (v *Region) VecF32Ptr(field string) (unsafe.Pointer, int, error) {
	offset, err := v.layout.FieldOffset(field)
	if err != nil {
		return nil, 0, err
	}

	ftype, _ := v.layout.FieldType(field)
	if ftype != TypeVecF32 {
		return nil, 0, fmt.Errorf("field %s is not vecf32 (got %s)", field, ftype.String())
	}

	size, _ := v.layout.FieldSize(field)
	length := size / 4 // float32 is 4 bytes

	ptr := unsafe.Add(v.base, offset)
	return ptr, length, nil
}

// VecF32Read reads a float32 array (allocates! use VecF32Ptr in hot paths)
//
//mem:allow(convenience) - Use VecF32Ptr in hot paths instead
func (v *Region) VecF32Read(field string) ([]float32, error) {
	ptr, length, err := v.VecF32Ptr(field)
	if err != nil {
		return nil, err
	}

	// Allocate slice
	result := make([]float32, length)

	// Copy from memory
	for i := 0; i < length; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		result[i] = *(*float32)(elemPtr)
	}

	return result, nil
}

// VecF32Write writes a float32 array
// WARNING: Allocates if src is larger than field capacity
//
//mem:hot (when src fits in field)
func (v *Region) VecF32Write(field string, src []float32) error {
	ptr, capacity, err := v.VecF32Ptr(field)
	if err != nil {
		return err
	}

	if len(src) > capacity {
		return fmt.Errorf("source array too large: %d > %d", len(src), capacity)
	}

	// Copy into memory
	for i := 0; i < len(src); i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		*(*float32)(elemPtr) = src[i]
	}

	return nil
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

// HasField checks if a field exists
func (v *Region) HasField(field string) bool {
	return v.layout.HasField(field)
}

// FieldType returns the type of a field
func (v *Region) FieldType(field string) (FieldType, error) {
	return v.layout.FieldType(field)
}

// FieldOffset returns the offset of a field
func (v *Region) FieldOffset(field string) (int, error) {
	return v.layout.FieldOffset(field)
}

// FieldSize returns the size of a field
func (v *Region) FieldSize(field string) (int, error) {
	return v.layout.FieldSize(field)
}

// Validate checks view consistency
func (v *Region) Validate() error {
	if v.base == nil {
		return fmt.Errorf("region view has null base pointer")
	}

	if v.size <= 0 {
		return fmt.Errorf("region view has invalid size: %d", v.size)
	}

	if v.layout == nil {
		return fmt.Errorf("region view has nil layout")
	}

	// Validate layout
	if err := v.layout.Validate(); err != nil {
		return fmt.Errorf("layout validation failed: %w", err)
	}

	return nil
}

// Zero clears all memory in the region
func (v *Region) Zero() {
	for i := int64(0); i < v.size; i++ {
		ptr := unsafe.Add(v.base, i)
		*(*uint8)(ptr) = 0
	}
}

// DumpHex returns a hex dump of the region (for debugging)
//
//mem:allow(debug_only) - Only for debugging
func (v *Region) DumpHex(maxBytes int) string {
	if maxBytes <= 0 || maxBytes > int(v.size) {
		maxBytes = int(v.size)
	}

	result := fmt.Sprintf("Region %s (base=%p, size=%d bytes):\n", v.name, v.base, v.size)

	for i := 0; i < maxBytes; i++ {
		if i%16 == 0 {
			result += fmt.Sprintf("%04x: ", i)
		}

		ptr := unsafe.Add(v.base, i)
		val := *(*uint8)(ptr)
		result += fmt.Sprintf("%02x ", val)

		if (i+1)%16 == 0 {
			result += "\n"
		}
	}

	if maxBytes%16 != 0 {
		result += "\n"
	}

	return result
}

// DumpFields returns a human-readable dump of all fields
func (v *Region) DumpFields() string {
	result := fmt.Sprintf("Region %s fields:\n", v.name)

	for _, field := range v.layout.Fields {
		var valueStr string

		switch field.Type {
		case TypeU8:
			val, _ := v.U8(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeU16:
			val, _ := v.U16(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeU32:
			val, _ := v.U32(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeU64:
			val, _ := v.U64(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeI8:
			val, _ := v.I8(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeI16:
			val, _ := v.I16(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeI32:
			val, _ := v.I32(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeI64:
			val, _ := v.I64(field.Name)
			valueStr = fmt.Sprintf("%d", val)
		case TypeF32:
			val, _ := v.F32(field.Name)
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				valueStr = fmt.Sprintf("%v", val)
			} else {
				valueStr = fmt.Sprintf("%.6f", val)
			}
		case TypeF64:
			val, _ := v.F64(field.Name)
			if math.IsNaN(val) || math.IsInf(val, 0) {
				valueStr = fmt.Sprintf("%v", val)
			} else {
				valueStr = fmt.Sprintf("%.6f", val)
			}
		case TypeBool:
			val, _ := v.Bool(field.Name)
			valueStr = fmt.Sprintf("%t", val)
		case TypeVecF32:
			ptr, length, _ := v.VecF32Ptr(field.Name)
			valueStr = fmt.Sprintf("[vecf32 ptr=%p len=%d]", ptr, length)
		default:
			valueStr = "unknown"
		}

		result += fmt.Sprintf("  [%3d +%2d] %-12s %-8s = %s\n",
			field.Offset, field.Size, field.Name, field.Type.String(), valueStr)
	}

	return result
}
