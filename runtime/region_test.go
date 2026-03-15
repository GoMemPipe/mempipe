package runtime

import (
	"strings"
	"testing"
	"unsafe"
)

// Helper to create a test arena with one region
func createTestArena(t *testing.T, fields []*HeaderField, size int64) (*RegionArena, *Region) {
	specs := []*RegionSpec{
		{
			Name:   "test",
			Size:   size,
			Mode:   ModeStream,
			Header: &Header{Fields: fields},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	view := arena.MustRegion("test")
	return arena, view
}

func TestRegionViewU32(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "b", Type: TypeU32, Offset: 4, Size: 4},
	}

	_, view := createTestArena(t, fields, 8)

	if err := view.SetU32("a", 42); err != nil {
		t.Errorf("SetU32 failed: %v", err)
	}
	if err := view.SetU32("b", 100); err != nil {
		t.Errorf("SetU32 failed: %v", err)
	}

	val, err := view.U32("a")
	if err != nil {
		t.Errorf("U32 failed: %v", err)
	}
	if val != 42 {
		t.Errorf("U32(a): got %d, want 42", val)
	}

	val, err = view.U32("b")
	if err != nil {
		t.Errorf("U32 failed: %v", err)
	}
	if val != 100 {
		t.Errorf("U32(b): got %d, want 100", val)
	}
}

func TestRegionViewU64(t *testing.T) {
	fields := []*HeaderField{
		{Name: "timestamp", Type: TypeU64, Offset: 0, Size: 8},
	}

	_, view := createTestArena(t, fields, 8)

	testVal := uint64(1698012345000000)
	if err := view.SetU64("timestamp", testVal); err != nil {
		t.Errorf("SetU64 failed: %v", err)
	}

	val, err := view.U64("timestamp")
	if err != nil {
		t.Errorf("U64 failed: %v", err)
	}
	if val != testVal {
		t.Errorf("U64: got %d, want %d", val, testVal)
	}
}

func TestRegionViewF32(t *testing.T) {
	fields := []*HeaderField{
		{Name: "temp", Type: TypeF32, Offset: 0, Size: 4},
	}

	_, view := createTestArena(t, fields, 4)

	testVal := float32(23.5)
	if err := view.SetF32("temp", testVal); err != nil {
		t.Errorf("SetF32 failed: %v", err)
	}

	val, err := view.F32("temp")
	if err != nil {
		t.Errorf("F32 failed: %v", err)
	}
	if val != testVal {
		t.Errorf("F32: got %f, want %f", val, testVal)
	}
}

func TestRegionViewF64(t *testing.T) {
	fields := []*HeaderField{
		{Name: "avg", Type: TypeF64, Offset: 0, Size: 8},
	}

	_, view := createTestArena(t, fields, 8)

	testVal := 42.123456789
	if err := view.SetF64("avg", testVal); err != nil {
		t.Errorf("SetF64 failed: %v", err)
	}

	val, err := view.F64("avg")
	if err != nil {
		t.Errorf("F64 failed: %v", err)
	}
	if val != testVal {
		t.Errorf("F64: got %f, want %f", val, testVal)
	}
}

func TestRegionViewBool(t *testing.T) {
	fields := []*HeaderField{
		{Name: "active", Type: TypeBool, Offset: 0, Size: 1},
		{Name: "enabled", Type: TypeBool, Offset: 1, Size: 1},
	}

	_, view := createTestArena(t, fields, 2)

	if err := view.SetBool("active", true); err != nil {
		t.Errorf("SetBool failed: %v", err)
	}
	if err := view.SetBool("enabled", false); err != nil {
		t.Errorf("SetBool failed: %v", err)
	}

	val, err := view.Bool("active")
	if err != nil {
		t.Errorf("Bool failed: %v", err)
	}
	if !val {
		t.Error("Bool(active): got false, want true")
	}

	val, err = view.Bool("enabled")
	if err != nil {
		t.Errorf("Bool failed: %v", err)
	}
	if val {
		t.Error("Bool(enabled): got true, want false")
	}
}

func TestRegionViewAllIntegerTypes(t *testing.T) {
	fields := []*HeaderField{
		{Name: "u8", Type: TypeU8, Offset: 0, Size: 1},
		{Name: "u16", Type: TypeU16, Offset: 2, Size: 2},
		{Name: "u32", Type: TypeU32, Offset: 4, Size: 4},
		{Name: "u64", Type: TypeU64, Offset: 8, Size: 8},
		{Name: "i8", Type: TypeI8, Offset: 16, Size: 1},
		{Name: "i16", Type: TypeI16, Offset: 18, Size: 2},
		{Name: "i32", Type: TypeI32, Offset: 20, Size: 4},
		{Name: "i64", Type: TypeI64, Offset: 24, Size: 8},
	}

	_, view := createTestArena(t, fields, 32)

	view.SetU8("u8", 255)
	val8, _ := view.U8("u8")
	if val8 != 255 {
		t.Errorf("u8: got %d, want 255", val8)
	}

	view.SetU16("u16", 65535)
	val16, _ := view.U16("u16")
	if val16 != 65535 {
		t.Errorf("u16: got %d, want 65535", val16)
	}

	view.SetU32("u32", 4294967295)
	val32, _ := view.U32("u32")
	if val32 != 4294967295 {
		t.Errorf("u32: got %d, want 4294967295", val32)
	}

	view.SetU64("u64", 18446744073709551615)
	val64, _ := view.U64("u64")
	if val64 != 18446744073709551615 {
		t.Errorf("u64: got %d, want 18446744073709551615", val64)
	}

	view.SetI8("i8", -128)
	vali8, _ := view.I8("i8")
	if vali8 != -128 {
		t.Errorf("i8: got %d, want -128", vali8)
	}

	view.SetI16("i16", -32768)
	vali16, _ := view.I16("i16")
	if vali16 != -32768 {
		t.Errorf("i16: got %d, want -32768", vali16)
	}

	view.SetI32("i32", -2147483648)
	vali32, _ := view.I32("i32")
	if vali32 != -2147483648 {
		t.Errorf("i32: got %d, want -2147483648", vali32)
	}

	view.SetI64("i64", -9223372036854775808)
	vali64, _ := view.I64("i64")
	if vali64 != -9223372036854775808 {
		t.Errorf("i64: got %d, want -9223372036854775808", vali64)
	}
}

func TestRegionViewTypeMismatch(t *testing.T) {
	fields := []*HeaderField{
		{Name: "u32_field", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "f64_field", Type: TypeF64, Offset: 8, Size: 8},
	}

	_, view := createTestArena(t, fields, 16)

	_, err := view.F64("u32_field")
	if err == nil {
		t.Error("Reading u32 field as f64 should fail")
	}

	_, err = view.U32("f64_field")
	if err == nil {
		t.Error("Reading f64 field as u32 should fail")
	}

	err = view.SetF64("u32_field", 3.14)
	if err == nil {
		t.Error("Setting u32 field as f64 should fail")
	}
}

func TestRegionViewFieldNotFound(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
	}

	_, view := createTestArena(t, fields, 4)

	_, err := view.U32("nonexistent")
	if err == nil {
		t.Error("Reading non-existent field should fail")
	}

	err = view.SetU32("nonexistent", 42)
	if err == nil {
		t.Error("Setting non-existent field should fail")
	}
}

func TestRegionViewVecF32Ptr(t *testing.T) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 40},
	}

	_, view := createTestArena(t, fields, 40)

	ptr, length, err := view.VecF32Ptr("samples")
	if err != nil {
		t.Errorf("VecF32Ptr failed: %v", err)
	}

	if length != 10 {
		t.Errorf("VecF32Ptr length: got %d, want 10", length)
	}

	if ptr == nil {
		t.Error("VecF32Ptr returned null pointer")
	}

	ptrVal := uintptr(ptr)
	baseVal := uintptr(view.Base())
	if ptrVal < baseVal || ptrVal >= baseVal+uintptr(view.Size()) {
		t.Error("VecF32Ptr returned pointer outside view bounds")
	}
}

func TestRegionViewVecF32Write(t *testing.T) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 20},
	}

	_, view := createTestArena(t, fields, 20)

	src := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	err := view.VecF32Write("samples", src)
	if err != nil {
		t.Errorf("VecF32Write failed: %v", err)
	}

	ptr, length, _ := view.VecF32Ptr("samples")
	for i := 0; i < length; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		val := *(*float32)(elemPtr)
		if val != src[i] {
			t.Errorf("VecF32 element %d: got %f, want %f", i, val, src[i])
		}
	}
}

func TestRegionViewVecF32Read(t *testing.T) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 20},
	}

	_, view := createTestArena(t, fields, 20)

	src := []float32{1.1, 2.2, 3.3, 4.4, 5.5}
	view.VecF32Write("samples", src)

	result, err := view.VecF32Read("samples")
	if err != nil {
		t.Errorf("VecF32Read failed: %v", err)
	}

	if len(result) != len(src) {
		t.Errorf("VecF32Read length: got %d, want %d", len(result), len(src))
	}

	for i := 0; i < len(src); i++ {
		if result[i] != src[i] {
			t.Errorf("VecF32Read element %d: got %f, want %f", i, result[i], src[i])
		}
	}
}

func TestRegionViewVecF32WriteTooLarge(t *testing.T) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 12},
	}

	_, view := createTestArena(t, fields, 12)

	src := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	err := view.VecF32Write("samples", src)
	if err == nil {
		t.Error("VecF32Write with too large array should fail")
	}
}

func TestRegionViewHasField(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
	}

	_, view := createTestArena(t, fields, 4)

	if !view.HasField("a") {
		t.Error("HasField(a) should be true")
	}

	if view.HasField("nonexistent") {
		t.Error("HasField(nonexistent) should be false")
	}
}

func TestRegionViewFieldType(t *testing.T) {
	fields := []*HeaderField{
		{Name: "u32_field", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "f64_field", Type: TypeF64, Offset: 8, Size: 8},
	}

	_, view := createTestArena(t, fields, 16)

	ftype, err := view.FieldType("u32_field")
	if err != nil {
		t.Errorf("FieldType failed: %v", err)
	}
	if ftype != TypeU32 {
		t.Errorf("FieldType: got %v, want TypeU32", ftype)
	}

	ftype, err = view.FieldType("f64_field")
	if err != nil {
		t.Errorf("FieldType failed: %v", err)
	}
	if ftype != TypeF64 {
		t.Errorf("FieldType: got %v, want TypeF64", ftype)
	}
}

func TestRegionViewFieldOffset(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "b", Type: TypeU64, Offset: 4, Size: 8},
	}

	_, view := createTestArena(t, fields, 12)

	offset, err := view.FieldOffset("a")
	if err != nil {
		t.Errorf("FieldOffset failed: %v", err)
	}
	if offset != 0 {
		t.Errorf("FieldOffset(a): got %d, want 0", offset)
	}

	offset, err = view.FieldOffset("b")
	if err != nil {
		t.Errorf("FieldOffset failed: %v", err)
	}
	if offset != 4 {
		t.Errorf("FieldOffset(b): got %d, want 4", offset)
	}
}

func TestRegionViewFieldSize(t *testing.T) {
	fields := []*HeaderField{
		{Name: "u32_field", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "u64_field", Type: TypeU64, Offset: 4, Size: 8},
	}

	_, view := createTestArena(t, fields, 12)

	size, err := view.FieldSize("u32_field")
	if err != nil {
		t.Errorf("FieldSize failed: %v", err)
	}
	if size != 4 {
		t.Errorf("FieldSize(u32_field): got %d, want 4", size)
	}

	size, err = view.FieldSize("u64_field")
	if err != nil {
		t.Errorf("FieldSize failed: %v", err)
	}
	if size != 8 {
		t.Errorf("FieldSize(u64_field): got %d, want 8", size)
	}
}

func TestRegionViewZero(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "b", Type: TypeU32, Offset: 4, Size: 4},
	}

	_, view := createTestArena(t, fields, 8)

	view.SetU32("a", 42)
	view.SetU32("b", 100)

	view.Zero()

	val, _ := view.U32("a")
	if val != 0 {
		t.Errorf("After Zero: a = %d, want 0", val)
	}

	val, _ = view.U32("b")
	if val != 0 {
		t.Errorf("After Zero: b = %d, want 0", val)
	}
}

func TestRegionViewDumpFields(t *testing.T) {
	fields := []*HeaderField{
		{Name: "count", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "avg", Type: TypeF64, Offset: 8, Size: 8},
		{Name: "active", Type: TypeBool, Offset: 16, Size: 1},
	}

	_, view := createTestArena(t, fields, 24)

	view.SetU32("count", 42)
	view.SetF64("avg", 3.14159)
	view.SetBool("active", true)

	dump := view.DumpFields()

	if !strings.Contains(dump, "count") {
		t.Error("DumpFields missing field: count")
	}
	if !strings.Contains(dump, "avg") {
		t.Error("DumpFields missing field: avg")
	}
	if !strings.Contains(dump, "active") {
		t.Error("DumpFields missing field: active")
	}
	if !strings.Contains(dump, "42") {
		t.Error("DumpFields missing value: 42")
	}
	if !strings.Contains(dump, "true") {
		t.Error("DumpFields missing value: true")
	}
}

func TestRegionViewValidate(t *testing.T) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
	}

	_, view := createTestArena(t, fields, 4)

	if err := view.Validate(); err != nil {
		t.Errorf("Validate failed: %v", err)
	}
}

func TestRegionViewMetadata(t *testing.T) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
	}

	_, view := createTestArena(t, fields, 8)

	if view.Name() != "test" {
		t.Errorf("Name: got %s, want test", view.Name())
	}

	if view.Size() != 8 {
		t.Errorf("Size: got %d, want 8", view.Size())
	}

	if view.Mode() != ModeStream {
		t.Errorf("Mode: got %v, want ModeStream", view.Mode())
	}

	layout := view.Layout()
	if layout == nil {
		t.Error("Layout returned nil")
	}
	if layout.Name != "test" {
		t.Errorf("Layout name: got %s, want test", layout.Name)
	}

	if view.Base() == nil {
		t.Error("Base returned null pointer")
	}
}
