package runtime

import (
"testing"
)

func createBenchArena(fields []*HeaderField, size int64) *Region {
	specs := []*RegionSpec{
		{
			Name:   "bench",
			Size:   size,
			Mode:   ModeStream,
			Header: &Header{Fields: fields},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, _ := NewArena(specs, layouts)
	return arena.MustRegion("bench")
}

//mem:nogc
func BenchmarkRegionViewU32Read(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)
	view.SetU32("val", 42)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val, err := view.U32("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = val
	}
}

//mem:nogc
func BenchmarkRegionViewU32Write(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.SetU32("val", uint32(i))
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewU64Read(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
	}
	view := createBenchArena(fields, 8)
	view.SetU64("val", 1698012345000000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val, err := view.U64("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = val
	}
}

//mem:nogc
func BenchmarkRegionViewU64Write(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
	}
	view := createBenchArena(fields, 8)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.SetU64("val", uint64(i))
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewF32Read(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeF32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)
	view.SetF32("val", 3.14159)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val, err := view.F32("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = val
	}
}

//mem:nogc
func BenchmarkRegionViewF32Write(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeF32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.SetF32("val", float32(i)*0.1)
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewF64Read(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeF64, Offset: 0, Size: 8},
	}
	view := createBenchArena(fields, 8)
	view.SetF64("val", 3.141592653589793)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val, err := view.F64("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = val
	}
}

//mem:nogc
func BenchmarkRegionViewF64Write(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeF64, Offset: 0, Size: 8},
	}
	view := createBenchArena(fields, 8)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.SetF64("val", float64(i)*0.123456)
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewBoolRead(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeBool, Offset: 0, Size: 1},
	}
	view := createBenchArena(fields, 1)
	view.SetBool("val", true)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val, err := view.Bool("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = val
	}
}

//mem:nogc
func BenchmarkRegionViewBoolWrite(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeBool, Offset: 0, Size: 1},
	}
	view := createBenchArena(fields, 1)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.SetBool("val", i%2 == 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewVecF32Ptr(b *testing.B) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 400},
	}
	view := createBenchArena(fields, 400)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ptr, length, err := view.VecF32Ptr("samples")
		if err != nil {
			b.Fatal(err)
		}
		_ = ptr
		_ = length
	}
}

func BenchmarkRegionViewVecF32Read(b *testing.B) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 40},
	}
	view := createBenchArena(fields, 40)

	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	view.VecF32Write("samples", src)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		result, err := view.VecF32Read("samples")
		if err != nil {
			b.Fatal(err)
		}
		_ = result
	}
}

//mem:nogc
func BenchmarkRegionViewVecF32Write(b *testing.B) {
	fields := []*HeaderField{
		{Name: "samples", Type: TypeVecF32, Offset: 0, Size: 40},
	}
	view := createBenchArena(fields, 40)

	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := view.VecF32Write("samples", src)
		if err != nil {
			b.Fatal(err)
		}
	}
}

//mem:nogc
func BenchmarkRegionViewRealisticWorkload(b *testing.B) {
	fields := []*HeaderField{
		{Name: "count", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "sum", Type: TypeU64, Offset: 4, Size: 8},
		{Name: "avg", Type: TypeF64, Offset: 12, Size: 8},
	}
	view := createBenchArena(fields, 20)

	view.SetU32("count", 10)
	view.SetU64("sum", 1000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		count, _ := view.U32("count")
		sum, _ := view.U64("sum")
		avg := float64(sum) / float64(count)
		view.SetF64("avg", avg)
	}
}

//mem:nogc
func BenchmarkRegionViewMultiFieldAccess(b *testing.B) {
	fields := []*HeaderField{
		{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
		{Name: "b", Type: TypeU32, Offset: 4, Size: 4},
		{Name: "c", Type: TypeU32, Offset: 8, Size: 4},
		{Name: "d", Type: TypeU32, Offset: 12, Size: 4},
	}
	view := createBenchArena(fields, 16)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		a, _ := view.U32("a")
		b, _ := view.U32("b")
		c, _ := view.U32("c")
		d, _ := view.U32("d")
		result := a + b + c + d
		view.SetU32("a", result)
	}
}

//mem:nogc
func BenchmarkRegionViewHasField(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		exists := view.HasField("val")
		_ = exists
	}
}

//mem:nogc
func BenchmarkRegionViewFieldOffset(b *testing.B) {
	fields := []*HeaderField{
		{Name: "val", Type: TypeU32, Offset: 0, Size: 4},
	}
	view := createBenchArena(fields, 4)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		offset, err := view.FieldOffset("val")
		if err != nil {
			b.Fatal(err)
		}
		_ = offset
	}
}

//mem:nogc
func BenchmarkRegionViewAllTypes(b *testing.B) {
	fields := []*HeaderField{
		{Name: "u8", Type: TypeU8, Offset: 0, Size: 1},
		{Name: "u16", Type: TypeU16, Offset: 2, Size: 2},
		{Name: "u32", Type: TypeU32, Offset: 4, Size: 4},
		{Name: "u64", Type: TypeU64, Offset: 8, Size: 8},
		{Name: "i8", Type: TypeI8, Offset: 16, Size: 1},
		{Name: "i16", Type: TypeI16, Offset: 18, Size: 2},
		{Name: "i32", Type: TypeI32, Offset: 20, Size: 4},
		{Name: "i64", Type: TypeI64, Offset: 24, Size: 8},
		{Name: "f32", Type: TypeF32, Offset: 32, Size: 4},
		{Name: "f64", Type: TypeF64, Offset: 36, Size: 8},
		{Name: "bool", Type: TypeBool, Offset: 44, Size: 1},
	}
	view := createBenchArena(fields, 48)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		view.U8("u8")
		view.U16("u16")
		view.U32("u32")
		view.U64("u64")
		view.I8("i8")
		view.I16("i16")
		view.I32("i32")
		view.I64("i64")
		view.F32("f32")
		view.F64("f64")
		view.Bool("bool")

		view.SetU8("u8", uint8(i))
		view.SetU16("u16", uint16(i))
		view.SetU32("u32", uint32(i))
		view.SetU64("u64", uint64(i))
		view.SetI8("i8", int8(i))
		view.SetI16("i16", int16(i))
		view.SetI32("i32", int32(i))
		view.SetI64("i64", int64(i))
		view.SetF32("f32", float32(i))
		view.SetF64("f64", float64(i))
		view.SetBool("bool", i%2 == 0)
	}
}
