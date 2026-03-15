package runtime

import (
"testing"
)

func BenchmarkArenaCreation(b *testing.B) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 1024,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
					{Name: "b", Type: TypeU64, Offset: 4, Size: 8},
					{Name: "c", Type: TypeF64, Offset: 12, Size: 8},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		arena, err := NewArena(specs, layouts)
		if err != nil {
			b.Fatal(err)
		}
		_ = arena
	}
}

//mem:nogc
func BenchmarkArenaView(b *testing.B) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 1024,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, _ := NewArena(specs, layouts)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		view, err := arena.Region("test")
		if err != nil {
			b.Fatal(err)
		}
		_ = view
	}
}

//mem:nogc
func BenchmarkArenaMustView(b *testing.B) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 1024,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, _ := NewArena(specs, layouts)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		view := arena.MustRegion("test")
		_ = view
	}
}

//mem:nogc
func BenchmarkArenaHasView(b *testing.B) {
	specs := []*RegionSpec{
		{
			Name:   "test",
			Size:   1024,
			Mode:   ModeStream,
			Header: &Header{},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, _ := NewArena(specs, layouts)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		exists := arena.HasRegion("test")
		_ = exists
	}
}

func BenchmarkArenaMultipleRegions(b *testing.B) {
	specs := []*RegionSpec{
		{Name: "region1", Size: 256, Mode: ModeStream, Header: &Header{}},
		{Name: "region2", Size: 512, Mode: ModeRing, Header: &Header{}},
		{Name: "region3", Size: 1024, Mode: ModeSlab, Header: &Header{}},
		{Name: "region4", Size: 128, Mode: ModeStream, Header: &Header{}},
	}

	layouts := NewLayoutTable()
	for _, spec := range specs {
		layout := ComputeLayout(spec)
		layouts.Add(layout)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		arena, err := NewArena(specs, layouts)
		if err != nil {
			b.Fatal(err)
		}
		_ = arena
	}
}
