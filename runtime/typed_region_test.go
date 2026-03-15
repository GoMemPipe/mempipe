package runtime

import (
	"testing"
)

// ---- test struct for TypedRegion ----

type SensorData struct {
	Temperature float32 `mempipe:"field:temperature"`
	Humidity    float32 `mempipe:"field:humidity"`
	Count       uint32  `mempipe:"field:count"`
	Active      bool    `mempipe:"field:active"`
}

func makeTypedArena(t *testing.T) *RegionArena {
	t.Helper()
	var zero SensorData
	layout, err := LayoutFromStruct("sensor", zero)
	if err != nil {
		t.Fatal(err)
	}

	spec := &RegionSpec{
		Name: "sensor",
		Size: layout.Size,
		Mode: ModeStream,
		Header: &Header{
			Fields: func() []*HeaderField {
				hf := make([]*HeaderField, len(layout.Fields))
				for i, f := range layout.Fields {
					hf[i] = &HeaderField{Name: f.Name, Type: f.Type, Offset: f.Offset, Size: f.Size}
				}
				return hf
			}(),
		},
	}

	lt := NewLayoutTable()
	lt.Add(layout)
	arena, err := NewArena([]*RegionSpec{spec}, lt)
	if err != nil {
		t.Fatal(err)
	}
	return arena
}

func TestTypedRegionGetSet(t *testing.T) {
	arena := makeTypedArena(t)

	tr, err := NewTypedRegion[SensorData](arena, "sensor")
	if err != nil {
		t.Fatal(err)
	}

	// Set
	tr.Set(SensorData{
		Temperature: 23.5,
		Humidity:    60.0,
		Count:       42,
		Active:      true,
	})

	// Get
	v := tr.Get()
	if v.Temperature != 23.5 {
		t.Errorf("Temperature: got %f, want 23.5", v.Temperature)
	}
	if v.Humidity != 60.0 {
		t.Errorf("Humidity: got %f, want 60.0", v.Humidity)
	}
	if v.Count != 42 {
		t.Errorf("Count: got %d, want 42", v.Count)
	}
	if v.Active != true {
		t.Error("Active: got false, want true")
	}
}

func TestTypedRegionZeroValue(t *testing.T) {
	arena := makeTypedArena(t)

	tr, err := NewTypedRegion[SensorData](arena, "sensor")
	if err != nil {
		t.Fatal(err)
	}

	v := tr.Get()
	if v.Count != 0 || v.Active != false {
		t.Errorf("Expected zero value, got %+v", v)
	}
}

func TestTypedRegionRoundTrip(t *testing.T) {
	arena := makeTypedArena(t)

	tr, err := NewTypedRegion[SensorData](arena, "sensor")
	if err != nil {
		t.Fatal(err)
	}

	for i := uint32(0); i < 100; i++ {
		tr.Set(SensorData{Temperature: float32(i), Count: i})
		v := tr.Get()
		if v.Count != i {
			t.Fatalf("Round-trip failed at %d: got %d", i, v.Count)
		}
	}
}

func TestTypedRegionFieldAccess(t *testing.T) {
	arena := makeTypedArena(t)

	tr, err := NewTypedRegion[SensorData](arena, "sensor")
	if err != nil {
		t.Fatal(err)
	}

	tr.Set(SensorData{Temperature: 99.9, Count: 7})

	// Access via underlying Region
	r := tr.Region()
	count, err := r.U32("count")
	if err != nil {
		t.Fatal(err)
	}
	if count != 7 {
		t.Errorf("Field access: got %d, want 7", count)
	}
}

// Benchmark: TypedRegion.Get must be zero-alloc
func BenchmarkTypedRegionGet(b *testing.B) {
	arena := func() *RegionArena {
		var zero SensorData
		layout, _ := LayoutFromStruct("sensor", zero)
		spec := &RegionSpec{Name: "sensor", Size: layout.Size, Mode: ModeStream,
			Header: &Header{Fields: func() []*HeaderField {
				hf := make([]*HeaderField, len(layout.Fields))
				for i, f := range layout.Fields {
					hf[i] = &HeaderField{Name: f.Name, Type: f.Type, Offset: f.Offset, Size: f.Size}
				}
				return hf
			}()}}
		lt := NewLayoutTable()
		lt.Add(layout)
		a, _ := NewArena([]*RegionSpec{spec}, lt)
		return a
	}()

	tr, _ := NewTypedRegion[SensorData](arena, "sensor")
	tr.Set(SensorData{Temperature: 1.0, Count: 1})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := tr.Get()
		_ = v
	}
}

// Benchmark: TypedRegion.Set must be zero-alloc
func BenchmarkTypedRegionSet(b *testing.B) {
	arena := func() *RegionArena {
		var zero SensorData
		layout, _ := LayoutFromStruct("sensor", zero)
		spec := &RegionSpec{Name: "sensor", Size: layout.Size, Mode: ModeStream,
			Header: &Header{Fields: func() []*HeaderField {
				hf := make([]*HeaderField, len(layout.Fields))
				for i, f := range layout.Fields {
					hf[i] = &HeaderField{Name: f.Name, Type: f.Type, Offset: f.Offset, Size: f.Size}
				}
				return hf
			}()}}
		lt := NewLayoutTable()
		lt.Add(layout)
		a, _ := NewArena([]*RegionSpec{spec}, lt)
		return a
	}()

	tr, _ := NewTypedRegion[SensorData](arena, "sensor")

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tr.Set(SensorData{Temperature: float32(i), Count: uint32(i)})
	}
}
