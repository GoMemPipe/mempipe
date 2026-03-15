package runtime

import (
	"testing"
)

func TestNewArena(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "region1",
			Size: 16,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
					{Name: "b", Type: TypeU64, Offset: 4, Size: 8},
				},
			},
		},
		{
			Name: "region2",
			Size: 8,
			Mode: ModeRing,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "c", Type: TypeF32, Offset: 0, Size: 4},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	for _, spec := range specs {
		layout := ComputeLayout(spec)
		layouts.Add(layout)
	}

	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	if arena.Size() != 128 {
		t.Errorf("Arena size: got %d, want 128 (64-byte aligned)", arena.Size())
	}

	if arena.RegionCount() != 2 {
		t.Errorf("Region count: got %d, want 2", arena.RegionCount())
	}

	if !arena.HasRegion("region1") {
		t.Error("Arena missing view for region1")
	}

	if !arena.HasRegion("region2") {
		t.Error("Arena missing view for region2")
	}
}

func TestArenaView(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 8,
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

	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	view, err := arena.Region("test")
	if err != nil {
		t.Errorf("View() failed: %v", err)
	}
	if view == nil {
		t.Fatal("View() returned nil")
	}
	if view.Name() != "test" {
		t.Errorf("View name: got %s, want test", view.Name())
	}

	view2 := arena.MustRegion("test")
	if view2 != view {
		t.Error("MustView() returned different view")
	}

	_, err = arena.Region("nonexistent")
	if err == nil {
		t.Error("View() should fail for non-existent region")
	}
}

func TestArenaMustViewPanic(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 8,
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

	defer func() {
		if r := recover(); r == nil {
			t.Error("MustView() should panic for non-existent region")
		}
	}()

	arena.MustRegion("nonexistent")
}

func TestArenaZero(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 16,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
					{Name: "b", Type: TypeU32, Offset: 4, Size: 4},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, _ := NewArena(specs, layouts)
	view := arena.MustRegion("test")

	view.SetU32("a", 42)
	view.SetU32("b", 100)

	val, _ := view.U32("a")
	if val != 42 {
		t.Errorf("Before zero: got %d, want 42", val)
	}

	arena.Zero()

	val, _ = view.U32("a")
	if val != 0 {
		t.Errorf("After zero: got %d, want 0", val)
	}

	val, _ = view.U32("b")
	if val != 0 {
		t.Errorf("After zero: got %d, want 0", val)
	}
}

func TestArenaZeroRegion(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "region1",
			Size: 8,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
				},
			},
		},
		{
			Name: "region2",
			Size: 8,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	for _, spec := range specs {
		layout := ComputeLayout(spec)
		layouts.Add(layout)
	}

	arena, _ := NewArena(specs, layouts)
	view1 := arena.MustRegion("region1")
	view2 := arena.MustRegion("region2")

	view1.SetU64("val", 42)
	view2.SetU64("val", 100)

	err := arena.ZeroRegion("region1")
	if err != nil {
		t.Errorf("ZeroRegion failed: %v", err)
	}

	val, _ := view1.U64("val")
	if val != 0 {
		t.Errorf("Region1 after zero: got %d, want 0", val)
	}

	val, _ = view2.U64("val")
	if val != 100 {
		t.Errorf("Region2 after zero: got %d, want 100", val)
	}
}

func TestArenaValidate(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 8,
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

	if err := arena.Validate(); err != nil {
		t.Errorf("Validate() failed: %v", err)
	}
}

func TestArenaCopy(t *testing.T) {
	specs := []*RegionSpec{
		{
			Name: "test",
			Size: 16,
			Mode: ModeStream,
			Header: &Header{
				Fields: []*HeaderField{
					{Name: "a", Type: TypeU32, Offset: 0, Size: 4},
					{Name: "b", Type: TypeU64, Offset: 4, Size: 8},
				},
			},
		},
	}

	layouts := NewLayoutTable()
	layout := ComputeLayout(specs[0])
	layouts.Add(layout)

	src, _ := NewArena(specs, layouts)
	srcView := src.MustRegion("test")
	srcView.SetU32("a", 42)
	srcView.SetU64("b", 12345)

	dest, _ := NewArena(specs, layouts)

	err := src.Copy(dest)
	if err != nil {
		t.Errorf("Copy() failed: %v", err)
	}

	destView := dest.MustRegion("test")
	val32, _ := destView.U32("a")
	if val32 != 42 {
		t.Errorf("Copied u32: got %d, want 42", val32)
	}

	val64, _ := destView.U64("b")
	if val64 != 12345 {
		t.Errorf("Copied u64: got %d, want 12345", val64)
	}
}

func TestArenaRegionNames(t *testing.T) {
	specs := []*RegionSpec{
		{Name: "alpha", Size: 8, Mode: ModeStream, Header: &Header{}},
		{Name: "beta", Size: 8, Mode: ModeStream, Header: &Header{}},
		{Name: "gamma", Size: 8, Mode: ModeStream, Header: &Header{}},
	}

	layouts := NewLayoutTable()
	for _, spec := range specs {
		layout := ComputeLayout(spec)
		layouts.Add(layout)
	}

	arena, _ := NewArena(specs, layouts)
	names := arena.RegionNames()

	if len(names) != 3 {
		t.Errorf("RegionNames count: got %d, want 3", len(names))
	}

	nameMap := make(map[string]bool)
	for _, name := range names {
		nameMap[name] = true
	}

	for _, expected := range []string{"alpha", "beta", "gamma"} {
		if !nameMap[expected] {
			t.Errorf("RegionNames missing: %s", expected)
		}
	}
}

func TestNewArenaErrors(t *testing.T) {
	t.Run("nil specs", func(t *testing.T) {
		_, err := NewArena(nil, nil)
		if err == nil {
			t.Error("NewArena(nil, nil) should fail")
		}
	})

	t.Run("nil layouts", func(t *testing.T) {
		specs := []*RegionSpec{{Name: "test", Size: 8, Mode: ModeStream}}
		_, err := NewArena(specs, nil)
		if err == nil {
			t.Error("NewArena with nil layouts should fail")
		}
	})

	t.Run("zero size regions", func(t *testing.T) {
		specs := []*RegionSpec{{Name: "test", Size: 0, Mode: ModeStream}}
		layouts := NewLayoutTable()
		_, err := NewArena(specs, layouts)
		if err == nil {
			t.Error("NewArena with zero size should fail")
		}
	})

	t.Run("missing layout", func(t *testing.T) {
		specs := []*RegionSpec{
			{Name: "test", Size: 8, Mode: ModeStream, Header: &Header{}},
		}
		layouts := NewLayoutTable()
		_, err := NewArena(specs, layouts)
		if err == nil {
			t.Error("NewArena with missing layout should fail")
		}
	})
}

func TestArenaStats(t *testing.T) {
	specs := []*RegionSpec{
		{Name: "a", Size: 16, Mode: ModeStream, Header: &Header{
			Fields: []*HeaderField{{Name: "x", Type: TypeU64, Offset: 0, Size: 8}},
		}},
		{Name: "b", Size: 32, Mode: ModeStream, Header: &Header{
			Fields: []*HeaderField{{Name: "y", Type: TypeU64, Offset: 0, Size: 8}},
		}},
	}
	layouts := NewLayoutTable()
	for _, s := range specs {
		layouts.Add(ComputeLayout(s))
	}
	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatal(err)
	}

	stats := arena.Stats()
	if stats.RegionCount != 2 {
		t.Errorf("RegionCount: got %d, want 2", stats.RegionCount)
	}
	if stats.UsedSize != 48 {
		t.Errorf("UsedSize: got %d, want 48", stats.UsedSize)
	}
	if stats.TotalSize < stats.UsedSize {
		t.Errorf("TotalSize %d < UsedSize %d", stats.TotalSize, stats.UsedSize)
	}
	if stats.FragmentationRatio < 0 || stats.FragmentationRatio > 1 {
		t.Errorf("FragmentationRatio out of range: %f", stats.FragmentationRatio)
	}
}

func TestArenaGrow(t *testing.T) {
	specs := []*RegionSpec{
		{Name: "data", Size: 16, Mode: ModeStream, Header: &Header{
			Fields: []*HeaderField{
				{Name: "val", Type: TypeU64, Offset: 0, Size: 8},
			},
		}},
	}
	layouts := NewLayoutTable()
	layouts.Add(ComputeLayout(specs[0]))

	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatal(err)
	}

	// Write a value
	r := arena.MustRegion("data")
	if err := r.SetU64("val", 12345); err != nil {
		t.Fatal(err)
	}

	// Grow the region
	if err := arena.Grow("data", 128); err != nil {
		t.Fatal(err)
	}

	// Value should be preserved
	r2 := arena.MustRegion("data")
	val, err := r2.U64("val")
	if err != nil {
		t.Fatal(err)
	}
	if val != 12345 {
		t.Errorf("Value after grow: got %d, want 12345", val)
	}

	// Size should be updated
	if r2.Size() != 128 {
		t.Errorf("Region size after grow: got %d, want 128", r2.Size())
	}

	// Shrink should fail
	if err := arena.Grow("data", 64); err == nil {
		t.Error("Grow with smaller size should fail")
	}
}

func TestArenaAlignment(t *testing.T) {
	specs := []*RegionSpec{
		{Name: "r1", Size: 10, Mode: ModeStream, Header: &Header{
			Fields: []*HeaderField{{Name: "a", Type: TypeU8, Offset: 0, Size: 1}},
		}},
		{Name: "r2", Size: 10, Mode: ModeStream, Header: &Header{
			Fields: []*HeaderField{{Name: "b", Type: TypeU8, Offset: 0, Size: 1}},
		}},
	}
	layouts := NewLayoutTable()
	for _, s := range specs {
		layouts.Add(ComputeLayout(s))
	}
	arena, err := NewArena(specs, layouts)
	if err != nil {
		t.Fatal(err)
	}

	r1 := arena.MustRegion("r1")
	r2 := arena.MustRegion("r2")

	// Both region bases should be 64-byte aligned
	if uintptr(r1.Base())%64 != 0 {
		t.Errorf("r1 base %#x not 64-byte aligned", uintptr(r1.Base()))
	}
	if uintptr(r2.Base())%64 != 0 {
		t.Errorf("r2 base %#x not 64-byte aligned", uintptr(r2.Base()))
	}
}
