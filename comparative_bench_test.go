package mempipe

import (
	"testing"
)

// ────────────────────────────────────────────────────────────────────────────
// Comparative benchmarks: MemPipe arena vs heap allocation
//
// These benchmarks quantify the advantage of arena-backed regions over
// idiomatic Go heap-allocated structs for pipeline workloads.
// ────────────────────────────────────────────────────────────────────────────

// ── Data types ──

type BenchSensor struct {
	Temp     float32 `mempipe:"field:temp"`
	Humidity float32 `mempipe:"field:humidity"`
	Count    uint32  `mempipe:"field:count"`
	Active   bool    `mempipe:"field:active"`
}

// heapSensor mirrors BenchSensor but lives on the Go heap.
type heapSensor struct {
	Temp     float32
	Humidity float32
	Count    uint32
	Active   bool
}

// ────────────────────────────────────────────────────────────────────────────
// Benchmark: Pipeline tick (mempipe typed region vs heap struct)
// ────────────────────────────────────────────────────────────────────────────

// BenchmarkComparativePipeline_Arena measures a pipeline tick using
// arena-backed typed regions. Expected: 0 allocs/op.
func BenchmarkComparativePipeline_Arena(b *testing.B) {
	pipe := NewPipeline()
	sensor := AddRegion[BenchSensor](pipe, "sensor")
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("process", func() {
		s := sensor.Get()
		s.Temp += 0.1
		s.Count++
		s.Active = true
		sensor.Set(s)

		c := counter.Get()
		c.Value = uint64(s.Count)
		counter.Set(c)
	})

	// Build pipeline once
	if err := pipe.Run(1); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	if err := pipe.Run(b.N); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkComparativePipeline_Heap measures equivalent work using heap-
// allocated Go structs with pointer indirection. This is the "normal Go" way.
func BenchmarkComparativePipeline_Heap(b *testing.B) {
	sensor := &heapSensor{}
	counter := &struct{ Value uint64 }{}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sensor.Temp += 0.1
		sensor.Count++
		sensor.Active = true
		counter.Value = uint64(sensor.Count)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Benchmark: Multi-region read/write (arena vs heap map lookup)
// ────────────────────────────────────────────────────────────────────────────

type BenchState struct {
	X float32 `mempipe:"field:x"`
	Y float32 `mempipe:"field:y"`
	Z float32 `mempipe:"field:z"`
}

// BenchmarkComparativeMultiRegion_Arena tests reading/writing multiple
// arena regions per tick. Expected: 0 allocs/op.
func BenchmarkComparativeMultiRegion_Arena(b *testing.B) {
	pipe := NewPipeline()
	r1 := AddRegion[BenchState](pipe, "r1")
	r2 := AddRegion[BenchState](pipe, "r2")
	r3 := AddRegion[BenchState](pipe, "r3")

	pipe.SimpleCell("compute", func() {
		v1 := r1.Get()
		v2 := r2.Get()
		v1.X += 0.1
		v2.Y = v1.X * 2
		r1.Set(v1)
		r2.Set(v2)

		v3 := r3.Get()
		v3.Z = v1.X + v2.Y
		r3.Set(v3)
	})

	if err := pipe.Run(1); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	if err := pipe.Run(b.N); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkComparativeMultiRegion_Heap measures equivalent multi-region
// work using heap-allocated maps (common Go pattern for named data stores).
func BenchmarkComparativeMultiRegion_Heap(b *testing.B) {
	store := map[string]*[3]float32{
		"r1": {},
		"r2": {},
		"r3": {},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		v1 := store["r1"]
		v2 := store["r2"]
		v1[0] += 0.1
		v2[1] = v1[0] * 2
		v3 := store["r3"]
		v3[2] = v1[0] + v2[1]
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Benchmark: High-frequency counter (tests scheduling overhead)
// ────────────────────────────────────────────────────────────────────────────

// BenchmarkComparativeCounter_Arena measures raw per-tick overhead of the
// mempipe scheduler + arena region access.
func BenchmarkComparativeCounter_Arena(b *testing.B) {
	pipe := NewPipeline()
	c := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("inc", func() {
		v := c.Get()
		v.Value++
		c.Set(v)
	})

	if err := pipe.Run(1); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	if err := pipe.Run(b.N); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkComparativeCounter_Heap measures a plain counter increment.
func BenchmarkComparativeCounter_Heap(b *testing.B) {
	var v uint64
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v++
	}
	_ = v
}
