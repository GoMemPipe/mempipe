package builder

import (
	"testing"
)

// Phase 4 Benchmarks: Prove builder Context is zero-alloc

func BenchmarkBuilderContextU32Read(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"val": Uint32,
	})
	pipe.Continuous(1)

	// Initialize (allowed to allocate)
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	// Set initial value
	ctx.SetU32(region, "val", 42)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val := ctx.U32(region, "val")
		_ = val
	}
}

func BenchmarkBuilderContextU32Write(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"val": Uint32,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ctx.SetU32(region, "val", uint32(i))
	}
}

func BenchmarkBuilderContextF64Read(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"val": Float64,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	// Set initial value
	ctx.SetF64(region, "val", 3.14159)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		val := ctx.F64(region, "val")
		_ = val
	}
}

func BenchmarkBuilderContextF64Write(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"val": Float64,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ctx.SetF64(region, "val", float64(i)*0.1)
	}
}

func BenchmarkBuilderContextBoolReadWrite(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"flag": Bool,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write
		ctx.SetBool(region, "flag", i%2 == 0)
		// Read
		val := ctx.Bool(region, "flag")
		_ = val
	}
}

func BenchmarkBuilderRealisticWorkload(b *testing.B) {
	pipe := NewPipeline()
	input := pipe.Region("input", Fields{
		"count": Uint32,
		"sum":   Uint64,
	})
	output := pipe.Region("output", Fields{
		"avg":    Float64,
		"active": Bool,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	// Set initial values
	ctx.SetU32(input, "count", 10)
	ctx.SetU64(input, "sum", 1000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Read inputs
		count := ctx.U32(input, "count")
		sum := ctx.U64(input, "sum")

		// Compute
		avg := float64(sum) / float64(count)
		active := avg > 50.0

		// Write outputs
		ctx.SetF64(output, "avg", avg)
		ctx.SetBool(output, "active", active)
	}
}

func BenchmarkBuilderMultiFieldAccess(b *testing.B) {
	pipe := NewPipeline()
	region := pipe.Region("bench", Fields{
		"a": Uint32,
		"b": Uint32,
		"c": Uint32,
		"d": Uint32,
	})
	pipe.Continuous(1)

	// Initialize
	if err := pipe.build(); err != nil {
		b.Fatal(err)
	}

	// Create context
	ctx := &Context{
		arena: pipe.arena,
		vars:  make(map[string]interface{}, 8),
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Read all
		a := ctx.U32(region, "a")
		b := ctx.U32(region, "b")
		c := ctx.U32(region, "c")
		d := ctx.U32(region, "d")

		// Compute
		result := a + b + c + d

		// Write back
		ctx.SetU32(region, "a", result)
	}
}

// Benchmark pipeline execution (full run including cell overhead)
func BenchmarkBuilderPipelineRun(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Create pipeline
		pipe := NewPipeline()
		data := pipe.Region("data", Fields{
			"count": Uint32,
		})

		pipe.Cell("increment", func(ctx *Context) {
			count := ctx.U32(data, "count")
			ctx.SetU32(data, "count", count+1)
		})

		pipe.Continuous(10)

		// Run
		if err := pipe.Run(); err != nil {
			b.Fatal(err)
		}
	}
}
