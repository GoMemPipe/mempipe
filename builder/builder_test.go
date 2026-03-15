package builder

import (
	"testing"
)

func TestBasicPipeline(t *testing.T) {
	pipe := NewPipeline()

	// Define a simple region
	data := pipe.Region("data", Fields{
		"value": Uint32,
	})

	// Define a cell that increments the value
	pipe.Cell("increment", func(ctx *Context) {
		val := ctx.U32(data, "value")
		ctx.SetU32(data, "value", val+1)
	})

	// Run for 5 iterations
	pipe.Continuous(5)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}

	// After 5 iterations, value should be 5
	// (starts at 0, incremented 5 times)
}

func TestMultipleRegions(t *testing.T) {
	pipe := NewPipeline()

	input := pipe.Region("input", Fields{
		"x": Float64,
		"y": Float64,
	})

	output := pipe.Region("output", Fields{
		"sum":     Float64,
		"product": Float64,
	})

	pipe.Cell("compute", func(ctx *Context) {
		x := ctx.F64(input, "x")
		y := ctx.F64(input, "y")

		ctx.SetF64(output, "sum", x+y)
		ctx.SetF64(output, "product", x*y)
	})

	pipe.Cell("init", func(ctx *Context) {
		ctx.SetF64(input, "x", 3.0)
		ctx.SetF64(input, "y", 4.0)
	})

	pipe.Continuous(1)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}
}

func TestAllTypes(t *testing.T) {
	pipe := NewPipeline()

	data := pipe.Region("data", Fields{
		"u8":  Uint8,
		"u16": Uint16,
		"u32": Uint32,
		"u64": Uint64,
		"i8":  Int8,
		"i16": Int16,
		"i32": Int32,
		"i64": Int64,
		"f32": Float32,
		"f64": Float64,
		"b":   Bool,
	})

	pipe.Cell("set_values", func(ctx *Context) {
		ctx.SetU8(data, "u8", 255)
		ctx.SetU16(data, "u16", 65535)
		ctx.SetU32(data, "u32", 4294967295)
		ctx.SetU64(data, "u64", 18446744073709551615)
		ctx.SetI8(data, "i8", -128)
		ctx.SetI16(data, "i16", -32768)
		ctx.SetI32(data, "i32", -2147483648)
		ctx.SetI64(data, "i64", -9223372036854775808)
		ctx.SetF32(data, "f32", 3.14159)
		ctx.SetF64(data, "f64", 2.71828)
		ctx.SetBool(data, "b", true)
	})

	pipe.Cell("verify_values", func(ctx *Context) {
		if ctx.U8(data, "u8") != 255 {
			t.Error("U8 value mismatch")
		}
		if ctx.Bool(data, "b") != true {
			t.Error("Bool value mismatch")
		}
		// Note: Float comparisons should use epsilon in production
		if ctx.F32(data, "f32") < 3.14 || ctx.F32(data, "f32") > 3.15 {
			t.Error("F32 value mismatch")
		}
	})

	pipe.Continuous(1)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}
}

func TestIterationContext(t *testing.T) {
	pipe := NewPipeline()

	data := pipe.Region("data", Fields{
		"iteration": Uint32,
	})

	expectedIterations := 10
	actualIterations := 0

	pipe.Cell("track_iteration", func(ctx *Context) {
		iter := ctx.Iteration()
		ctx.SetU32(data, "iteration", uint32(iter))
		actualIterations++

		// Verify iteration number matches
		if uint32(iter) != ctx.U32(data, "iteration") {
			t.Errorf("Iteration mismatch: expected %d, got %d", iter, ctx.U32(data, "iteration"))
		}
	})

	pipe.Continuous(expectedIterations)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}

	if actualIterations != expectedIterations {
		t.Errorf("Expected %d iterations, got %d", expectedIterations, actualIterations)
	}
}

func TestLocalVariables(t *testing.T) {
	pipe := NewPipeline()

	data := pipe.Region("data", Fields{
		"result": Uint64,
	})

	pipe.Cell("use_vars", func(ctx *Context) {
		// Store some values in local variables
		ctx.SetVar("temp", uint64(100))
		ctx.SetVar("multiplier", uint64(5))

		// Retrieve and use them
		temp := ctx.GetVar("temp").(uint64)
		mult := ctx.GetVar("multiplier").(uint64)

		result := temp * mult
		ctx.SetU64(data, "result", result)
	})

	pipe.Continuous(1)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}
}

func TestConditionalLogic(t *testing.T) {
	pipe := NewPipeline()

	sensors := pipe.Region("sensors", Fields{
		"temperature": Float32,
	})

	alerts := pipe.Region("alerts", Fields{
		"high_temp": Bool,
		"low_temp":  Bool,
	})

	pipe.Cell("set_temp", func(ctx *Context) {
		// Vary temperature across iterations
		temp := float32(10.0 + float64(ctx.Iteration())*5.0)
		ctx.SetF32(sensors, "temperature", temp)
	})

	pipe.Cell("check_alerts", func(ctx *Context) {
		temp := ctx.F32(sensors, "temperature")

		if temp > 30.0 {
			ctx.SetBool(alerts, "high_temp", true)
		} else {
			ctx.SetBool(alerts, "high_temp", false)
		}

		if temp < 15.0 {
			ctx.SetBool(alerts, "low_temp", true)
		} else {
			ctx.SetBool(alerts, "low_temp", false)
		}
	})

	pipe.Continuous(10)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}
}

func TestAccumulation(t *testing.T) {
	pipe := NewPipeline()

	state := pipe.Region("state", Fields{
		"sum":   Uint64,
		"count": Uint32,
	})

	pipe.Cell("accumulate", func(ctx *Context) {
		sum := ctx.U64(state, "sum")
		count := ctx.U32(state, "count")

		// Add current iteration number to sum
		sum += uint64(ctx.Iteration() + 1)
		count++

		ctx.SetU64(state, "sum", sum)
		ctx.SetU32(state, "count", count)
	})

	n := 10
	pipe.Continuous(n)

	if err := pipe.Run(); err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}

	// Verify sum: 1+2+3+...+10 = 55
	// But we need to access the final state somehow
	// For now, just verify it runs without error
}

func BenchmarkSimplePipeline(b *testing.B) {
	pipe := NewPipeline()

	data := pipe.Region("data", Fields{
		"value": Uint64,
	})

	pipe.Cell("increment", func(ctx *Context) {
		val := ctx.U64(data, "value")
		ctx.SetU64(data, "value", val+1)
	})

	pipe.Continuous(b.N)

	if err := pipe.Run(); err != nil {
		b.Fatalf("Pipeline failed: %v", err)
	}
}

func BenchmarkMultiRegion(b *testing.B) {
	pipe := NewPipeline()

	r1 := pipe.Region("r1", Fields{"v": Uint64})
	r2 := pipe.Region("r2", Fields{"v": Uint64})
	r3 := pipe.Region("r3", Fields{"v": Uint64})

	pipe.Cell("cell1", func(ctx *Context) {
		ctx.SetU64(r1, "v", uint64(ctx.Iteration()))
	})

	pipe.Cell("cell2", func(ctx *Context) {
		v := ctx.U64(r1, "v")
		ctx.SetU64(r2, "v", v*2)
	})

	pipe.Cell("cell3", func(ctx *Context) {
		v := ctx.U64(r2, "v")
		ctx.SetU64(r3, "v", v+100)
	})

	pipe.Continuous(b.N)

	if err := pipe.Run(); err != nil {
		b.Fatalf("Pipeline failed: %v", err)
	}
}
