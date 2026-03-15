package mempipe

import (
	"context"
	"testing"
	"time"
)

// ---- test structs ----

type Counter struct {
	Value uint64 `mempipe:"field:value"`
}

type SensorReading struct {
	Temp     float32 `mempipe:"field:temp"`
	Humidity float32 `mempipe:"field:humidity"`
	Count    uint32  `mempipe:"field:count"`
	Active   bool    `mempipe:"field:active"`
}

func TestPipelineBasic(t *testing.T) {
	pipe := NewPipeline()
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("increment", func() {
		v := counter.Get()
		v.Value++
		counter.Set(v)
	})

	if err := pipe.Run(10); err != nil {
		t.Fatal(err)
	}

	v := counter.Get()
	if v.Value != 10 {
		t.Errorf("Counter: got %d, want 10", v.Value)
	}
}

func TestPipelineMultiRegion(t *testing.T) {
	pipe := NewPipeline()
	sensor := AddRegion[SensorReading](pipe, "sensor")
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("process", func() {
		s := sensor.Get()
		s.Temp += 1.0
		s.Count++
		sensor.Set(s)

		c := counter.Get()
		c.Value = uint64(s.Count)
		counter.Set(c)
	})

	if err := pipe.Run(5); err != nil {
		t.Fatal(err)
	}

	s := sensor.Get()
	if s.Count != 5 {
		t.Errorf("Sensor count: got %d, want 5", s.Count)
	}
	c := counter.Get()
	if c.Value != 5 {
		t.Errorf("Counter: got %d, want 5", c.Value)
	}
}

func TestPipelineOnIteration(t *testing.T) {
	pipe := NewPipeline()
	counter := AddRegion[Counter](pipe, "counter")

	iterations := 0
	pipe.OnIteration(func(iter int) {
		iterations++
	})

	pipe.SimpleCell("noop", func() {
		v := counter.Get()
		v.Value++
		counter.Set(v)
	})

	if err := pipe.Run(7); err != nil {
		t.Fatal(err)
	}

	if iterations != 7 {
		t.Errorf("OnIteration called %d times, want 7", iterations)
	}
}

func TestPipelineValidation(t *testing.T) {
	t.Run("duplicate region", func(t *testing.T) {
		pipe := NewPipeline()
		AddRegion[Counter](pipe, "dup")
		AddRegion[Counter](pipe, "dup")
		err := pipe.Run(1)
		if err == nil {
			t.Error("Expected error for duplicate region")
		}
	})

	t.Run("unknown input region", func(t *testing.T) {
		pipe := NewPipeline()
		AddRegion[Counter](pipe, "data")
		pipe.Cell("bad", func() {}, []string{"nonexistent"}, nil)
		err := pipe.Run(1)
		if err == nil {
			t.Error("Expected error for unknown input region")
		}
	})
}

func TestPipelineRunContinuous(t *testing.T) {
	pipe := NewPipeline()
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("inc", func() {
		v := counter.Get()
		v.Value++
		counter.Set(v)
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := pipe.RunContinuous(ctx)
	if err != context.DeadlineExceeded {
		t.Errorf("Expected DeadlineExceeded, got %v", err)
	}

	v := counter.Get()
	if v.Value == 0 {
		t.Error("Counter should be > 0 after RunContinuous")
	}
}

func TestPipelineWithWorkers(t *testing.T) {
	pipe := NewPipeline(WithWorkers(4))
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("inc", func() {
		v := counter.Get()
		v.Value++
		counter.Set(v)
	})

	if err := pipe.Run(100); err != nil {
		t.Fatal(err)
	}

	v := counter.Get()
	if v.Value != 100 {
		t.Errorf("Counter: got %d, want 100", v.Value)
	}
}

func TestPipelineFieldRegion(t *testing.T) {
	pipe := NewPipeline()

	pipe.AddFieldRegion("data", map[string]string{
		"x": "f64",
		"y": "f64",
	})

	pipe.SimpleCell("noop", func() {})

	if err := pipe.Run(1); err != nil {
		t.Fatal(err)
	}

	if pipe.Arena() == nil {
		t.Error("Arena should be non-nil after Run")
	}
}

func BenchmarkPipelineTypedRegion(b *testing.B) {
	pipe := NewPipeline()
	counter := AddRegion[Counter](pipe, "counter")

	pipe.SimpleCell("inc", func() {
		v := counter.Get()
		v.Value++
		counter.Set(v)
	})

	// Build once
	if err := pipe.Run(1); err != nil {
		b.Fatal(err)
	}

	// Reset
	counter.Set(Counter{})

	b.ResetTimer()
	b.ReportAllocs()

	if err := pipe.Run(b.N); err != nil {
		b.Fatal(err)
	}
}
