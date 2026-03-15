package audio

import (
	"testing"
	"unsafe"

	"github.com/GoMemPipe/mempipe/runtime"
)

func setupBenchAudio(b *testing.B) (*AudioModule, *runtime.Region) {
	specs := []*runtime.RegionSpec{
		{
			Name: "audio",
			Size: 4096 * 4,
			Mode: runtime.ModeStream,
			Header: &runtime.Header{
				Fields: []*runtime.HeaderField{
					{Name: "samples", Type: runtime.TypeVecF32, Offset: 0, Size: 4096 * 4},
				},
			},
		},
	}

	layouts := runtime.NewLayoutTable()
	layout := runtime.ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, err := runtime.NewArena(specs, layouts)
	if err != nil {
		b.Fatalf("NewArena failed: %v", err)
	}

	view := arena.MustRegion("audio")
	mod := NewAudioModule(44100, 12345)

	if err := mod.Attach(view, "samples"); err != nil {
		b.Fatalf("Attach failed: %v", err)
	}

	return mod, view
}

//mem:nogc
func BenchmarkAudioGenSine(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.GenSine(1024)
	}
}

//mem:nogc
func BenchmarkAudioGenNoise(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.GenNoise(1024, 1.0)
	}
}

//mem:nogc
func BenchmarkAudioGenSilence(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.GenSilence(1024)
	}
}

//mem:nogc
func BenchmarkAudioPullFramePtr(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)
	mod.GenSine(1024)

	b.ResetTimer()
	b.ReportAllocs()

	var ptr unsafe.Pointer
	var length int
	var err error

	for i := 0; i < b.N; i++ {
		ptr, length, err = mod.PullFramePtr()
		if err != nil {
			b.Fatalf("PullFramePtr failed: %v", err)
		}
	}

	_ = ptr
	_ = length
}

func BenchmarkAudioWriteSamples(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.WriteSamples(src)
	}
}

//mem:nogc
func BenchmarkAudioSetFrequency(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	freq := 440.0
	for i := 0; i < b.N; i++ {
		mod.SetFrequency(freq)
		freq += 1.0
		if freq > 880.0 {
			freq = 440.0
		}
	}
}

//mem:nogc
func BenchmarkAudioResetPhase(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.ResetPhase()
	}
}

//mem:nogc
func BenchmarkAudioResetPRNG(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	seed := uint64(12345)
	for i := 0; i < b.N; i++ {
		mod.ResetPRNG(seed)
		seed++
	}
}

//mem:nogc
func BenchmarkAudioRealisticWorkload(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.GenSine(256)

		ptr, length, _ := mod.PullFramePtr()

		var sum float32
		for j := 0; j < 256 && j < length; j++ {
			elemPtr := unsafe.Add(ptr, j*4)
			sample := *(*float32)(elemPtr)
			sum += sample
		}

		mod.GenNoise(256, 0.5)

		_, _, _ = mod.PullFramePtr()

		_ = sum
	}
}

//mem:nogc
func BenchmarkAudioFullPipeline(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mod.GenSine(512)
		mod.GenNoise(256, 0.3)
		mod.GenSilence(256)
		_, _, _ = mod.PullFramePtr()
	}
}

//mem:nogc
func BenchmarkAudioHighFrequencyUpdates(b *testing.B) {
	mod, _ := setupBenchAudio(b)

	b.ResetTimer()
	b.ReportAllocs()

	freq := 440.0
	for i := 0; i < b.N; i++ {
		mod.SetFrequency(freq)
		freq += 0.1
		if freq > 880.0 {
			freq = 440.0
		}

		mod.GenSine(64)

		mod.ResetPRNG(uint64(i))

		mod.GenNoise(64, 0.5)
	}
}

//mem:nogc
func BenchmarkAudioMemoryAccess(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.GenSine(1024)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ptr, length, _ := mod.PullFramePtr()

		var sum float32
		for j := 0; j < length; j++ {
			elemPtr := unsafe.Add(ptr, j*4)
			sample := *(*float32)(elemPtr)
			sum += sample
		}

		_ = sum
	}
}

// --- DSP operator benchmarks ---

//mem:nogc
func BenchmarkAudioLowPassFilter(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)
	mod.GenSine(1024)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.LowPassFilter(1024, 1000.0)
	}
}

//mem:nogc
func BenchmarkAudioHighPassFilter(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)
	mod.GenSine(1024)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.HighPassFilter(1024, 200.0)
	}
}

//mem:nogc
func BenchmarkAudioGain(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)
	mod.GenSine(1024)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.Gain(1024, 0.5)
	}
}

//mem:nogc
func BenchmarkAudioFilterChain(b *testing.B) {
	mod, _ := setupBenchAudio(b)
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.GenSine(1024)
		mod.LowPassFilter(1024, 2000.0)
		mod.Gain(1024, 0.8)
		_, _, _ = mod.PullFramePtr()
	}
}

func BenchmarkAudioFilterChain4096(b *testing.B) {
	specs := []*runtime.RegionSpec{
		{
			Name: "audio",
			Size: 4096 * 4 * 4,
			Mode: runtime.ModeStream,
			Header: &runtime.Header{
				Fields: []*runtime.HeaderField{
					{Name: "samples", Type: runtime.TypeVecF32, Offset: 0, Size: 4096 * 4 * 4},
				},
			},
		},
	}
	layouts := runtime.NewLayoutTable()
	layout := runtime.ComputeLayout(specs[0])
	layouts.Add(layout)
	arena, _ := runtime.NewArena(specs, layouts)
	view := arena.MustRegion("audio")
	mod := NewAudioModule(44100, 12345)
	mod.Attach(view, "samples")
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.GenSine(4096)
		mod.LowPassFilter(4096, 2000.0)
		mod.Gain(4096, 0.8)
	}
}

func BenchmarkAudioFilterChain16384(b *testing.B) {
	specs := []*runtime.RegionSpec{
		{
			Name: "audio",
			Size: 16384 * 4 * 4,
			Mode: runtime.ModeStream,
			Header: &runtime.Header{
				Fields: []*runtime.HeaderField{
					{Name: "samples", Type: runtime.TypeVecF32, Offset: 0, Size: 16384 * 4 * 4},
				},
			},
		},
	}
	layouts := runtime.NewLayoutTable()
	layout := runtime.ComputeLayout(specs[0])
	layouts.Add(layout)
	arena, _ := runtime.NewArena(specs, layouts)
	view := arena.MustRegion("audio")
	mod := NewAudioModule(44100, 12345)
	mod.Attach(view, "samples")
	mod.SetFrequency(440.0)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		mod.GenSine(16384)
		mod.LowPassFilter(16384, 2000.0)
		mod.Gain(16384, 0.8)
	}
}
