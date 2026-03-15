package audio

import (
	"math"
	"testing"
	"unsafe"

	"github.com/GoMemPipe/mempipe/runtime"
)

// Helper to create a test arena with audio buffer
func createTestAudioArena(t *testing.T, bufferSize int) (*runtime.RegionArena, *runtime.Region) {
	specs := []*runtime.RegionSpec{
		{
			Name: "audio",
			Size: int64(bufferSize * 4),
			Mode: runtime.ModeStream,
			Header: &runtime.Header{
				Fields: []*runtime.HeaderField{
					{Name: "samples", Type: runtime.TypeVecF32, Offset: 0, Size: bufferSize * 4},
				},
			},
		},
	}

	layouts := runtime.NewLayoutTable()
	layout := runtime.ComputeLayout(specs[0])
	layouts.Add(layout)

	arena, err := runtime.NewArena(specs, layouts)
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	view := arena.MustRegion("audio")
	return arena, view
}

func TestNewAudioModule(t *testing.T) {
	mod := NewAudioModule(44100, 12345)

	if mod == nil {
		t.Fatal("NewAudioModule returned nil")
	}

	if mod.sampleRate != 44100 {
		t.Errorf("Sample rate: got %f, want 44100", mod.sampleRate)
	}

	if mod.prngState != 12345 {
		t.Errorf("PRNG state: got %d, want 12345", mod.prngState)
	}

	if len(mod.ringBuf) != 4096 {
		t.Errorf("Ring buffer size: got %d, want 4096", len(mod.ringBuf))
	}
}

func TestAudioModuleAttach(t *testing.T) {
	_, view := createTestAudioArena(t, 1024)
	mod := NewAudioModule(44100, 12345)

	err := mod.Attach(view, "samples")
	if err != nil {
		t.Errorf("Attach failed: %v", err)
	}

	if mod.attachedView == nil {
		t.Error("attachedView is nil after successful attach")
	}

	if mod.attachedField != "samples" {
		t.Errorf("attachedField: got %s, want samples", mod.attachedField)
	}

	err = mod.Attach(view, "nonexistent")
	if err == nil {
		t.Error("Attach to non-existent field should fail")
	}
}

func TestGenSine(t *testing.T) {
	_, view := createTestAudioArena(t, 1024)
	mod := NewAudioModule(44100, 12345)

	if err := mod.Attach(view, "samples"); err != nil {
		t.Fatalf("Attach failed: %v", err)
	}

	mod.SetFrequency(440.0)

	count, err := mod.GenSine(100)
	if err != nil {
		t.Errorf("GenSine failed: %v", err)
	}

	if count != 100 {
		t.Errorf("GenSine count: got %d, want 100", count)
	}

	ptr, length, err := mod.PullFramePtr()
	if err != nil {
		t.Fatalf("PullFramePtr failed: %v", err)
	}

	if length < 100 {
		t.Fatalf("Buffer too small: got %d, want >= 100", length)
	}

	for i := 0; i < 100; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)

		if sample < -1.0 || sample > 1.0 {
			t.Errorf("Sample %d out of range: %f", i, sample)
		}
	}

	firstPtr := ptr
	firstSample := *(*float32)(firstPtr)
	if math.Abs(float64(firstSample)) > 0.1 {
		t.Errorf("First sample should be close to 0, got %f", firstSample)
	}
}

func TestGenNoise(t *testing.T) {
	_, view := createTestAudioArena(t, 1024)
	mod := NewAudioModule(44100, 12345)

	if err := mod.Attach(view, "samples"); err != nil {
		t.Fatalf("Attach failed: %v", err)
	}

	count, err := mod.GenNoise(100, 0.5)
	if err != nil {
		t.Errorf("GenNoise failed: %v", err)
	}

	if count != 100 {
		t.Errorf("GenNoise count: got %d, want 100", count)
	}

	ptr, _, err := mod.PullFramePtr()
	if err != nil {
		t.Fatalf("PullFramePtr failed: %v", err)
	}

	for i := 0; i < 100; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)

		if sample < -0.5 || sample > 0.5 {
			t.Errorf("Sample %d out of range: %f", i, sample)
		}
	}

	mod2 := NewAudioModule(44100, 12345)
	_, view2 := createTestAudioArena(t, 1024)
	mod2.Attach(view2, "samples")
	mod2.GenNoise(100, 0.5)

	ptr2, _, _ := mod2.PullFramePtr()

	for i := 0; i < 100; i++ {
		elemPtr1 := unsafe.Add(ptr, i*4)
		elemPtr2 := unsafe.Add(ptr2, i*4)
		sample1 := *(*float32)(elemPtr1)
		sample2 := *(*float32)(elemPtr2)

		if sample1 != sample2 {
			t.Errorf("Sample %d not deterministic: %f != %f", i, sample1, sample2)
		}
	}
}

func TestGenSilence(t *testing.T) {
	_, view := createTestAudioArena(t, 1024)
	mod := NewAudioModule(44100, 12345)

	mod.Attach(view, "samples")
	mod.GenNoise(100, 1.0)

	count, err := mod.GenSilence(100)
	if err != nil {
		t.Errorf("GenSilence failed: %v", err)
	}

	if count != 100 {
		t.Errorf("GenSilence count: got %d, want 100", count)
	}

	ptr, _, err := mod.PullFramePtr()
	if err != nil {
		t.Fatalf("PullFramePtr failed: %v", err)
	}

	for i := 0; i < 100; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)

		if sample != 0.0 {
			t.Errorf("Sample %d not zero: %f", i, sample)
		}
	}
}

func TestReadWriteSamples(t *testing.T) {
	_, view := createTestAudioArena(t, 1024)
	mod := NewAudioModule(44100, 12345)

	mod.Attach(view, "samples")

	src := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	count, err := mod.WriteSamples(src)
	if err != nil {
		t.Errorf("WriteSamples failed: %v", err)
	}

	if count != 5 {
		t.Errorf("WriteSamples count: got %d, want 5", count)
	}

	dst := make([]float32, 5)
	count, err = mod.ReadSamples(dst)
	if err != nil {
		t.Errorf("ReadSamples failed: %v", err)
	}

	if count != 5 {
		t.Errorf("ReadSamples count: got %d, want 5", count)
	}

	for i := 0; i < 5; i++ {
		if dst[i] != src[i] {
			t.Errorf("Sample %d mismatch: got %f, want %f", i, dst[i], src[i])
		}
	}
}

func TestPhaseReset(t *testing.T) {
	mod := NewAudioModule(44100, 12345)
	mod.SetFrequency(440.0)

	_, view := createTestAudioArena(t, 1024)
	mod.Attach(view, "samples")
	mod.GenSine(100)

	if mod.GetPhase() == 0.0 {
		t.Error("Phase should be non-zero after generating samples")
	}

	mod.ResetPhase()

	if mod.GetPhase() != 0.0 {
		t.Errorf("Phase after reset: got %f, want 0.0", mod.GetPhase())
	}
}

func TestPRNGReset(t *testing.T) {
	mod := NewAudioModule(44100, 12345)

	_, view := createTestAudioArena(t, 1024)
	mod.Attach(view, "samples")
	mod.GenNoise(100, 1.0)

	if mod.GetPRNGState() == 12345 {
		t.Error("PRNG state should have changed after generating noise")
	}

	mod.ResetPRNG(12345)

	if mod.GetPRNGState() != 12345 {
		t.Errorf("PRNG state after reset: got %d, want 12345", mod.GetPRNGState())
	}
}

func TestGenWithoutAttach(t *testing.T) {
	mod := NewAudioModule(44100, 12345)

	_, err := mod.GenSine(100)
	if err == nil {
		t.Error("GenSine without attach should fail")
	}

	_, err = mod.GenNoise(100, 1.0)
	if err == nil {
		t.Error("GenNoise without attach should fail")
	}

	_, err = mod.GenSilence(100)
	if err == nil {
		t.Error("GenSilence without attach should fail")
	}
}

func TestCapacityLimit(t *testing.T) {
	_, view := createTestAudioArena(t, 10)
	mod := NewAudioModule(44100, 12345)

	mod.Attach(view, "samples")

	count, err := mod.GenSine(100)
	if err != nil {
		t.Errorf("GenSine failed: %v", err)
	}

	if count != 10 {
		t.Errorf("Generated count: got %d, want 10 (limited by capacity)", count)
	}
}
