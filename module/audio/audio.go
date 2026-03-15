// Package audio provides zero-allocation audio generation and processing.
// All operations write directly to arena-backed vecf32 fields — zero GC.
package audio

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/GoMemPipe/mempipe/module"
	"github.com/GoMemPipe/mempipe/runtime"
)

// AudioModule provides zero-allocation audio generation and DSP.
type AudioModule struct {
	// Attached region for zero-copy output
	attachedView  *runtime.Region
	attachedField string

	// Ring buffer for audio generation (pre-allocated)
	ringBuf []float32

	// Generator state (no allocations)
	phase    float64 // Phase for sine wave generation
	phaseInc float64 // Phase increment per sample

	// Deterministic PRNG state
	prngState uint64

	// Sample rate
	sampleRate float64
}

// NewAudioModule creates a new audio module with deterministic PRNG.
//
//mem:allow(init) - One-time allocation during initialization
func NewAudioModule(sampleRate float64, seed uint64) *AudioModule {
	return &AudioModule{
		ringBuf:    make([]float32, 4096),
		phase:      0.0,
		phaseInc:   0.0,
		prngState:  seed,
		sampleRate: sampleRate,
	}
}

// --- Module interface ---

func (m *AudioModule) Name() string { return "audio" }
func (m *AudioModule) Init() error  { return nil }

// Attach binds the audio module to a vecf32 field in a region.
//
//mem:hot
func (m *AudioModule) Attach(view *runtime.Region, field string) error {
	ftype, err := view.FieldType(field)
	if err != nil {
		return fmt.Errorf("field not found: %w", err)
	}
	if ftype != runtime.TypeVecF32 {
		return fmt.Errorf("field %s is not vecf32 (got type %d)", field, ftype)
	}
	m.attachedView = view
	m.attachedField = field
	return nil
}

// SetFrequency sets the sine wave frequency.
//
//mem:hot
//mem:nogc
func (m *AudioModule) SetFrequency(freq float64) {
	m.phaseInc = 2.0 * math.Pi * freq / m.sampleRate
}

// GenSine generates a sine wave directly into the attached region.
//
//mem:hot
//mem:nogc
func (m *AudioModule) GenSine(count int) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		sample := float32(math.Sin(m.phase))
		elemPtr := unsafe.Add(ptr, i*4)
		*(*float32)(elemPtr) = sample
		m.phase += m.phaseInc
		if m.phase >= 2.0*math.Pi {
			m.phase -= 2.0 * math.Pi
		}
	}
	return count, nil
}

// GenNoise generates deterministic white noise into the attached region.
//
//mem:hot
//mem:nogc
func (m *AudioModule) GenNoise(count int, amplitude float64) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		m.prngState ^= m.prngState << 13
		m.prngState ^= m.prngState >> 7
		m.prngState ^= m.prngState << 17
		normalized := float64(m.prngState)/float64(^uint64(0))*2.0 - 1.0
		sample := float32(normalized * amplitude)
		elemPtr := unsafe.Add(ptr, i*4)
		*(*float32)(elemPtr) = sample
	}
	return count, nil
}

// GenSilence fills the attached region with zeros.
//
//mem:hot
//mem:nogc
func (m *AudioModule) GenSilence(count int) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		*(*float32)(elemPtr) = 0.0
	}
	return count, nil
}

// PullFramePtr returns (pointer, length, error) to the attached audio buffer.
//
//mem:hot
//mem:nogc
func (m *AudioModule) PullFramePtr() (unsafe.Pointer, int, error) {
	if m.attachedView == nil {
		return nil, 0, fmt.Errorf("no region attached - call Attach() first")
	}
	return m.attachedView.VecF32Ptr(m.attachedField)
}

// ReadSamples copies samples from the region into dst.
//
//mem:allow(convenience) - Use PullFramePtr in hot paths
func (m *AudioModule) ReadSamples(dst []float32) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	count := len(dst)
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		dst[i] = *(*float32)(elemPtr)
	}
	return count, nil
}

// WriteSamples copies samples from src into the region.
//
//mem:hot (if src is pre-allocated)
func (m *AudioModule) WriteSamples(src []float32) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	count := len(src)
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		*(*float32)(elemPtr) = src[i]
	}
	return count, nil
}

// --- DSP Operators (zero-alloc, operate in-place on region memory) ---

// LowPassFilter applies a first-order IIR low-pass filter in-place.
// cutoff is the cutoff frequency in Hz.
//
//mem:hot
//mem:nogc
func (m *AudioModule) LowPassFilter(count int, cutoff float64) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	// RC constant: α = dt / (RC + dt), dt = 1/sampleRate, RC = 1/(2π·cutoff)
	dt := 1.0 / m.sampleRate
	rc := 1.0 / (2.0 * math.Pi * cutoff)
	alpha := float32(dt / (rc + dt))

	prev := float32(0.0)
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)
		prev = prev + alpha*(sample-prev)
		*(*float32)(elemPtr) = prev
	}
	return count, nil
}

// HighPassFilter applies a first-order IIR high-pass filter in-place.
// cutoff is the cutoff frequency in Hz.
//
//mem:hot
//mem:nogc
func (m *AudioModule) HighPassFilter(count int, cutoff float64) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	dt := 1.0 / m.sampleRate
	rc := 1.0 / (2.0 * math.Pi * cutoff)
	alpha := float32(rc / (rc + dt))

	var prevIn, prevOut float32
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)
		out := alpha * (prevOut + sample - prevIn)
		prevIn = sample
		prevOut = out
		*(*float32)(elemPtr) = out
	}
	return count, nil
}

// Gain multiplies all samples by a constant factor in-place.
//
//mem:hot
//mem:nogc
func (m *AudioModule) Gain(count int, factor float32) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	ptr, capacity, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	if count > capacity {
		count = capacity
	}
	for i := 0; i < count; i++ {
		elemPtr := unsafe.Add(ptr, i*4)
		sample := *(*float32)(elemPtr)
		*(*float32)(elemPtr) = sample * factor
	}
	return count, nil
}

// Mix adds samples from srcRegion/srcField into the attached region,
// weighted by w. dst[i] = dst[i]*(1-w) + src[i]*w.
//
//mem:hot
//mem:nogc
func (m *AudioModule) Mix(count int, srcRegion *runtime.Region, srcField string, w float32) (int, error) {
	if m.attachedView == nil {
		return 0, fmt.Errorf("no region attached - call Attach() first")
	}
	dstPtr, dstCap, err := m.attachedView.VecF32Ptr(m.attachedField)
	if err != nil {
		return 0, err
	}
	srcPtr, srcCap, err := srcRegion.VecF32Ptr(srcField)
	if err != nil {
		return 0, err
	}
	if count > dstCap {
		count = dstCap
	}
	if count > srcCap {
		count = srcCap
	}
	oneMinusW := 1.0 - w
	for i := 0; i < count; i++ {
		dElem := unsafe.Add(dstPtr, i*4)
		sElem := unsafe.Add(srcPtr, i*4)
		d := *(*float32)(dElem)
		s := *(*float32)(sElem)
		*(*float32)(dElem) = d*oneMinusW + s*w
	}
	return count, nil
}

// --- State management ---

// ResetPhase resets the sine wave phase to zero.
func (m *AudioModule) ResetPhase()           { m.phase = 0.0 }
func (m *AudioModule) ResetPRNG(seed uint64) { m.prngState = seed }
func (m *AudioModule) GetPhase() float64     { return m.phase }
func (m *AudioModule) GetPRNGState() uint64  { return m.prngState }

func init() {
	// Register a default audio module; users should create their own
	// with specific sample rate and seed for production use.
	module.Register(NewAudioModule(44100, 0))
}
