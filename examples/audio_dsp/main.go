// Command audio_dsp demonstrates real-time audio DSP with MemPipe pipelines.
//
// It builds a pipeline that processes audio in 128-sample frames:
//
//	oscillator → low-pass filter → gain → output buffer
//
// All processing happens in arena memory with zero allocations per frame.
// Includes a WASM-compatible design for Web Audio AudioWorklet integration.
//
// Usage:
//
//	go run ./examples/audio_dsp
package main

import (
	"fmt"
	"math"
	"time"

	mempipe "github.com/GoMemPipe/mempipe"
)

const (
	sampleRate = 44100
	blockSize  = 128 // Web Audio renders in 128-sample blocks
	frequency  = 440.0
	filterCut  = 2000.0
	gainLevel  = 0.8
)

// ── Region structs (arena-backed, zero-alloc access) ──

// OscState holds the oscillator's persistent state.
type OscState struct {
	Phase    float32 `mempipe:"field:phase"`
	FreqHz   float32 `mempipe:"field:freq_hz"`
	PhaseInc float32 `mempipe:"field:phase_inc"`
}

// FilterState holds the 1-pole low-pass filter state.
type FilterState struct {
	Coeff float32 `mempipe:"field:coeff"` // filter coefficient
	Prev  float32 `mempipe:"field:prev"`  // previous output sample
}

// GainState holds the output gain level.
type GainState struct {
	Level float32 `mempipe:"field:level"`
}

// AudioBlock holds a block of audio samples.
// We store samples as a fixed array of float32 fields.
// For simplicity, we use a summary struct with RMS and peak.
type AudioBlock struct {
	RMS      float32 `mempipe:"field:rms"`
	Peak     float32 `mempipe:"field:peak"`
	FrameNum uint32  `mempipe:"field:frame_num"`
}

func main() {
	fmt.Println("── MemPipe Audio DSP Example ──")
	fmt.Printf("Sample rate: %d Hz, Block size: %d samples\n", sampleRate, blockSize)
	fmt.Printf("Oscillator: %.0f Hz sine, Filter cutoff: %.0f Hz, Gain: %.1f\n", frequency, filterCut, gainLevel)
	fmt.Println()

	// ── Build pipeline ──
	pipe := mempipe.NewPipeline()

	osc := mempipe.AddRegion[OscState](pipe, "oscillator")
	filt := mempipe.AddRegion[FilterState](pipe, "filter")
	gain := mempipe.AddRegion[GainState](pipe, "gain")
	output := mempipe.AddRegion[AudioBlock](pipe, "output")

	// Working buffers (not in arena — we use local arrays for per-sample processing)
	// In a real WASM deployment, these would be memory-mapped views.
	var oscBuf [blockSize]float32
	var filtBuf [blockSize]float32
	var outBuf [blockSize]float32

	// ── Cell 1: Oscillator (source — no input dependencies) ──
	// Generates a sine wave, writing samples into oscBuf.
	pipe.Cell("oscillator", func() {
		s := osc.Get()
		if s.FreqHz == 0 {
			s.FreqHz = frequency
			s.PhaseInc = float32(2.0 * math.Pi * frequency / sampleRate)
		}
		for i := 0; i < blockSize; i++ {
			oscBuf[i] = float32(math.Sin(float64(s.Phase)))
			s.Phase += s.PhaseInc
			if s.Phase > 2*math.Pi {
				s.Phase -= 2 * math.Pi
			}
		}
		osc.Set(s)
	}, nil, []string{"oscillator"})

	// ── Cell 2: Low-pass filter ──
	// 1-pole IIR: y[n] = coeff * x[n] + (1-coeff) * y[n-1]
	pipe.Cell("filter", func() {
		f := filt.Get()
		if f.Coeff == 0 {
			// Compute coefficient from cutoff frequency
			dt := 1.0 / float64(sampleRate)
			rc := 1.0 / (2.0 * math.Pi * filterCut)
			f.Coeff = float32(dt / (rc + dt))
		}
		for i := 0; i < blockSize; i++ {
			f.Prev = f.Coeff*oscBuf[i] + (1-f.Coeff)*f.Prev
			filtBuf[i] = f.Prev
		}
		filt.Set(f)
	}, []string{"oscillator"}, []string{"filter"})

	// ── Cell 3: Gain ──
	pipe.Cell("gain", func() {
		g := gain.Get()
		if g.Level == 0 {
			g.Level = gainLevel
			gain.Set(g)
		}
		for i := 0; i < blockSize; i++ {
			outBuf[i] = filtBuf[i] * g.Level
		}
	}, []string{"filter"}, []string{"gain"})

	// ── Cell 4: Output metering ──
	pipe.Cell("output", func() {
		o := output.Get()
		o.FrameNum++
		var sumSq float32
		o.Peak = 0
		for i := 0; i < blockSize; i++ {
			s := outBuf[i]
			sumSq += s * s
			if s < 0 {
				s = -s
			}
			if s > o.Peak {
				o.Peak = s
			}
		}
		o.RMS = float32(math.Sqrt(float64(sumSq / blockSize)))
		output.Set(o)
	}, []string{"gain"}, []string{"output"})

	// ── Run pipeline ──
	// Simulate 1 second of audio (44100/128 ≈ 345 frames)
	numFrames := sampleRate / blockSize
	fmt.Printf("Processing %d frames (%.2f seconds)...\n\n", numFrames, float64(numFrames*blockSize)/sampleRate)

	start := time.Now()
	pipe.OnIteration(func(iter int) {
		if iter > 0 && iter%(numFrames/5) == 0 {
			o := output.Get()
			fmt.Printf("  Frame %4d: RMS=%.4f  Peak=%.4f\n", o.FrameNum, o.RMS, o.Peak)
		}
	})

	if err := pipe.Run(numFrames); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	elapsed := time.Since(start)
	o := output.Get()
	fmt.Printf("\n  Final frame %d: RMS=%.4f  Peak=%.4f\n", o.FrameNum, o.RMS, o.Peak)
	fmt.Println()

	// ── Performance report ──
	audioSec := float64(numFrames*blockSize) / sampleRate
	fmt.Println("── Performance ──")
	fmt.Printf("Processed %.2f sec of audio in %v\n", audioSec, elapsed)
	fmt.Printf("Real-time factor: %.1fx\n", audioSec/elapsed.Seconds())
	fmt.Printf("Per-frame latency: %.2f µs (budget: %.0f µs)\n",
		float64(elapsed.Microseconds())/float64(numFrames),
		float64(blockSize)/sampleRate*1e6)

	// Print a few output samples for verification
	fmt.Println()
	fmt.Println("── Last block samples (first 8) ──")
	for i := 0; i < 8 && i < blockSize; i++ {
		bar := ""
		normalized := (outBuf[i] + 1.0) / 2.0
		for j := 0; j < int(normalized*30); j++ {
			bar += "█"
		}
		fmt.Printf("  [%3d] %+.4f %s\n", i, outBuf[i], bar)
	}
}
