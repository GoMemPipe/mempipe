// Package runtime provides the scheduler for cell execution
package runtime

import (
	"sync/atomic"
)

// RuntimeClock provides a deterministic, monotonic tick counter
// This replaces OS-level time with an internal, memory-based clock
// that is fully deterministic and replayable
type RuntimeClock struct {
	ticks      uint64 // Current tick count (atomic for thread-safety)
	ticksPerMs uint64 // How many ticks equal 1 millisecond (default: 1)
	mu         rwMutex
	tickRate   uint64 // Configurable tick rate
}

// NewRuntimeClock creates a new deterministic clock
// ticksPerMs defines how many ticks equal 1 millisecond (default: 1 tick = 1ms)
func NewRuntimeClock(ticksPerMs uint64) *RuntimeClock {
	if ticksPerMs == 0 {
		ticksPerMs = 1 // Default: 1 tick = 1 millisecond
	}
	return &RuntimeClock{
		ticks:      0,
		ticksPerMs: ticksPerMs,
		tickRate:   ticksPerMs,
	}
}

// Tick increments the clock by one tick
// This is called by the scheduler on each execution cycle
func (c *RuntimeClock) Tick() uint64 {
	return atomic.AddUint64(&c.ticks, 1)
}

// TickBy increments the clock by n ticks
func (c *RuntimeClock) TickBy(n uint64) uint64 {
	return atomic.AddUint64(&c.ticks, n)
}

// Now returns the current tick count
func (c *RuntimeClock) Now() uint64 {
	return atomic.LoadUint64(&c.ticks)
}

// NowMs returns the current time in milliseconds
func (c *RuntimeClock) NowMs() uint64 {
	ticks := atomic.LoadUint64(&c.ticks)
	return ticks / c.ticksPerMs
}

// Since returns the number of ticks elapsed since the given tick
func (c *RuntimeClock) Since(then uint64) uint64 {
	now := atomic.LoadUint64(&c.ticks)
	if now < then {
		return 0
	}
	return now - then
}

// SinceMs returns the number of milliseconds elapsed since the given tick
func (c *RuntimeClock) SinceMs(then uint64) uint64 {
	return c.Since(then) / c.ticksPerMs
}

// Reset resets the clock to zero (useful for testing)
func (c *RuntimeClock) Reset() {
	atomic.StoreUint64(&c.ticks, 0)
}

// SetTicks sets the clock to a specific tick value (useful for deterministic replay)
func (c *RuntimeClock) SetTicks(ticks uint64) {
	atomic.StoreUint64(&c.ticks, ticks)
}

// TicksPerMs returns the number of ticks per millisecond
func (c *RuntimeClock) TicksPerMs() uint64 {
	return c.ticksPerMs
}

// MsToTicks converts milliseconds to ticks
func (c *RuntimeClock) MsToTicks(ms uint64) uint64 {
	return ms * c.ticksPerMs
}

// TicksToMs converts ticks to milliseconds
func (c *RuntimeClock) TicksToMs(ticks uint64) uint64 {
	return ticks / c.ticksPerMs
}
