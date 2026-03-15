// Package time provides deterministic tick-based time for MemPipe.
// Time is based on RuntimeClock ticks, not wall-clock time — fully
// OS-independent and reproducible.
package time

import (
	"github.com/GoMemPipe/mempipe/module"
	"github.com/GoMemPipe/mempipe/runtime"
)

// TimeModule provides deterministic time functions based on tick counts.
type TimeModule struct {
	clock     *runtime.RuntimeClock
	startTick uint64
}

// NewTimeModule creates a new time module.
// Pass nil for clock; it can be set later via SetClock.
func NewTimeModule(clock *runtime.RuntimeClock) *TimeModule {
	var start uint64
	if clock != nil {
		start = clock.Now()
	}
	return &TimeModule{clock: clock, startTick: start}
}

func (m *TimeModule) Name() string { return "time" }
func (m *TimeModule) Init() error  { return nil }

// --- Typed public methods (zero-alloc) ---

// Now returns the current tick count.
func (m *TimeModule) Now() uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.Now()
}

// Ticks is an alias for Now.
func (m *TimeModule) Ticks() uint64 { return m.Now() }

// Ms returns the current time in milliseconds (converted from ticks).
func (m *TimeModule) Ms() uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.NowMs()
}

// Since returns the number of ticks elapsed since the given tick.
func (m *TimeModule) Since(tick uint64) uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.Since(tick)
}

// Elapsed returns the number of ticks since module initialization.
func (m *TimeModule) Elapsed() uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.Since(m.startTick)
}

// Sleep is a cooperative delay placeholder.
// In a pure memory system sleep is a no-op; the scheduler advances time.
func (m *TimeModule) Sleep(ms uint64) {
	if m.clock != nil {
		_ = m.clock.MsToTicks(ms) // future cooperative sleep
	}
}

// SetClock updates the clock reference (used by runtime initialization).
func (m *TimeModule) SetClock(clock *runtime.RuntimeClock) {
	m.clock = clock
	if clock != nil {
		m.startTick = clock.Now()
	}
}

// --- Global singleton ---

var globalTimeModule *TimeModule

func init() {
	globalTimeModule = NewTimeModule(nil)
	module.Register(globalTimeModule)
}

// GetGlobalTimeModule returns the global time module instance.
func GetGlobalTimeModule() *TimeModule {
	return globalTimeModule
}
