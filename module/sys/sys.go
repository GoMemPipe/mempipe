// Package sys provides system information functions for MemPipe.
// All information is from the MemPipe runtime, not the host OS.
package sys

import (
	"fmt"

	"github.com/GoMemPipe/mempipe/module"
	"github.com/GoMemPipe/mempipe/runtime"
)

const (
	mempipeVersion = "1.0.0-dev"
	mempipeArch    = "bare"
)

// SysModule provides system information functions.
type SysModule struct {
	clock     *runtime.RuntimeClock
	startTick uint64
}

// NewSysModule creates a new sys module.
func NewSysModule(clock *runtime.RuntimeClock) *SysModule {
	var start uint64
	if clock != nil {
		start = clock.Now()
	}
	return &SysModule{clock: clock, startTick: start}
}

func (m *SysModule) Name() string { return "sys" }
func (m *SysModule) Init() error  { return nil }

// --- Typed public methods ---

// Version returns the MemPipe version string.
func (m *SysModule) Version() string { return mempipeVersion }

// Arch returns the architecture string.
func (m *SysModule) Arch() string { return mempipeArch }

// Ticks returns the current tick count. Returns 0 if no clock is set.
func (m *SysModule) Ticks() uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.Now()
}

// UptimeMs returns the uptime in milliseconds. Returns 0 if no clock is set.
func (m *SysModule) UptimeMs() uint64 {
	if m.clock == nil {
		return 0
	}
	return m.clock.TicksToMs(m.clock.Since(m.startTick))
}

// Info returns a human-readable system info string.
func (m *SysModule) Info() string {
	return fmt.Sprintf("MemPipe=%s Arch=%s Uptime=%dms Ticks=%d",
		mempipeVersion, mempipeArch, m.UptimeMs(), m.Ticks())
}

// ArenaStats returns statistics for the given arena.
func (m *SysModule) ArenaStats(arena *runtime.RegionArena) runtime.ArenaStats {
	return arena.Stats()
}

// SetClock updates the clock reference.
func (m *SysModule) SetClock(clock *runtime.RuntimeClock) {
	m.clock = clock
	if clock != nil {
		m.startTick = clock.Now()
	}
}

// --- Global singleton ---

var globalSysModule *SysModule

func init() {
	globalSysModule = NewSysModule(nil)
	module.Register(globalSysModule)
}

// GetGlobalSysModule returns the global sys module instance.
func GetGlobalSysModule() *SysModule { return globalSysModule }
