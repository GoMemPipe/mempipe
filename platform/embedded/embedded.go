// Package embedded provides TinyGo-compatible stubs for MemPipe.
//
// Build with: go build -tags embedded  (or tinygo build -tags embedded)
//
// This package provides no-op implementations for features that are
// unavailable on bare-metal embedded targets:
//   - HTTP networking (no network stack)
//   - Filesystem I/O (no OS filesystem)
//   - OS signals (no OS)
//   - Logging to stdout (uses memory-only logging)
//
// The core MemPipe functionality — arena memory, regions, inference engine —
// works without this package. This package only provides safe fallbacks
// when module code attempts platform-specific operations.
//
// TinyGo compatibility notes:
//   - sync.RWMutex: supported in TinyGo 0.28+ (single-threaded fallback)
//   - reflect: limited in TinyGo — TypedRegion/LayoutFromStruct won't work
//     on embedded; use manual LayoutTable construction instead
//   - unsafe.Pointer: fully supported
//   - encoding/binary: supported
//   - os.ReadFile: not available — use LoadModelFromBytes instead of LoadModel
package embedded

import "errors"

// ────────────────────────────────────────────────────────────────────────────
// Stub errors
// ────────────────────────────────────────────────────────────────────────────

var (
	// ErrNotSupported is returned by operations not available on embedded targets.
	ErrNotSupported = errors.New("operation not supported on embedded target")

	// ErrNoFilesystem is returned when file I/O is attempted.
	ErrNoFilesystem = errors.New("filesystem not available on embedded target")

	// ErrNoNetwork is returned when network operations are attempted.
	ErrNoNetwork = errors.New("network not available on embedded target")
)

// ────────────────────────────────────────────────────────────────────────────
// HTTP stubs
// ────────────────────────────────────────────────────────────────────────────

// HTTPGet is a no-op stub. Returns ErrNoNetwork.
func HTTPGet(_ string) ([]byte, error) {
	return nil, ErrNoNetwork
}

// HTTPPost is a no-op stub. Returns ErrNoNetwork.
func HTTPPost(_, _ string, _ []byte) ([]byte, error) {
	return nil, ErrNoNetwork
}

// ────────────────────────────────────────────────────────────────────────────
// Filesystem stubs
// ────────────────────────────────────────────────────────────────────────────

// ReadFile is a no-op stub. Returns ErrNoFilesystem.
func ReadFile(_ string) ([]byte, error) {
	return nil, ErrNoFilesystem
}

// WriteFile is a no-op stub. Returns ErrNoFilesystem.
func WriteFile(_ string, _ []byte) error {
	return ErrNoFilesystem
}

// ────────────────────────────────────────────────────────────────────────────
// Signal stubs
// ────────────────────────────────────────────────────────────────────────────

// WaitForShutdown is a no-op on embedded. Returns immediately.
func WaitForShutdown() {}

// ────────────────────────────────────────────────────────────────────────────
// Logging stubs
// ────────────────────────────────────────────────────────────────────────────

// LogBuffer is a fixed-size ring buffer for embedded logging.
// On embedded targets without stdout, log messages are stored here
// and can be read back via JTAG/SWD or a debug probe.
type LogBuffer struct {
	buf  [4096]byte
	head int
	size int
}

var globalLog LogBuffer

// Log writes a message to the embedded log buffer.
func Log(msg string) {
	for i := 0; i < len(msg); i++ {
		globalLog.buf[globalLog.head] = msg[i]
		globalLog.head = (globalLog.head + 1) % len(globalLog.buf)
		if globalLog.size < len(globalLog.buf) {
			globalLog.size++
		}
	}
}

// LogBytes returns the current log buffer contents.
func LogBytes() []byte {
	if globalLog.size == 0 {
		return nil
	}
	if globalLog.size < len(globalLog.buf) {
		return globalLog.buf[:globalLog.size]
	}
	// Ring buffer wrapped — return in order
	start := globalLog.head
	out := make([]byte, len(globalLog.buf))
	for i := range out {
		out[i] = globalLog.buf[(start+i)%len(globalLog.buf)]
	}
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// TinyGo compatibility helpers
// ────────────────────────────────────────────────────────────────────────────

// NopMutex is a no-op mutex for single-threaded embedded environments.
// Use this as a drop-in replacement for sync.RWMutex when needed.
type NopMutex struct{}

func (NopMutex) Lock()    {}
func (NopMutex) Unlock()  {}
func (NopMutex) RLock()   {}
func (NopMutex) RUnlock() {}
