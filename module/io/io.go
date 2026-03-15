// Package io provides unified pipe-style I/O for MemPipe.
// All output goes to memory pipes, not OS stdout/stderr.
package io

import (
	"fmt"

	"github.com/GoMemPipe/mempipe/module"
	"github.com/GoMemPipe/mempipe/runtime"
)

// IOModule provides pure memory-based I/O functions.
// All output goes to memory pipes, not OS stdout/stderr.
type IOModule struct {
	pipeManager *runtime.PipeManager
	stdout      *runtime.MemoryPipe
	stderr      *runtime.MemoryPipe
	mu          rwMutex
}

// NewIOModule creates a new IO module with memory-backed pipes.
func NewIOModule(pipeManager *runtime.PipeManager) *IOModule {
	stdout := pipeManager.GetOrCreatePipe("/sys/stdout", 64*1024)
	stderr := pipeManager.GetOrCreatePipe("/sys/stderr", 64*1024)
	return &IOModule{
		pipeManager: pipeManager,
		stdout:      stdout,
		stderr:      stderr,
	}
}

func (m *IOModule) Name() string { return "io" }
func (m *IOModule) Init() error  { return nil }

// --- Typed public methods ---

// Write writes data to stdout without a newline.
func (m *IOModule) Write(data string) error {
	return m.stdout.Write(data)
}

// Writeln writes data to stdout followed by a newline.
func (m *IOModule) Writeln(data string) error {
	return m.stdout.Writeln(data)
}

// Println writes space-separated args to stdout with a newline.
func (m *IOModule) Println(args ...any) error {
	data := fmt.Sprint(args...)
	return m.stdout.Writeln(data)
}

// Printf writes formatted output to stdout.
func (m *IOModule) Printf(format string, args ...any) error {
	data := fmt.Sprintf(format, args...)
	return m.stdout.Write(data)
}

// Eprint writes space-separated args to stderr without a newline.
func (m *IOModule) Eprint(args ...any) error {
	data := fmt.Sprint(args...)
	return m.stderr.Write(data)
}

// Eprintln writes space-separated args to stderr with a newline.
func (m *IOModule) Eprintln(args ...any) error {
	data := fmt.Sprint(args...)
	return m.stderr.Writeln(data)
}

// ReadStdout reads and drains all data from stdout.
func (m *IOModule) ReadStdout() string {
	return m.stdout.Read()
}

// ReadStderr reads and drains all data from stderr.
func (m *IOModule) ReadStderr() string {
	return m.stderr.Read()
}

// ReadPipe reads all data from a named pipe.
func (m *IOModule) ReadPipe(name string) (string, error) {
	pipe, err := m.pipeManager.GetPipe(name)
	if err != nil {
		return "", err
	}
	return pipe.Read(), nil
}

// ReadLines returns all lines from a named pipe (or stdout if empty name).
func (m *IOModule) ReadLines(name string) ([]string, error) {
	if name == "" {
		name = "/sys/stdout"
	}
	pipe, err := m.pipeManager.GetPipe(name)
	if err != nil {
		return nil, err
	}
	return pipe.ReadLines(), nil
}

// CreatePipe creates a named memory pipe with the given capacity.
func (m *IOModule) CreatePipe(name string, capacity int) error {
	if capacity <= 0 {
		capacity = 64 * 1024
	}
	_, err := m.pipeManager.CreatePipe(name, capacity)
	return err
}

// ClosePipe deletes a named pipe.
func (m *IOModule) ClosePipe(name string) error {
	return m.pipeManager.DeletePipe(name)
}

// Shutdown implements module.Shutdowner.
func (m *IOModule) Shutdown() error {
	return nil
}

// --- Accessors ---

// GetStdout returns the stdout pipe.
func (m *IOModule) GetStdout() *runtime.MemoryPipe { return m.stdout }

// GetStderr returns the stderr pipe.
func (m *IOModule) GetStderr() *runtime.MemoryPipe { return m.stderr }

// GetPipeManager returns the pipe manager.
func (m *IOModule) GetPipeManager() *runtime.PipeManager { return m.pipeManager }

// SetPipeManager replaces the pipe manager (used by runtime initialization).
func (m *IOModule) SetPipeManager(pm *runtime.PipeManager) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pipeManager = pm
	m.stdout = pm.GetOrCreatePipe("/sys/stdout", 64*1024)
	m.stderr = pm.GetOrCreatePipe("/sys/stderr", 64*1024)
}

// --- Global singleton ---

var globalIOModule *IOModule

func init() {
	globalIOModule = NewIOModule(runtime.NewPipeManager())
	module.Register(globalIOModule)
}

// GetGlobalIOModule returns the global IO module instance.
func GetGlobalIOModule() *IOModule {
	return globalIOModule
}
