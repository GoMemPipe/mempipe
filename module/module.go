// Package module provides the module system for MemPipe.
//
// Modules expose typed Go methods directly — no interface{} boxing.
// The Registry stores modules for discovery and lifecycle management.
package module

import (
	"fmt"
)

// Module is the minimal interface every MemPipe module must satisfy.
// Modules expose functionality via typed Go methods on the concrete type,
// not through a generic function map.
type Module interface {
	// Name returns the module name (e.g. "math", "time", "io").
	Name() string

	// Init is called once when the module is registered.
	Init() error
}

// Ticker is an optional lifecycle interface.
// Modules that implement Ticker have their Tick method called on every
// scheduler tick.
type Ticker interface {
	Module
	Tick(tickCount uint64)
}

// Shutdowner is an optional lifecycle interface.
// Modules that implement Shutdowner have Shutdown called during
// graceful pipeline teardown.
type Shutdowner interface {
	Module
	Shutdown() error
}

// Registry manages all loaded modules. Thread-safe.
type Registry struct {
	mu      rwMutex
	modules map[string]Module
}

var globalRegistry = NewRegistry()

// NewRegistry creates a new module registry.
func NewRegistry() *Registry {
	return &Registry{
		modules: make(map[string]Module),
	}
}

// Register adds a module to the registry and calls Init().
func (r *Registry) Register(m Module) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := m.Name()
	if _, exists := r.modules[name]; exists {
		return fmt.Errorf("module %s is already registered", name)
	}

	if err := m.Init(); err != nil {
		return fmt.Errorf("failed to initialize module %s: %w", name, err)
	}

	r.modules[name] = m
	return nil
}

// Get retrieves a module by name as the base Module interface.
func (r *Registry) Get(name string) (Module, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	m, ok := r.modules[name]
	return m, ok
}

// List returns all registered module names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.modules))
	for name := range r.modules {
		names = append(names, name)
	}
	return names
}

// Tickers returns all modules that implement the Ticker interface.
func (r *Registry) Tickers() []Ticker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var out []Ticker
	for _, m := range r.modules {
		if t, ok := m.(Ticker); ok {
			out = append(out, t)
		}
	}
	return out
}

// Shutdown calls Shutdown on every module that implements Shutdowner.
func (r *Registry) Shutdown() error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, m := range r.modules {
		if s, ok := m.(Shutdowner); ok {
			if err := s.Shutdown(); err != nil {
				return fmt.Errorf("shutdown %s: %w", m.Name(), err)
			}
		}
	}
	return nil
}

// --- Generic helpers (Go 1.18+) ---

// MustGet retrieves a module by name and type-asserts it to T.
// Panics if the module is not found or is not of type T.
func MustGet[T Module](r *Registry, name string) T {
	r.mu.RLock()
	defer r.mu.RUnlock()

	m, ok := r.modules[name]
	if !ok {
		panic(fmt.Sprintf("module %s not found", name))
	}
	typed, ok := m.(T)
	if !ok {
		panic(fmt.Sprintf("module %s is not of requested type", name))
	}
	return typed
}

// Lookup retrieves a module by name and type-asserts it to T.
func Lookup[T Module](r *Registry, name string) (T, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	m, ok := r.modules[name]
	if !ok {
		var zero T
		return zero, false
	}
	typed, ok := m.(T)
	if !ok {
		var zero T
		return zero, false
	}
	return typed, true
}

// --- Global registry accessors ---

// Register adds a module to the global registry.
func Register(m Module) error {
	return globalRegistry.Register(m)
}

// Get retrieves a module from the global registry by name.
func Get(name string) (Module, bool) {
	return globalRegistry.Get(name)
}

// List returns all registered module names from the global registry.
func List() []string {
	return globalRegistry.List()
}

// GetGlobalRegistry returns the global registry.
func GetGlobalRegistry() *Registry {
	return globalRegistry
}
