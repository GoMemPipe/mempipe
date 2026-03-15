// Package runtime provides the scheduler for cell execution
package runtime

import (
	"fmt"
	"strings"
)

// MemoryPipe is an in-memory ring buffer for storing I/O data
// This replaces OS-level stdout/stderr with pure memory storage
type MemoryPipe struct {
	name     string
	buffer   []byte
	capacity int
	size     int
	head     int // Write position
	tail     int // Read position
	mu       rwMutex
	lines    []string // Store complete lines for easier retrieval
	linesMu  rwMutex
}

// NewMemoryPipe creates a new memory-backed pipe
func NewMemoryPipe(name string, capacity int) *MemoryPipe {
	if capacity <= 0 {
		capacity = 64 * 1024 // Default: 64KB
	}
	return &MemoryPipe{
		name:     name,
		buffer:   make([]byte, capacity),
		capacity: capacity,
		size:     0,
		head:     0,
		tail:     0,
		lines:    make([]string, 0, 256),
	}
}

// Write writes data to the pipe
func (p *MemoryPipe) Write(data string) error {
	return p.WriteBytes([]byte(data))
}

// WriteBytes writes raw bytes to the pipe
func (p *MemoryPipe) WriteBytes(data []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// If data is larger than capacity, only store the tail
	if len(data) >= p.capacity {
		copy(p.buffer, data[len(data)-p.capacity:])
		p.head = 0
		p.tail = 0
		p.size = p.capacity
		return nil
	}

	// Write data to ring buffer
	for _, b := range data {
		p.buffer[p.head] = b
		p.head = (p.head + 1) % p.capacity

		if p.size < p.capacity {
			p.size++
		} else {
			// Buffer is full, advance tail
			p.tail = (p.tail + 1) % p.capacity
		}
	}

	return nil
}

// Writeln writes data with a newline and stores it as a complete line
func (p *MemoryPipe) Writeln(data string) error {
	line := data + "\n"

	// Store in lines array for easy retrieval
	p.linesMu.Lock()
	p.lines = append(p.lines, data)
	p.linesMu.Unlock()

	return p.Write(line)
}

// Read reads all available data from the pipe (non-destructive)
func (p *MemoryPipe) Read() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.size == 0 {
		return ""
	}

	result := make([]byte, p.size)
	if p.tail < p.head || p.size < p.capacity {
		// Contiguous data
		copy(result, p.buffer[p.tail:p.tail+p.size])
	} else {
		// Wrapped data
		firstChunk := p.capacity - p.tail
		copy(result, p.buffer[p.tail:])
		copy(result[firstChunk:], p.buffer[:p.head])
	}

	return string(result)
}

// ReadLines returns all stored lines
func (p *MemoryPipe) ReadLines() []string {
	p.linesMu.RLock()
	defer p.linesMu.RUnlock()

	result := make([]string, len(p.lines))
	copy(result, p.lines)
	return result
}

// ReadLastN returns the last N lines
func (p *MemoryPipe) ReadLastN(n int) []string {
	p.linesMu.RLock()
	defer p.linesMu.RUnlock()

	if n <= 0 || len(p.lines) == 0 {
		return []string{}
	}

	if n >= len(p.lines) {
		result := make([]string, len(p.lines))
		copy(result, p.lines)
		return result
	}

	result := make([]string, n)
	copy(result, p.lines[len(p.lines)-n:])
	return result
}

// Clear clears the pipe
func (p *MemoryPipe) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.size = 0
	p.head = 0
	p.tail = 0

	p.linesMu.Lock()
	p.lines = p.lines[:0]
	p.linesMu.Unlock()
}

// Size returns the current size of data in the pipe
func (p *MemoryPipe) Size() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.size
}

// LineCount returns the number of lines stored
func (p *MemoryPipe) LineCount() int {
	p.linesMu.RLock()
	defer p.linesMu.RUnlock()
	return len(p.lines)
}

// Name returns the pipe name
func (p *MemoryPipe) Name() string {
	return p.name
}

// String returns a string representation of the pipe contents
func (p *MemoryPipe) String() string {
	return p.Read()
}

// PipeManager manages multiple named pipes
type PipeManager struct {
	pipes map[string]*MemoryPipe
	mu    rwMutex
}

// NewPipeManager creates a new pipe manager
func NewPipeManager() *PipeManager {
	return &PipeManager{
		pipes: make(map[string]*MemoryPipe),
	}
}

// CreatePipe creates a new named pipe
func (pm *PipeManager) CreatePipe(name string, capacity int) (*MemoryPipe, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if _, exists := pm.pipes[name]; exists {
		return nil, fmt.Errorf("pipe %s already exists", name)
	}

	pipe := NewMemoryPipe(name, capacity)
	pm.pipes[name] = pipe
	return pipe, nil
}

// GetPipe retrieves a pipe by name
func (pm *PipeManager) GetPipe(name string) (*MemoryPipe, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	pipe, exists := pm.pipes[name]
	if !exists {
		return nil, fmt.Errorf("pipe %s not found", name)
	}
	return pipe, nil
}

// GetOrCreatePipe gets or creates a pipe
func (pm *PipeManager) GetOrCreatePipe(name string, capacity int) *MemoryPipe {
	pipe, err := pm.GetPipe(name)
	if err == nil {
		return pipe
	}

	pipe, _ = pm.CreatePipe(name, capacity)
	return pipe
}

// DeletePipe removes a pipe
func (pm *PipeManager) DeletePipe(name string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if _, exists := pm.pipes[name]; !exists {
		return fmt.Errorf("pipe %s not found", name)
	}

	delete(pm.pipes, name)
	return nil
}

// ListPipes returns all pipe names
func (pm *PipeManager) ListPipes() []string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	names := make([]string, 0, len(pm.pipes))
	for name := range pm.pipes {
		names = append(names, name)
	}
	return names
}

// DumpAll returns a formatted string of all pipe contents
func (pm *PipeManager) DumpAll() string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	var sb strings.Builder
	for name, pipe := range pm.pipes {
		sb.WriteString(fmt.Sprintf("=== Pipe: %s ===\n", name))
		sb.WriteString(pipe.Read())
		sb.WriteString("\n\n")
	}
	return sb.String()
}

// ClearAll clears all pipes
func (pm *PipeManager) ClearAll() {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	for _, pipe := range pm.pipes {
		pipe.Clear()
	}
}
