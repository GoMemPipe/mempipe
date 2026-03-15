// Package runtime provides the scheduler for cell execution
package runtime

import (
	"fmt"
	"time"
)

// LogLevel defines the severity of a log message
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
)

// String returns the string representation of a log level
func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// LogEntry represents a single log entry
type LogEntry struct {
	Tick    uint64
	Level   LogLevel
	Message string
}

// MemoryLogger is a memory-backed logger that stores logs instead of printing
// This replaces the standard log package with deterministic, in-memory logging
type MemoryLogger struct {
	entries  []LogEntry
	maxSize  int
	clock    *RuntimeClock
	mu       rwMutex
	minLevel LogLevel
	enabled  bool
}

// NewMemoryLogger creates a new memory-backed logger
func NewMemoryLogger(clock *RuntimeClock, maxSize int) *MemoryLogger {
	if maxSize <= 0 {
		maxSize = 10000 // Default: store last 10k log entries
	}
	return &MemoryLogger{
		entries:  make([]LogEntry, 0, maxSize),
		maxSize:  maxSize,
		clock:    clock,
		minLevel: LogLevelInfo,
		enabled:  true,
	}
}

// SetLevel sets the minimum log level
func (l *MemoryLogger) SetLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.minLevel = level
}

// Enable enables logging
func (l *MemoryLogger) Enable() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.enabled = true
}

// Disable disables logging
func (l *MemoryLogger) Disable() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.enabled = false
}

// log adds a log entry
func (l *MemoryLogger) log(level LogLevel, format string, args ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.enabled || level < l.minLevel {
		return
	}

	message := fmt.Sprintf(format, args...)
	tick := uint64(0)
	if l.clock != nil {
		tick = l.clock.Now()
	}

	entry := LogEntry{
		Tick:    tick,
		Level:   level,
		Message: message,
	}

	// Add entry and maintain size limit
	l.entries = append(l.entries, entry)
	if len(l.entries) > l.maxSize {
		// Remove oldest entries
		copy(l.entries, l.entries[len(l.entries)-l.maxSize:])
		l.entries = l.entries[:l.maxSize]
	}
}

// Debug logs a debug message
func (l *MemoryLogger) Debug(format string, args ...interface{}) {
	l.log(LogLevelDebug, format, args...)
}

// Info logs an info message
func (l *MemoryLogger) Info(format string, args ...interface{}) {
	l.log(LogLevelInfo, format, args...)
}

// Warn logs a warning message
func (l *MemoryLogger) Warn(format string, args ...interface{}) {
	l.log(LogLevelWarn, format, args...)
}

// Error logs an error message
func (l *MemoryLogger) Error(format string, args ...interface{}) {
	l.log(LogLevelError, format, args...)
}

// Printf logs an info message (for compatibility with standard log)
func (l *MemoryLogger) Printf(format string, args ...interface{}) {
	l.Info(format, args...)
}

// Println logs an info message (for compatibility with standard log)
func (l *MemoryLogger) Println(args ...interface{}) {
	l.Info(fmt.Sprint(args...))
}

// GetEntries returns all log entries
func (l *MemoryLogger) GetEntries() []LogEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make([]LogEntry, len(l.entries))
	copy(result, l.entries)
	return result
}

// GetEntriesSince returns log entries since the given tick
func (l *MemoryLogger) GetEntriesSince(tick uint64) []LogEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make([]LogEntry, 0)
	for _, entry := range l.entries {
		if entry.Tick >= tick {
			result = append(result, entry)
		}
	}
	return result
}

// GetLastN returns the last N log entries
func (l *MemoryLogger) GetLastN(n int) []LogEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if n <= 0 || len(l.entries) == 0 {
		return []LogEntry{}
	}

	if n >= len(l.entries) {
		result := make([]LogEntry, len(l.entries))
		copy(result, l.entries)
		return result
	}

	result := make([]LogEntry, n)
	copy(result, l.entries[len(l.entries)-n:])
	return result
}

// Clear clears all log entries
func (l *MemoryLogger) Clear() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries = l.entries[:0]
}

// Count returns the number of log entries
func (l *MemoryLogger) Count() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.entries)
}

// Format formats a log entry for display
func (l *MemoryLogger) Format(entry LogEntry) string {
	// Calculate time in ms from ticks
	ms := entry.Tick
	if l.clock != nil {
		ms = l.clock.TicksToMs(entry.Tick)
	}

	return fmt.Sprintf("[%8dms] [%5s] %s", ms, entry.Level.String(), entry.Message)
}

// FormatAll formats all log entries
func (l *MemoryLogger) FormatAll() string {
	entries := l.GetEntries()
	result := ""
	for _, entry := range entries {
		result += l.Format(entry) + "\n"
	}
	return result
}

// DualLogger wraps MemoryLogger and optionally prints to standard logger
// This allows gradual migration and debugging
type DualLogger struct {
	memory    *MemoryLogger
	printToOS bool
}

// NewDualLogger creates a logger that logs to memory and optionally to OS
func NewDualLogger(memory *MemoryLogger, printToOS bool) *DualLogger {
	return &DualLogger{
		memory:    memory,
		printToOS: printToOS,
	}
}

// log logs to memory and optionally to OS
func (l *DualLogger) log(level LogLevel, format string, args ...interface{}) {
	l.memory.log(level, format, args...)

	if l.printToOS {
		// Print to standard logger for debugging
		message := fmt.Sprintf(format, args...)
		timestamp := time.Now().Format("15:04:05.000")
		fmt.Printf("[%s] [%5s] %s\n", timestamp, level.String(), message)
	}
}

// Debug logs a debug message
func (l *DualLogger) Debug(format string, args ...interface{}) {
	l.log(LogLevelDebug, format, args...)
}

// Info logs an info message
func (l *DualLogger) Info(format string, args ...interface{}) {
	l.log(LogLevelInfo, format, args...)
}

// Warn logs a warning message
func (l *DualLogger) Warn(format string, args ...interface{}) {
	l.log(LogLevelWarn, format, args...)
}

// Error logs an error message
func (l *DualLogger) Error(format string, args ...interface{}) {
	l.log(LogLevelError, format, args...)
}

// Printf logs an info message
func (l *DualLogger) Printf(format string, args ...interface{}) {
	l.Info(format, args...)
}

// Println logs an info message
func (l *DualLogger) Println(args ...interface{}) {
	l.Info(fmt.Sprint(args...))
}

// SetPrintToOS enables/disables printing to OS logger
func (l *DualLogger) SetPrintToOS(enabled bool) {
	l.printToOS = enabled
}

// GetMemoryLogger returns the underlying memory logger
func (l *DualLogger) GetMemoryLogger() *MemoryLogger {
	return l.memory
}
