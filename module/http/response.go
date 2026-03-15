// Package http provides platform-specific HTTP client functionality for MemPipe.
// Build tags select the implementation: native (!wasm && !embedded),
// WASM (wasm), or embedded (embedded/no-op).
package http

// Response represents an HTTP response with timing information.
type Response struct {
	StatusCode   int
	Body         string
	Headers      map[string]string
	ResponseTime float64 // milliseconds
	Error        error
}
