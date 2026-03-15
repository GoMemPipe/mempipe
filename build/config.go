// Package build provides build configuration, version info, and
// compile-time feature detection for MemPipe.
//
// Platform-specific capabilities are determined at compile time via build tags
// rather than runtime string matching. Use the exported functions:
//
//	build.Platform()      → "native", "wasm", or "embedded"
//	build.HasHTTP()       → true on native and wasm
//	build.HasFilesystem() → true only on native
//	build.HasConcurrency()→ true only on native
//
// Build with:
//
//	(default)                      → native
//	GOOS=js GOARCH=wasm            → wasm
//	-tags embedded                 → embedded (TinyGo)
package build

// Version information
const (
	Version     = "0.1.0"
	ProjectName = "MemPipe"
	Description = "Memory-Level Data Pipeline Language"
)
