//go:build js

package build

// Platform returns the current build platform.
func Platform() string { return "wasm" }

// HasHTTP reports whether this build supports HTTP networking.
// WASM uses JavaScript fetch under the hood.
func HasHTTP() bool { return true }

// HasFilesystem reports whether this build has filesystem access.
// Browser WASM has no filesystem.
func HasFilesystem() bool { return false }

// HasConcurrency reports whether goroutines and sync primitives are fully available.
// WASM has limited goroutine support.
func HasConcurrency() bool { return false }

// HasSignalHandling reports whether OS signals are available.
func HasSignalHandling() bool { return false }

// HasNetworkIO reports whether raw network I/O (TCP/UDP) is available.
// WASM uses JavaScript APIs, not raw sockets.
func HasNetworkIO() bool { return false }

// HasOSInteraction reports whether OS-level APIs are available.
func HasOSInteraction() bool { return false }
