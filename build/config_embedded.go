//go:build embedded && !js

package build

// Platform returns the current build platform.
func Platform() string { return "embedded" }

// HasHTTP reports whether this build supports HTTP networking.
func HasHTTP() bool { return false }

// HasFilesystem reports whether this build has filesystem access.
func HasFilesystem() bool { return false }

// HasConcurrency reports whether goroutines and sync primitives are fully available.
// Embedded/TinyGo has limited concurrency support.
func HasConcurrency() bool { return false }

// HasSignalHandling reports whether OS signals are available.
func HasSignalHandling() bool { return false }

// HasNetworkIO reports whether raw network I/O is available.
func HasNetworkIO() bool { return false }

// HasOSInteraction reports whether OS-level APIs are available.
func HasOSInteraction() bool { return false }
