//go:build !js && !embedded

package build

// Platform returns the current build platform.
func Platform() string { return "native" }

// HasHTTP reports whether this build supports HTTP networking.
func HasHTTP() bool { return true }

// HasFilesystem reports whether this build has filesystem access.
func HasFilesystem() bool { return true }

// HasConcurrency reports whether goroutines and sync primitives are fully available.
func HasConcurrency() bool { return true }

// HasSignalHandling reports whether OS signals are available.
func HasSignalHandling() bool { return true }

// HasNetworkIO reports whether raw network I/O (TCP/UDP) is available.
func HasNetworkIO() bool { return true }

// HasOSInteraction reports whether OS-level APIs (exec, environ, etc.) are available.
func HasOSInteraction() bool { return true }
