//go:build !embedded

package inference

import "sync"

// rwMutex is the real sync.RWMutex on native and WASM targets.
type rwMutex = sync.RWMutex
