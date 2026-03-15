//go:build !embedded

package io

import "sync"

// rwMutex is the real sync.RWMutex on native and WASM targets.
type rwMutex = sync.RWMutex
