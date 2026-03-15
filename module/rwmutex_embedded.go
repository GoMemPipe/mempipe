//go:build embedded

package module

// rwMutex is a no-op mutex for single-threaded embedded/TinyGo targets.
// Eliminates sync overhead and the risk of hardware deadlocks in ISR
// contexts on bare-metal microcontrollers.
type rwMutex struct{}

func (rwMutex) Lock()    {}
func (rwMutex) Unlock()  {}
func (rwMutex) RLock()   {}
func (rwMutex) RUnlock() {}
