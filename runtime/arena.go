// Package runtime provides zero-allocation memory management via RegionArena
package runtime

import (
	"fmt"
	"log"
	"unsafe"
)

// RegionArena manages pre-allocated memory for all regions
// This is the foundation of the zero-GC hot path: one allocation for all regions
type RegionArena struct {
	memory  []byte             // Single pre-allocated slab
	regions map[string]*Region // Region name -> typed region
	offsets map[string]int64   // Region name -> byte offset in slab
	size    int64              // Total arena size (including alignment padding)
}

// ArenaStats contains arena statistics for debugging/monitoring.
type ArenaStats struct {
	TotalSize          int64   // Total slab size including padding
	UsedSize           int64   // Sum of all region sizes (no padding)
	RegionCount        int     // Number of regions
	FragmentationRatio float64 // (Total - Used) / Total
}

// NewArena creates a RegionArena from region specs and a layout table.
// Memory is allocated ONCE for all regions. Each region is aligned to
// CacheLineSize (64 bytes) for SIMD-friendly access.
//
//mem:allow(single_alloc) - This is the ONE allocation for the entire arena
func NewArena(specs []*RegionSpec, layouts *LayoutTable) (*RegionArena, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("no region specs provided")
	}

	if layouts == nil {
		return nil, fmt.Errorf("layout table cannot be nil")
	}

	// Calculate total memory needed with 64-byte alignment per region
	var totalSize int64
	for _, spec := range specs {
		totalSize = alignOffset(totalSize, CacheLineSize)
		totalSize += spec.Size
	}
	// Final alignment so the arena size is a multiple of the cache line
	totalSize = alignOffset(totalSize, CacheLineSize)

	if totalSize == 0 {
		return nil, fmt.Errorf("all regions have zero size")
	}

	// Single allocation for entire arena
	memory := make([]byte, totalSize)
	log.Printf("[ARENA] Allocated %d bytes for %d regions (64-byte aligned)", totalSize, len(specs))

	arena := &RegionArena{
		memory:  memory,
		regions: make(map[string]*Region, len(specs)),
		offsets: make(map[string]int64, len(specs)),
		size:    totalSize,
	}

	// Create typed regions for each region, cache-line aligned
	offset := int64(0)
	for _, spec := range specs {
		offset = alignOffset(offset, CacheLineSize)

		layout, err := layouts.Get(spec.Name)
		if err != nil {
			return nil, fmt.Errorf("layout not found for region %s: %w", spec.Name, err)
		}

		// Validate layout before creating region
		if err := layout.Validate(); err != nil {
			return nil, fmt.Errorf("invalid layout for region %s: %w", spec.Name, err)
		}

		// Create region at this aligned offset in the arena
		basePtr := unsafe.Pointer(&memory[offset])

		region := &Region{
			name:   spec.Name,
			base:   basePtr,
			size:   spec.Size,
			mode:   spec.Mode,
			layout: layout,
		}

		arena.regions[spec.Name] = region
		arena.offsets[spec.Name] = offset
		log.Printf("[ARENA] Region '%s' at offset %d (size=%d, fields=%d, aligned=64)",
			spec.Name, offset, spec.Size, len(layout.Fields))

		offset += spec.Size
	}

	return arena, nil
}

// alignOffset rounds offset up to the given alignment boundary.
func alignOffset(offset int64, alignment int) int64 {
	a := int64(alignment)
	return (offset + a - 1) / a * a
}

// View retrieves a typed view by region name
// This is a hot path function - must be zero-alloc
//
//mem:hot
func (a *RegionArena) Region(name string) (*Region, error) {
	view, exists := a.regions[name]
	if !exists {
		return nil, fmt.Errorf("region not found: %s", name)
	}
	return view, nil
}

// MustView retrieves a view or panics (for use when error is impossible)
// Use this in hot paths where the region is known to exist
//
//mem:hot
func (a *RegionArena) MustRegion(name string) *Region {
	view, exists := a.regions[name]
	if !exists {
		panic(fmt.Sprintf("region not found: %s", name))
	}
	return view
}

// HasView checks if a region view exists
func (a *RegionArena) HasRegion(name string) bool {
	_, exists := a.regions[name]
	return exists
}

// Size returns the total arena size in bytes
func (a *RegionArena) Size() int64 {
	return a.size
}

// RegionCount returns the number of regions in the arena
func (a *RegionArena) RegionCount() int {
	return len(a.regions)
}

// Stats returns arena statistics for debugging/monitoring.
func (a *RegionArena) Stats() ArenaStats {
	var used int64
	for _, r := range a.regions {
		used += r.size
	}
	frag := 0.0
	if a.size > 0 {
		frag = float64(a.size-used) / float64(a.size)
	}
	return ArenaStats{
		TotalSize:          a.size,
		UsedSize:           used,
		RegionCount:        len(a.regions),
		FragmentationRatio: frag,
	}
}

// Grow increases the size of a region by re-allocating the entire slab.
// This is an offline operation — do NOT call during hot-path execution.
//
//mem:allow(grow)
func (a *RegionArena) Grow(name string, newSize int64) error {
	r, ok := a.regions[name]
	if !ok {
		return fmt.Errorf("region not found: %s", name)
	}
	if newSize <= r.size {
		return fmt.Errorf("new size %d must be larger than current %d", newSize, r.size)
	}

	delta := newSize - r.size
	oldMem := a.memory
	newTotal := a.size + delta

	// Align the new total
	newTotal = alignOffset(newTotal, CacheLineSize)

	newMem := make([]byte, newTotal)
	arenaBase := uintptr(unsafe.Pointer(&oldMem[0]))

	// Copy old data and rebuild region pointers, inserting extra space for
	// the grown region.
	dstOff := int64(0)

	// We need deterministic iteration order — use a slice of names.
	type entry struct {
		name   string
		region *Region
		oldOff int64
	}
	entries := make([]entry, 0, len(a.regions))
	for n, reg := range a.regions {
		entries = append(entries, entry{n, reg, int64(uintptr(reg.base) - arenaBase)})
	}
	// Sort by old offset (insertion sort — typically < 20 regions)
	for i := 1; i < len(entries); i++ {
		key := entries[i]
		j := i - 1
		for j >= 0 && entries[j].oldOff > key.oldOff {
			entries[j+1] = entries[j]
			j--
		}
		entries[j+1] = key
	}

	for _, e := range entries {
		dstOff = alignOffset(dstOff, CacheLineSize)
		sz := e.region.size
		if e.name == name {
			sz = newSize
		}
		// Copy old data (up to old region size)
		srcStart := e.oldOff
		copyLen := e.region.size
		if copyLen > int64(len(oldMem))-srcStart {
			copyLen = int64(len(oldMem)) - srcStart
		}
		copy(newMem[dstOff:dstOff+copyLen], oldMem[srcStart:srcStart+copyLen])

		// Update region
		e.region.base = unsafe.Pointer(&newMem[dstOff])
		e.region.size = sz
		a.offsets[e.name] = dstOff

		dstOff += sz
	}

	a.memory = newMem
	a.size = newTotal
	return nil
}

// RegionNames returns a slice of all region names
func (a *RegionArena) RegionNames() []string {
	names := make([]string, 0, len(a.regions))
	for name := range a.regions {
		names = append(names, name)
	}
	return names
}

// DumpStats prints arena statistics
func (a *RegionArena) DumpStats() {
	s := a.Stats()
	log.Printf("[ARENA] Statistics:")
	log.Printf("  Total size: %d bytes", s.TotalSize)
	log.Printf("  Used size: %d bytes", s.UsedSize)
	log.Printf("  Regions: %d", s.RegionCount)
	log.Printf("  Fragmentation: %.2f%%", s.FragmentationRatio*100)
	log.Printf("  Memory address: %p", &a.memory[0])

	for name, region := range a.regions {
		log.Printf("  Region '%s': base=%p, size=%d, fields=%d",
			name, region.base, region.size, len(region.layout.Fields))
	}
}

// Zero clears all memory in the arena (for testing/reset)
func (a *RegionArena) Zero() {
	for i := range a.memory {
		a.memory[i] = 0
	}
}

// ZeroRegion clears memory for a specific region
func (a *RegionArena) ZeroRegion(name string) error {
	view, err := a.Region(name)
	if err != nil {
		return err
	}

	startOffset := a.offsets[name]
	endOffset := startOffset + view.size

	for i := startOffset; i < endOffset; i++ {
		a.memory[i] = 0
	}

	return nil
}

// Validate checks arena consistency
func (a *RegionArena) Validate() error {
	arenaStart := uintptr(unsafe.Pointer(&a.memory[0]))
	arenaEnd := arenaStart + uintptr(a.size)

	for name, region := range a.regions {
		regBase := uintptr(region.base)
		if regBase < arenaStart || regBase >= arenaEnd {
			return fmt.Errorf("region '%s' base pointer outside arena bounds", name)
		}

		regionEnd := regBase + uintptr(region.size)
		if regionEnd > arenaEnd {
			return fmt.Errorf("region '%s' extends beyond arena bounds", name)
		}

		if err := region.Validate(); err != nil {
			return fmt.Errorf("region '%s' validation failed: %w", name, err)
		}
	}

	return nil
}

// Copy copies data from one arena to another (for checkpointing)
// Note: arenas must have identical layouts
func (a *RegionArena) Copy(dest *RegionArena) error {
	if dest.size != a.size {
		return fmt.Errorf("arena size mismatch: src=%d, dest=%d", a.size, dest.size)
	}

	if len(dest.regions) != len(a.regions) {
		return fmt.Errorf("region count mismatch: src=%d, dest=%d", len(a.regions), len(dest.regions))
	}

	// Fast copy entire arena
	copy(dest.memory, a.memory)
	return nil
}

// Snapshot returns a read-only copy of the arena memory (for debugging)
// WARNING: This allocates! Only use in non-hot paths.
//
//mem:allow(debug_only) - This is for debugging, not hot path
func (a *RegionArena) Snapshot() []byte {
	snapshot := make([]byte, len(a.memory))
	copy(snapshot, a.memory)
	return snapshot
}
