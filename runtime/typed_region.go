// Package runtime provides TypedRegion[T] — a generic, zero-allocation
// accessor that maps a Go struct to arena memory via struct tags.
package runtime

import (
	"fmt"
	"reflect"
	"unsafe"
)

// fieldMapping describes how a single Go struct field maps to arena memory.
type fieldMapping struct {
	structOffset uintptr // offset within Go struct
	arenaOffset  int     // offset within arena region
	size         int     // field size in bytes
}

// TypedRegion provides generic read/write of an entire Go struct to/from
// a named region in the arena.  The mapping between struct fields and arena
// offsets is computed once at construction time via LayoutFromStruct.
//
// All hot-path methods (Get, Set) are zero-allocation.
type TypedRegion[T any] struct {
	region   *Region
	layout   *RegionLayout
	mappings []fieldMapping
}

// NewTypedRegion creates a TypedRegion[T] backed by the named region in the
// given arena.  It calls LayoutFromStruct once to compute the field mapping
// (results are cached per type).
//
//mem:allow(build_time) — reflection and allocation happen here, not on hot path.
func NewTypedRegion[T any](arena *RegionArena, name string) (*TypedRegion[T], error) {
	var zero T
	layout, err := LayoutFromStruct(name, zero)
	if err != nil {
		return nil, fmt.Errorf("TypedRegion[%s]: %w", name, err)
	}

	region, err := arena.Region(name)
	if err != nil {
		return nil, fmt.Errorf("TypedRegion[%s]: %w", name, err)
	}

	// Build struct field → arena offset mappings.
	t := reflect.TypeOf(zero)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	mappings := make([]fieldMapping, 0, len(layout.Fields))
	for _, fl := range layout.Fields {
		// Find the corresponding Go struct field.
		sf, found := findStructField(t, fl.Name)
		if !found {
			return nil, fmt.Errorf("TypedRegion[%s]: struct has no field matching %q", name, fl.Name)
		}
		mappings = append(mappings, fieldMapping{
			structOffset: sf.Offset,
			arenaOffset:  fl.Offset,
			size:         fl.Size,
		})
	}

	return &TypedRegion[T]{
		region:   region,
		layout:   layout,
		mappings: mappings,
	}, nil
}

// Get reads the entire struct from arena memory.  Zero allocations.
//
//mem:hot
//mem:nogc
func (r *TypedRegion[T]) Get() T {
	var v T
	dst := unsafe.Pointer(&v)
	for i := range r.mappings {
		m := &r.mappings[i]
		src := unsafe.Add(r.region.base, m.arenaOffset)
		copyN(unsafe.Add(dst, m.structOffset), src, m.size)
	}
	return v
}

// Set writes the entire struct into arena memory.  Zero allocations.
//
//mem:hot
//mem:nogc
func (r *TypedRegion[T]) Set(v T) {
	src := unsafe.Pointer(&v)
	for i := range r.mappings {
		m := &r.mappings[i]
		dst := unsafe.Add(r.region.base, m.arenaOffset)
		copyN(dst, unsafe.Add(src, m.structOffset), m.size)
	}
}

// Region returns the underlying Region for field-level access.
func (r *TypedRegion[T]) Region() *Region {
	return r.region
}

// Layout returns the computed layout.
func (r *TypedRegion[T]) Layout() *RegionLayout {
	return r.layout
}

// copyN copies n bytes from src to dst using sized loads where possible.
// This is the inner loop on the hot path — no allocations.
//
//mem:hot
//mem:nogc
func copyN(dst, src unsafe.Pointer, n int) {
	switch n {
	case 1:
		*(*uint8)(dst) = *(*uint8)(src)
	case 2:
		*(*uint16)(dst) = *(*uint16)(src)
	case 4:
		*(*uint32)(dst) = *(*uint32)(src)
	case 8:
		*(*uint64)(dst) = *(*uint64)(src)
	default:
		// Fallback for larger / odd sizes (e.g. vecf32)
		d := (*[1 << 30]byte)(dst)
		s := (*[1 << 30]byte)(src)
		for i := 0; i < n; i++ {
			d[i] = s[i]
		}
	}
}

// findStructField looks up a struct field by its mempipe tag name (lowercased
// Go field name or explicit field:name tag).
func findStructField(t reflect.Type, mempipeName string) (reflect.StructField, bool) {
	for i := 0; i < t.NumField(); i++ {
		sf := t.Field(i)
		tag := sf.Tag.Get("mempipe")
		if tag == "" || tag == "-" {
			continue
		}
		// Extract the field:name from the tag
		name := extractFieldName(tag, sf.Name)
		if name == mempipeName {
			return sf, true
		}
	}
	return reflect.StructField{}, false
}

// extractFieldName returns the mempipe field name from a tag string.
// Falls back to lowercased Go field name if no explicit field:name is present.
func extractFieldName(tag, goName string) string {
	for _, part := range splitTag(tag) {
		if len(part) > 6 && part[:6] == "field:" {
			return part[6:]
		}
	}
	// Default: lowercase Go name
	return toLower(goName)
}

// splitTag splits a comma-separated tag string.
func splitTag(tag string) []string {
	parts := make([]string, 0, 4)
	start := 0
	for i := 0; i < len(tag); i++ {
		if tag[i] == ',' {
			parts = append(parts, trimSpace(tag[start:i]))
			start = i + 1
		}
	}
	parts = append(parts, trimSpace(tag[start:]))
	return parts
}

// toLower converts ASCII to lowercase without importing strings.
func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

// trimSpace trims leading/trailing ASCII spaces.
func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && s[start] == ' ' {
		start++
	}
	for end > start && s[end-1] == ' ' {
		end--
	}
	return s[start:end]
}
