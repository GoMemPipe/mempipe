// Package runtime provides layout information for zero-GC memory access.
// These types were originally in the ir package and are now the canonical
// source of truth for memory layout definitions.
package runtime

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// RegionMode defines how a region behaves
type RegionMode int

const (
	ModeStream RegionMode = iota
	ModeRing
	ModeSlab
	ModeWindowed
	ModeAppend
)

func (m RegionMode) String() string {
	switch m {
	case ModeStream:
		return "stream"
	case ModeRing:
		return "ring"
	case ModeSlab:
		return "slab"
	case ModeWindowed:
		return "windowed"
	case ModeAppend:
		return "append"
	default:
		return "unknown"
	}
}

// ParseMode converts string to RegionMode
func ParseMode(s string) (RegionMode, error) {
	switch s {
	case "stream":
		return ModeStream, nil
	case "ring":
		return ModeRing, nil
	case "slab":
		return ModeSlab, nil
	case "windowed":
		return ModeWindowed, nil
	case "append":
		return ModeAppend, nil
	default:
		return 0, fmt.Errorf("unknown region mode: %s", s)
	}
}

// FieldType represents data types in region headers
type FieldType int

const (
	TypeU8 FieldType = iota
	TypeU16
	TypeU32
	TypeU64
	TypeI8
	TypeI16
	TypeI32
	TypeI64
	TypeF32
	TypeF64
	TypeBool   // Boolean (stored as u8)
	TypeVecF32 // Variable-length float32 array
)

// Size returns the byte size of this field type
func (t FieldType) Size() int {
	switch t {
	case TypeU8, TypeI8, TypeBool:
		return 1
	case TypeU16, TypeI16:
		return 2
	case TypeU32, TypeI32, TypeF32:
		return 4
	case TypeU64, TypeI64, TypeF64:
		return 8
	case TypeVecF32:
		return 0 // Variable length; size must be specified at region declaration
	default:
		return 0
	}
}

// String returns a human-readable representation of field type
func (t FieldType) String() string {
	switch t {
	case TypeU8:
		return "u8"
	case TypeU16:
		return "u16"
	case TypeU32:
		return "u32"
	case TypeU64:
		return "u64"
	case TypeI8:
		return "i8"
	case TypeI16:
		return "i16"
	case TypeI32:
		return "i32"
	case TypeI64:
		return "i64"
	case TypeF32:
		return "f32"
	case TypeF64:
		return "f64"
	case TypeBool:
		return "bool"
	case TypeVecF32:
		return "vecf32"
	default:
		return "unknown"
	}
}

// ParseFieldType converts string to FieldType
func ParseFieldType(s string) (FieldType, error) {
	switch s {
	case "u8":
		return TypeU8, nil
	case "u16":
		return TypeU16, nil
	case "u32":
		return TypeU32, nil
	case "u64":
		return TypeU64, nil
	case "i8":
		return TypeI8, nil
	case "i16":
		return TypeI16, nil
	case "i32":
		return TypeI32, nil
	case "i64":
		return TypeI64, nil
	case "f32":
		return TypeF32, nil
	case "f64":
		return TypeF64, nil
	case "bool":
		return TypeBool, nil
	case "vecf32":
		return TypeVecF32, nil
	default:
		return 0, fmt.Errorf("unknown field type: %s", s)
	}
}

// HeaderField is a single field in a region header
type HeaderField struct {
	Name   string
	Type   FieldType
	Offset int // computed during layout
	Size   int // computed during layout
}

// Header defines the metadata layout of a region
type Header struct {
	Fields []*HeaderField
}

// RegionLayout defines the memory layout of a region with fixed offsets.
// This is computed at pipeline build time and used by the runtime for zero-allocation access.
type RegionLayout struct {
	Name   string        // Region name
	Size   int64         // Total size in bytes
	Mode   RegionMode    // Region mode (stream, ring, etc.)
	Fields []FieldLayout // Field layouts with offsets
}

// FieldLayout defines a single field's type and location within a region
type FieldLayout struct {
	Name   string    // Field name
	Type   FieldType // Field type (u32, f64, vecf32, etc.)
	Offset int       // Byte offset from region base
	Size   int       // Size in bytes
}

// LayoutTable is a collection of all region layouts in a program.
// Used by the arena to create typed views.
type LayoutTable struct {
	Layouts map[string]*RegionLayout // region name -> layout
}

// NewLayoutTable creates an empty layout table
func NewLayoutTable() *LayoutTable {
	return &LayoutTable{
		Layouts: make(map[string]*RegionLayout),
	}
}

// Add adds a region layout to the table
func (lt *LayoutTable) Add(layout *RegionLayout) {
	lt.Layouts[layout.Name] = layout
}

// Get retrieves a layout by region name
func (lt *LayoutTable) Get(name string) (*RegionLayout, error) {
	layout, exists := lt.Layouts[name]
	if !exists {
		return nil, fmt.Errorf("layout not found for region: %s", name)
	}
	return layout, nil
}

// FieldOffset looks up the offset of a field by name
func (rl *RegionLayout) FieldOffset(fieldName string) (int, error) {
	for _, field := range rl.Fields {
		if field.Name == fieldName {
			return field.Offset, nil
		}
	}
	return 0, fmt.Errorf("field %s not found in region %s", fieldName, rl.Name)
}

// FieldType looks up the type of a field by name
func (rl *RegionLayout) FieldType(fieldName string) (FieldType, error) {
	for _, field := range rl.Fields {
		if field.Name == fieldName {
			return field.Type, nil
		}
	}
	return 0, fmt.Errorf("field %s not found in region %s", fieldName, rl.Name)
}

// FieldSize looks up the size of a field by name
func (rl *RegionLayout) FieldSize(fieldName string) (int, error) {
	for _, field := range rl.Fields {
		if field.Name == fieldName {
			return field.Size, nil
		}
	}
	return 0, fmt.Errorf("field %s not found in region %s", fieldName, rl.Name)
}

// HasField checks if a field exists in the layout
func (rl *RegionLayout) HasField(fieldName string) bool {
	for _, field := range rl.Fields {
		if field.Name == fieldName {
			return true
		}
	}
	return false
}

// Validate checks the layout for consistency
func (rl *RegionLayout) Validate() error {
	// Check for duplicate field names
	seen := make(map[string]bool)
	for _, field := range rl.Fields {
		if seen[field.Name] {
			return fmt.Errorf("duplicate field name: %s in region %s", field.Name, rl.Name)
		}
		seen[field.Name] = true
	}

	// Check that offsets are within region size
	for _, field := range rl.Fields {
		if int64(field.Offset+field.Size) > rl.Size {
			return fmt.Errorf("field %s extends beyond region size (offset=%d, size=%d, region_size=%d)",
				field.Name, field.Offset, field.Size, rl.Size)
		}
	}

	// Check for field overlap
	for i := 0; i < len(rl.Fields); i++ {
		for j := i + 1; j < len(rl.Fields); j++ {
			f1 := rl.Fields[i]
			f2 := rl.Fields[j]

			f1End := f1.Offset + f1.Size
			f2End := f2.Offset + f2.Size

			if f1.Offset < f2End && f1End > f2.Offset {
				return fmt.Errorf("fields %s and %s overlap in region %s",
					f1.Name, f2.Name, rl.Name)
			}
		}
	}

	return nil
}

// TotalFieldsSize returns the sum of all field sizes
func (rl *RegionLayout) TotalFieldsSize() int {
	total := 0
	for _, field := range rl.Fields {
		total += field.Size
	}
	return total
}

// String returns a human-readable representation of the layout
func (rl *RegionLayout) String() string {
	s := fmt.Sprintf("Region %s (size=%d bytes, mode=%s)\n", rl.Name, rl.Size, rl.Mode)
	for _, field := range rl.Fields {
		s += fmt.Sprintf("  [%3d +%2d] %-12s %s\n",
			field.Offset, field.Size, field.Name, field.Type.String())
	}
	return s
}

// RegionSpec defines the specification for creating a region in an arena.
// This replaces the old ir.Region for arena creation.
type RegionSpec struct {
	Name   string
	Size   int64
	Mode   RegionMode
	Header *Header // Optional header for field definitions
}

// ComputeLayout creates a RegionLayout from a RegionSpec with header fields.
// This is a convenience function for building layouts from header definitions.
func ComputeLayout(spec *RegionSpec) *RegionLayout {
	layout := &RegionLayout{
		Name:   spec.Name,
		Size:   spec.Size,
		Mode:   spec.Mode,
		Fields: make([]FieldLayout, 0),
	}

	if spec.Header != nil && spec.Header.Fields != nil {
		for _, field := range spec.Header.Fields {
			layout.Fields = append(layout.Fields, FieldLayout{
				Name:   field.Name,
				Type:   field.Type,
				Offset: field.Offset,
				Size:   field.Size,
			})
		}
	}

	return layout
}

// ============================================================================
// Layout Construction Helpers
// ============================================================================

// CacheLineSize is the alignment boundary for SIMD-friendly access.
const CacheLineSize = 64

// AlignedSize rounds size up to the nearest multiple of alignment.
// If alignment is 0 it defaults to 8 (natural word alignment).
func AlignedSize(size int64, alignment int) int64 {
	if alignment <= 0 {
		alignment = 8
	}
	a := int64(alignment)
	return (size + a - 1) / a * a
}

// LayoutFromFields creates a RegionLayout from a field map with auto-computed
// offsets.  Fields are laid out sequentially in alphabetical order for
// determinism.  The region size is auto-computed as the smallest value that
// contains all fields, rounded up to 8-byte alignment.
func LayoutFromFields(name string, fields map[string]FieldType) *RegionLayout {
	// Sort field names for deterministic ordering
	sorted := make([]string, 0, len(fields))
	for k := range fields {
		sorted = append(sorted, k)
	}
	// Simple insertion sort (no sort import needed, fields are small)
	for i := 1; i < len(sorted); i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	layout := &RegionLayout{
		Name:   name,
		Mode:   ModeStream,
		Fields: make([]FieldLayout, 0, len(fields)),
	}

	offset := 0
	for _, fname := range sorted {
		ft := fields[fname]
		sz := ft.Size()
		if sz == 0 {
			continue // skip variable-length fields without explicit size
		}
		layout.Fields = append(layout.Fields, FieldLayout{
			Name:   fname,
			Type:   ft,
			Offset: offset,
			Size:   sz,
		})
		offset += sz
	}

	layout.Size = AlignedSize(int64(offset), 8)
	return layout
}

// LayoutFromFieldsOrdered creates a RegionLayout preserving the caller-supplied
// field order.  Use this when field ordering matters (e.g. for cache-line
// packing).
func LayoutFromFieldsOrdered(name string, fields []FieldLayout) *RegionLayout {
	layout := &RegionLayout{
		Name:   name,
		Mode:   ModeStream,
		Fields: make([]FieldLayout, len(fields)),
	}
	offset := 0
	for i, f := range fields {
		sz := f.Size
		if sz == 0 {
			sz = f.Type.Size()
		}
		layout.Fields[i] = FieldLayout{
			Name:   f.Name,
			Type:   f.Type,
			Offset: offset,
			Size:   sz,
		}
		offset += sz
	}
	layout.Size = AlignedSize(int64(offset), 8)
	return layout
}

// ============================================================================
// Struct Tag Parser
// ============================================================================

// layoutCache caches parsed struct tag layouts to avoid repeated reflection.
var layoutCache sync.Map // map[reflect.Type]*RegionLayout

// LayoutFromStruct creates a RegionLayout by reflecting on struct tags.
// Tag format: `mempipe:"field:name[,type:t][,offset:n][,size:n]"`
// If `type` is omitted, it is inferred from the Go field type.
// If `offset` is omitted, fields are packed sequentially.
// Results are cached per type — this is called once at build time, not on hot paths.
func LayoutFromStruct(name string, v any) (*RegionLayout, error) {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("LayoutFromStruct requires a struct, got %s", t.Kind())
	}

	// Check cache
	if cached, ok := layoutCache.Load(t); ok {
		layout := cached.(*RegionLayout)
		// Return a copy with the requested name
		cp := *layout
		cp.Name = name
		return &cp, nil
	}

	layout := &RegionLayout{
		Name:   name,
		Mode:   ModeStream,
		Fields: make([]FieldLayout, 0, t.NumField()),
	}

	offset := 0
	for i := 0; i < t.NumField(); i++ {
		sf := t.Field(i)
		tag := sf.Tag.Get("mempipe")
		if tag == "" || tag == "-" {
			continue
		}

		fl, err := parseFieldTag(sf, tag, offset)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", sf.Name, err)
		}

		layout.Fields = append(layout.Fields, fl)
		offset = fl.Offset + fl.Size
	}

	layout.Size = AlignedSize(int64(offset), 8)

	// Validate before caching
	if err := layout.Validate(); err != nil {
		return nil, fmt.Errorf("layout validation: %w", err)
	}

	layoutCache.Store(t, layout)
	return layout, nil
}

// parseFieldTag parses a single struct field's mempipe tag.
func parseFieldTag(sf reflect.StructField, tag string, currentOffset int) (FieldLayout, error) {
	fl := FieldLayout{}
	parts := strings.Split(tag, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)
		kv := strings.SplitN(part, ":", 2)
		if len(kv) != 2 {
			return fl, fmt.Errorf("invalid tag part %q (expected key:value)", part)
		}
		key, val := kv[0], kv[1]

		switch key {
		case "field":
			fl.Name = val
		case "type":
			ft, err := ParseFieldType(val)
			if err != nil {
				return fl, err
			}
			fl.Type = ft
		case "offset":
			n, err := parseInt(val)
			if err != nil {
				return fl, fmt.Errorf("invalid offset %q: %w", val, err)
			}
			fl.Offset = n
		case "size":
			n, err := parseInt(val)
			if err != nil {
				return fl, fmt.Errorf("invalid size %q: %w", val, err)
			}
			fl.Size = n
		default:
			return fl, fmt.Errorf("unknown tag key %q", key)
		}
	}

	// Default field name to Go field name in lowercase
	if fl.Name == "" {
		fl.Name = strings.ToLower(sf.Name)
	}

	// Infer type from Go field type if not specified
	if fl.Type == 0 && fl.Name != "" {
		ft, err := inferFieldType(sf.Type)
		if err != nil {
			return fl, fmt.Errorf("cannot infer type for %s: %w", sf.Name, err)
		}
		fl.Type = ft
	}

	// Auto-compute offset if not specified
	if fl.Offset == 0 && currentOffset > 0 {
		fl.Offset = currentOffset
	}

	// Auto-compute size from type
	if fl.Size == 0 {
		fl.Size = fl.Type.Size()
	}

	if fl.Size == 0 {
		return fl, fmt.Errorf("field %s has zero size (vecf32 requires explicit size:N)", fl.Name)
	}

	return fl, nil
}

// inferFieldType maps a Go reflect.Type to a FieldType.
func inferFieldType(t reflect.Type) (FieldType, error) {
	switch t.Kind() {
	case reflect.Uint8:
		return TypeU8, nil
	case reflect.Uint16:
		return TypeU16, nil
	case reflect.Uint32:
		return TypeU32, nil
	case reflect.Uint64:
		return TypeU64, nil
	case reflect.Int8:
		return TypeI8, nil
	case reflect.Int16:
		return TypeI16, nil
	case reflect.Int32:
		return TypeI32, nil
	case reflect.Int64:
		return TypeI64, nil
	case reflect.Float32:
		return TypeF32, nil
	case reflect.Float64:
		return TypeF64, nil
	case reflect.Bool:
		return TypeBool, nil
	case reflect.Array:
		if t.Elem().Kind() == reflect.Float32 {
			return TypeVecF32, nil
		}
	}
	return 0, fmt.Errorf("unsupported Go type: %s", t)
}

// parseInt is a minimal integer parser (avoids importing strconv).
func parseInt(s string) (int, error) {
	n := 0
	neg := false
	i := 0
	if len(s) > 0 && s[0] == '-' {
		neg = true
		i = 1
	}
	if i >= len(s) {
		return 0, fmt.Errorf("empty number")
	}
	for ; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("invalid digit %c", c)
		}
		n = n*10 + int(c-'0')
	}
	if neg {
		n = -n
	}
	return n, nil
}
