// Package builder provides a Go-native API for defining and running MemPipe pipelines
// without writing .mempipe DSL files. It allows developers to define regions, cells,
// and continuous execution directly in Go code with ergonomic, intuitive syntax.
package builder

import (
	"fmt"
	"log"

	mio "github.com/GoMemPipe/mempipe/module/io"
	"github.com/GoMemPipe/mempipe/runtime"
)

// FieldType represents the type of a field in a region
type FieldType string

const (
	// Uint8 represents an unsigned 8-bit integer
	Uint8 FieldType = "u8"
	// Uint16 represents an unsigned 16-bit integer
	Uint16 FieldType = "u16"
	// Uint32 represents an unsigned 32-bit integer
	Uint32 FieldType = "u32"
	// Uint64 represents an unsigned 64-bit integer
	Uint64 FieldType = "u64"
	// Int8 represents a signed 8-bit integer
	Int8 FieldType = "i8"
	// Int16 represents a signed 16-bit integer
	Int16 FieldType = "i16"
	// Int32 represents a signed 32-bit integer
	Int32 FieldType = "i32"
	// Int64 represents a signed 64-bit integer
	Int64 FieldType = "i64"
	// Float32 represents a 32-bit floating point number
	Float32 FieldType = "f32"
	// Float64 represents a 64-bit floating point number
	Float64 FieldType = "f64"
	// Bool represents a boolean value (stored as u8)
	Bool FieldType = "bool"
)

// Fields maps field names to their types
type Fields map[string]FieldType

// Region represents a named memory region with typed fields
type Region struct {
	name   string
	fields Fields
	spec   *runtime.RegionSpec
}

// Name returns the region name
func (r *Region) Name() string {
	return r.name
}

// cellFunc is the function type for cell execution logic
type cellFunc func(*Context)

// cell represents a computational cell with its function
type cell struct {
	name string
	fn   cellFunc
}

// Pipeline represents a complete MemPipe pipeline definition
type Pipeline struct {
	regions     []*Region
	cells       []*cell
	iterations  int
	arena       *runtime.RegionArena
	scheduler   *runtime.Scheduler
	initialized bool
}

// NewPipeline creates a new pipeline builder
func NewPipeline() *Pipeline {
	return &Pipeline{
		regions:    make([]*Region, 0),
		cells:      make([]*cell, 0),
		iterations: 1, // default to single iteration
	}
}

// Region defines a new memory region with the given name and fields
// Returns a Region handle that can be used in cell contexts
func (p *Pipeline) Region(name string, fields Fields) *Region {
	// Calculate total size based on field types
	var totalSize int64
	irFields := make([]*runtime.HeaderField, 0, len(fields))
	offset := 0

	// We need deterministic ordering, so we'll iterate in a stable way
	// by creating the IR fields in the order they're accessed
	fieldList := make([]struct {
		name  string
		ftype FieldType
	}, 0, len(fields))

	for fname, ftype := range fields {
		fieldList = append(fieldList, struct {
			name  string
			ftype FieldType
		}{fname, ftype})
	}

	// Build header fields
	for _, f := range fieldList {
		ft, size := convertFieldType(f.ftype)
		hf := &runtime.HeaderField{
			Name:   f.name,
			Type:   ft,
			Offset: offset,
			Size:   size,
		}
		irFields = append(irFields, hf)
		offset += size
	}

	totalSize = int64(offset)
	if totalSize == 0 {
		totalSize = 1024 // default size if no header
	}

	// Create region spec
	spec := &runtime.RegionSpec{
		Name: name,
		Size: totalSize,
		Mode: runtime.ModeStream, // default to stream mode
		Header: &runtime.Header{
			Fields: irFields,
		},
	}

	region := &Region{
		name:   name,
		fields: fields,
		spec:   spec,
	}

	p.regions = append(p.regions, region)
	return region
}

// Cell defines a new computational cell with the given name and function
// The function receives a Context that provides access to region fields
func (p *Pipeline) Cell(name string, fn cellFunc) *Pipeline {
	p.cells = append(p.cells, &cell{
		name: name,
		fn:   fn,
	})
	return p
}

// Continuous sets the number of iterations for continuous execution
// If n <= 0, defaults to 1
func (p *Pipeline) Continuous(n int) *Pipeline {
	if n <= 0 {
		n = 1
	}
	p.iterations = n
	return p
}

// Run executes the pipeline
func (p *Pipeline) Run() error {
	if !p.initialized {
		if err := p.build(); err != nil {
			return fmt.Errorf("failed to build pipeline: %w", err)
		}
	}

	log.Printf("[BUILDER] Starting pipeline execution with %d iterations", p.iterations)

	// Create context once and reuse to avoid GC overhead
	// Phase 4: Use arena-backed context for zero-alloc field access
	ctx := &Context{
		arena:     p.arena,
		vars:      make(map[string]interface{}, 8), // Pre-allocate small capacity for local vars
		iteration: 0,
	}

	// Execute iterations
	for iter := 0; iter < p.iterations; iter++ {
		if p.iterations > 1 {
			log.Printf("[BUILDER] === Iteration %d/%d ===", iter+1, p.iterations)
		}

		// Reset context for this iteration (no allocations)
		ctx.reset(iter)

		// Execute all cells sequentially
		for _, c := range p.cells {
			log.Printf("[BUILDER] Executing cell: %s", c.name)

			// Execute the cell function
			c.fn(ctx)

			log.Printf("[BUILDER] Completed cell: %s", c.name)
		}
	}

	log.Printf("[BUILDER] Pipeline execution completed")
	return nil
}

// Stdout returns the memory-native stdout contents.
func (p *Pipeline) Stdout() string {
	if p.scheduler == nil {
		return ""
	}
	return p.scheduler.Stdout()
}

// Stderr returns the memory-native stderr contents.
func (p *Pipeline) Stderr() string {
	if p.scheduler == nil {
		return ""
	}
	return p.scheduler.Stderr()
}

// build constructs the pipeline from the definition
func (p *Pipeline) build() error {
	log.Println("[BUILDER] Building pipeline from definition")

	// Collect specs and compute layouts
	specs := make([]*runtime.RegionSpec, 0, len(p.regions))
	layouts := runtime.NewLayoutTable()

	for _, region := range p.regions {
		specs = append(specs, region.spec)
		log.Printf("[BUILDER] Added region: %s (size=%d bytes)", region.name, region.spec.Size)

		layout := runtime.ComputeLayout(region.spec)
		if err := layout.Validate(); err != nil {
			return fmt.Errorf("invalid layout for region %s: %w", region.name, err)
		}
		layouts.Add(layout)
	}

	// Create arena
	arena, err := runtime.NewArena(specs, layouts)
	if err != nil {
		return fmt.Errorf("failed to create arena: %w", err)
	}
	p.arena = arena

	// Create scheduler
	p.scheduler = runtime.NewScheduler()

	p.initialized = true

	return nil
}

// Context provides access to region fields during cell execution
// Reused across iterations to avoid allocations
// Phase 4: Now uses arena-backed typed views (zero-alloc)
type Context struct {
	arena     *runtime.RegionArena   // Phase 3: Zero-alloc arena
	vars      map[string]interface{} // For local variables only
	iteration int
}

// reset prepares the context for a new iteration without allocating
func (ctx *Context) reset(iteration int) {
	ctx.iteration = iteration
	// Clear vars map without reallocating
	for k := range ctx.vars {
		delete(ctx.vars, k)
	}
}

// U8 reads a uint8 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) U8(region *Region, field string) uint8 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.U8(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// U16 reads a uint16 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) U16(region *Region, field string) uint16 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.U16(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// U32 reads a uint32 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) U32(region *Region, field string) uint32 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.U32(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// U64 reads a uint64 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) U64(region *Region, field string) uint64 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.U64(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// I8 reads an int8 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) I8(region *Region, field string) int8 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.I8(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// I16 reads an int16 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) I16(region *Region, field string) int16 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.I16(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// I32 reads an int32 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) I32(region *Region, field string) int32 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.I32(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// I64 reads an int64 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) I64(region *Region, field string) int64 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.I64(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// F32 reads a float32 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) F32(region *Region, field string) float32 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.F32(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// F64 reads a float64 field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) F64(region *Region, field string) float64 {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.F64(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return 0
	}
	return val
}

// Bool reads a boolean field from a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) Bool(region *Region, field string) bool {
	view := ctx.arena.MustRegion(region.name)
	val, err := view.Bool(field)
	if err != nil {
		log.Printf("Error reading field %s.%s: %v", region.name, field, err)
		return false
	}
	return val
}

// SetU8 writes a uint8 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetU8(region *Region, field string, value uint8) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetU8(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetU16 writes a uint16 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetU16(region *Region, field string, value uint16) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetU16(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetU32 writes a uint32 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetU32(region *Region, field string, value uint32) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetU32(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetU64 writes a uint64 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetU64(region *Region, field string, value uint64) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetU64(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetI8 writes an int8 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetI8(region *Region, field string, value int8) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetI8(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetI16 writes an int16 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetI16(region *Region, field string, value int16) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetI16(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetI32 writes an int32 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetI32(region *Region, field string, value int32) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetI32(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetI64 writes an int64 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetI64(region *Region, field string, value int64) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetI64(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetF32 writes a float32 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetF32(region *Region, field string, value float32) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetF32(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetF64 writes a float64 field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetF64(region *Region, field string, value float64) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetF64(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// SetBool writes a boolean field to a region
// Phase 4: Zero-alloc access via arena Region
//
//mem:hot
//mem:nogc
func (ctx *Context) SetBool(region *Region, field string, value bool) {
	view := ctx.arena.MustRegion(region.name)
	if err := view.SetBool(field, value); err != nil {
		log.Printf("Error writing field %s.%s: %v", region.name, field, err)
	}
}

// Iteration returns the current iteration number (0-based)
func (ctx *Context) Iteration() int {
	return ctx.iteration
}

// SetVar stores a value in a local variable
func (ctx *Context) SetVar(name string, value interface{}) {
	ctx.vars[name] = value
}

// GetVar retrieves a value from a local variable
func (ctx *Context) GetVar(name string) interface{} {
	return ctx.vars[name]
}

// Println writes to memory-native stdout (not OS stdout)
// This maintains MemPipe's OS-independent execution model
func (ctx *Context) Println(args ...interface{}) {
	ioMod := mio.GetGlobalIOModule()
	if ioMod == nil {
		log.Printf("Error: io module not available")
		return
	}
	_ = ioMod.Println(args...)
}

// Printf writes formatted output to memory-native stdout
func (ctx *Context) Printf(format string, args ...interface{}) {
	ioMod := mio.GetGlobalIOModule()
	if ioMod == nil {
		log.Printf("Error: io module not available")
		return
	}
	_ = ioMod.Printf(format, args...)
}

// Phase 4: readField and writeField removed - now using arena-backed zero-alloc access

// Helper functions to convert field types

func convertFieldType(ft FieldType) (runtime.FieldType, int) {
	switch ft {
	case Uint8:
		return runtime.TypeU8, 1
	case Uint16:
		return runtime.TypeU16, 2
	case Uint32:
		return runtime.TypeU32, 4
	case Uint64:
		return runtime.TypeU64, 8
	case Int8:
		return runtime.TypeI8, 1
	case Int16:
		return runtime.TypeI16, 2
	case Int32:
		return runtime.TypeI32, 4
	case Int64:
		return runtime.TypeI64, 8
	case Float32:
		return runtime.TypeF32, 4
	case Float64:
		return runtime.TypeF64, 8
	case Bool:
		return runtime.TypeBool, 1
	default:
		return runtime.TypeU32, 4 // default
	}
}

// Type conversion helpers

func toU8(v interface{}) uint8 {
	switch val := v.(type) {
	case uint8:
		return val
	case uint16:
		return uint8(val)
	case uint32:
		return uint8(val)
	case uint64:
		return uint8(val)
	case int8:
		return uint8(val)
	case int16:
		return uint8(val)
	case int32:
		return uint8(val)
	case int64:
		return uint8(val)
	case int:
		return uint8(val)
	case float32:
		return uint8(val)
	case float64:
		return uint8(val)
	default:
		return 0
	}
}

func toU16(v interface{}) uint16 {
	switch val := v.(type) {
	case uint8:
		return uint16(val)
	case uint16:
		return val
	case uint32:
		return uint16(val)
	case uint64:
		return uint16(val)
	case int8:
		return uint16(val)
	case int16:
		return uint16(val)
	case int32:
		return uint16(val)
	case int64:
		return uint16(val)
	case int:
		return uint16(val)
	case float32:
		return uint16(val)
	case float64:
		return uint16(val)
	default:
		return 0
	}
}

func toU32(v interface{}) uint32 {
	switch val := v.(type) {
	case uint8:
		return uint32(val)
	case uint16:
		return uint32(val)
	case uint32:
		return val
	case uint64:
		return uint32(val)
	case int8:
		return uint32(val)
	case int16:
		return uint32(val)
	case int32:
		return uint32(val)
	case int64:
		return uint32(val)
	case int:
		return uint32(val)
	case float32:
		return uint32(val)
	case float64:
		return uint32(val)
	default:
		return 0
	}
}

func toU64(v interface{}) uint64 {
	switch val := v.(type) {
	case uint8:
		return uint64(val)
	case uint16:
		return uint64(val)
	case uint32:
		return uint64(val)
	case uint64:
		return val
	case int8:
		return uint64(val)
	case int16:
		return uint64(val)
	case int32:
		return uint64(val)
	case int64:
		return uint64(val)
	case int:
		return uint64(val)
	case float32:
		return uint64(val)
	case float64:
		return uint64(val)
	default:
		return 0
	}
}

func toI8(v interface{}) int8 {
	switch val := v.(type) {
	case uint8:
		return int8(val)
	case uint16:
		return int8(val)
	case uint32:
		return int8(val)
	case uint64:
		return int8(val)
	case int8:
		return val
	case int16:
		return int8(val)
	case int32:
		return int8(val)
	case int64:
		return int8(val)
	case int:
		return int8(val)
	case float32:
		return int8(val)
	case float64:
		return int8(val)
	default:
		return 0
	}
}

func toI16(v interface{}) int16 {
	switch val := v.(type) {
	case uint8:
		return int16(val)
	case uint16:
		return int16(val)
	case uint32:
		return int16(val)
	case uint64:
		return int16(val)
	case int8:
		return int16(val)
	case int16:
		return val
	case int32:
		return int16(val)
	case int64:
		return int16(val)
	case int:
		return int16(val)
	case float32:
		return int16(val)
	case float64:
		return int16(val)
	default:
		return 0
	}
}

func toI32(v interface{}) int32 {
	switch val := v.(type) {
	case uint8:
		return int32(val)
	case uint16:
		return int32(val)
	case uint32:
		return int32(val)
	case uint64:
		return int32(val)
	case int8:
		return int32(val)
	case int16:
		return int32(val)
	case int32:
		return val
	case int64:
		return int32(val)
	case int:
		return int32(val)
	case float32:
		return int32(val)
	case float64:
		return int32(val)
	default:
		return 0
	}
}

func toI64(v interface{}) int64 {
	switch val := v.(type) {
	case uint8:
		return int64(val)
	case uint16:
		return int64(val)
	case uint32:
		return int64(val)
	case uint64:
		return int64(val)
	case int8:
		return int64(val)
	case int16:
		return int64(val)
	case int32:
		return int64(val)
	case int64:
		return val
	case int:
		return int64(val)
	case float32:
		return int64(val)
	case float64:
		return int64(val)
	default:
		return 0
	}
}

func toF32(v interface{}) float32 {
	switch val := v.(type) {
	case uint8:
		return float32(val)
	case uint16:
		return float32(val)
	case uint32:
		return float32(val)
	case uint64:
		return float32(val)
	case int8:
		return float32(val)
	case int16:
		return float32(val)
	case int32:
		return float32(val)
	case int64:
		return float32(val)
	case int:
		return float32(val)
	case float32:
		return val
	case float64:
		return float32(val)
	default:
		return 0
	}
}

func toF64(v interface{}) float64 {
	switch val := v.(type) {
	case uint8:
		return float64(val)
	case uint16:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case int:
		return float64(val)
	case float32:
		return float64(val)
	case float64:
		return val
	default:
		return 0
	}
}

func toBool(v interface{}) bool {
	switch val := v.(type) {
	case bool:
		return val
	case uint8:
		return val != 0
	case uint16:
		return val != 0
	case uint32:
		return val != 0
	case uint64:
		return val != 0
	case int8:
		return val != 0
	case int16:
		return val != 0
	case int32:
		return val != 0
	case int64:
		return val != 0
	case int:
		return val != 0
	case float32:
		return val != 0
	case float64:
		return val != 0
	default:
		return false
	}
}
