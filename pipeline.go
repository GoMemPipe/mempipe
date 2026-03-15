package mempipe

import (
	"context"
	"fmt"
	"log"

	"github.com/GoMemPipe/mempipe/runtime"
)

// CellFunc is the function signature for cell logic.  Cells read and write
// regions via RegionHandle.Get / RegionHandle.Set.
type CellFunc func()

// cellDef stores the definition of a cell before the pipeline is built.
type cellDef struct {
	name    string
	fn      CellFunc
	inputs  []string
	outputs []string
}

// regionDef stores the definition of a region before the pipeline is built.
type regionDef struct {
	name   string
	layout *runtime.RegionLayout
	spec   *runtime.RegionSpec
}

// Pipeline is the top-level container.  Use NewPipeline() to create one,
// then register regions and cells, and finally call Run() or RunContinuous().
type Pipeline struct {
	// Configuration
	workers       int
	arenaSizeHint int64

	// Definitions (pre-build)
	regionDefs []*regionDef
	cellDefs   []*cellDef

	// Runtime (post-build)
	arena     *runtime.RegionArena
	scheduler *runtime.Scheduler
	built     bool

	// Callbacks
	onIteration func(iter int)
}

// NewPipeline creates a new Pipeline with the given options.
func NewPipeline(opts ...Option) *Pipeline {
	p := &Pipeline{
		workers:    1,
		regionDefs: make([]*regionDef, 0),
		cellDefs:   make([]*cellDef, 0),
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// RegionHandle is a type-safe handle to a named region in the arena.
// Use Get() and Set() to read/write the full struct, or Region() for
// field-level access.
type RegionHandle[T any] struct {
	name   string
	pipe   *Pipeline
	typed  *runtime.TypedRegion[T] // set after build
	region *runtime.Region         // set after build
}

// AddRegion registers a struct-typed region with the pipeline.
// The layout is derived from T's `mempipe` struct tags.
func AddRegion[T any](pipe *Pipeline, name string, opts ...RegionOption) *RegionHandle[T] {
	o := &regionOpts{mode: "stream"}
	for _, fn := range opts {
		fn(o)
	}

	var zero T
	layout, err := runtime.LayoutFromStruct(name, zero)
	if err != nil {
		panic(fmt.Sprintf("mempipe.AddRegion[%s]: %v", name, err))
	}

	mode, _ := runtime.ParseMode(o.mode)
	size := layout.Size
	if o.sizeHint > size {
		size = o.sizeHint
	}

	spec := &runtime.RegionSpec{
		Name: name,
		Size: size,
		Mode: mode,
		Header: &runtime.Header{
			Fields: layoutToHeaderFields(layout),
		},
	}

	pipe.regionDefs = append(pipe.regionDefs, &regionDef{
		name:   name,
		layout: layout,
		spec:   spec,
	})

	return &RegionHandle[T]{name: name, pipe: pipe}
}

// AddFieldRegion registers a region using explicit field declarations
// (no struct tags).  Fields maps field names to type strings
// ("u8","u16","u32","u64","i8","i16","i32","i64","f32","f64","bool").
func (p *Pipeline) AddFieldRegion(name string, fields map[string]string, opts ...RegionOption) {
	o := &regionOpts{mode: "stream"}
	for _, fn := range opts {
		fn(o)
	}

	ftMap := make(map[string]runtime.FieldType, len(fields))
	for fname, ftype := range fields {
		ft, err := runtime.ParseFieldType(ftype)
		if err != nil {
			panic(fmt.Sprintf("mempipe.AddFieldRegion(%s): field %s: %v", name, fname, err))
		}
		ftMap[fname] = ft
	}

	layout := runtime.LayoutFromFields(name, ftMap)
	mode, _ := runtime.ParseMode(o.mode)
	size := layout.Size
	if o.sizeHint > size {
		size = o.sizeHint
	}

	spec := &runtime.RegionSpec{
		Name: name,
		Size: size,
		Mode: mode,
		Header: &runtime.Header{
			Fields: layoutToHeaderFields(layout),
		},
	}

	p.regionDefs = append(p.regionDefs, &regionDef{
		name:   name,
		layout: layout,
		spec:   spec,
	})
}

// Cell registers a named cell (computation step).  inputs/outputs are
// region names used for dependency ordering.
func (p *Pipeline) Cell(name string, fn CellFunc, inputs []string, outputs []string) *Pipeline {
	p.cellDefs = append(p.cellDefs, &cellDef{
		name:    name,
		fn:      fn,
		inputs:  inputs,
		outputs: outputs,
	})
	return p
}

// SimpleCell registers a cell with no explicit dependency declarations.
// The cell runs in registration order.
func (p *Pipeline) SimpleCell(name string, fn CellFunc) *Pipeline {
	return p.Cell(name, fn, nil, nil)
}

// OnIteration registers a callback invoked before each iteration.
func (p *Pipeline) OnIteration(fn func(iter int)) {
	p.onIteration = fn
}

// Validate checks the pipeline for configuration errors:
//   - duplicate region/cell names
//   - dependency cycles
//   - references to non-existent regions
func (p *Pipeline) Validate() error {
	// Check duplicate region names
	rnames := make(map[string]bool, len(p.regionDefs))
	for _, rd := range p.regionDefs {
		if rnames[rd.name] {
			return fmt.Errorf("duplicate region name: %s", rd.name)
		}
		rnames[rd.name] = true
	}

	// Check duplicate cell names and region references
	cnames := make(map[string]bool, len(p.cellDefs))
	for _, cd := range p.cellDefs {
		if cnames[cd.name] {
			return fmt.Errorf("duplicate cell name: %s", cd.name)
		}
		cnames[cd.name] = true
		for _, inp := range cd.inputs {
			if !rnames[inp] {
				return fmt.Errorf("cell %s references unknown input region: %s", cd.name, inp)
			}
		}
		for _, out := range cd.outputs {
			if !rnames[out] {
				return fmt.Errorf("cell %s references unknown output region: %s", cd.name, out)
			}
		}
	}

	return nil
}

// build materializes the arena and scheduler from the definitions.
func (p *Pipeline) build() error {
	if err := p.Validate(); err != nil {
		return fmt.Errorf("validation: %w", err)
	}

	specs := make([]*runtime.RegionSpec, len(p.regionDefs))
	layouts := runtime.NewLayoutTable()
	for i, rd := range p.regionDefs {
		specs[i] = rd.spec
		layouts.Add(rd.layout)
	}

	arena, err := runtime.NewArena(specs, layouts)
	if err != nil {
		return fmt.Errorf("arena: %w", err)
	}
	p.arena = arena

	sched := runtime.NewScheduler()
	if p.workers > 1 {
		sched.SetPolicy(runtime.ScheduleParallel, p.workers)
	}
	for _, cd := range p.cellDefs {
		// Wrap CellFunc so the types align (package-level type != runtime type).
		fn := cd.fn // capture for closure
		sched.AddCell(&runtime.CellSpec{
			Name:    cd.name,
			Inputs:  cd.inputs,
			Outputs: cd.outputs,
			Fn:      func() { fn() },
		})
	}
	p.scheduler = sched
	p.built = true
	return nil
}

// Run executes the pipeline for the given number of iterations.
func (p *Pipeline) Run(iterations int) error {
	if !p.built {
		if err := p.build(); err != nil {
			return err
		}
	}

	var prepare func(int) error
	if p.onIteration != nil {
		cb := p.onIteration
		prepare = func(iter int) error { cb(iter); return nil }
	}
	return p.scheduler.RunIterations(iterations, prepare)
}

// RunContinuous runs the pipeline until the context is cancelled.
func (p *Pipeline) RunContinuous(ctx context.Context) error {
	if !p.built {
		if err := p.build(); err != nil {
			return err
		}
	}

	iter := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if p.onIteration != nil {
			p.onIteration(iter)
		}
		if err := p.scheduler.Run(); err != nil {
			return fmt.Errorf("iteration %d: %w", iter, err)
		}
		iter++
	}
}

// Arena returns the underlying arena (available after build / first Run).
func (p *Pipeline) Arena() *runtime.RegionArena {
	return p.arena
}

// Clock returns the runtime clock.
func (p *Pipeline) Clock() *runtime.RuntimeClock {
	if p.scheduler == nil {
		return nil
	}
	return p.scheduler.Clock()
}

// Stdout returns memory-native stdout contents.
func (p *Pipeline) Stdout() string {
	if p.scheduler == nil {
		return ""
	}
	return p.scheduler.Stdout()
}

// Stderr returns memory-native stderr contents.
func (p *Pipeline) Stderr() string {
	if p.scheduler == nil {
		return ""
	}
	return p.scheduler.Stderr()
}

// Get reads the full struct value from arena memory.  Zero allocations.
//
//mem:hot
//mem:nogc
func (h *RegionHandle[T]) Get() T {
	if h.typed == nil {
		h.resolve()
	}
	return h.typed.Get()
}

// Set writes the full struct value to arena memory.  Zero allocations.
//
//mem:hot
//mem:nogc
func (h *RegionHandle[T]) Set(v T) {
	if h.typed == nil {
		h.resolve()
	}
	h.typed.Set(v)
}

// Region returns the underlying runtime.Region for field-level access.
func (h *RegionHandle[T]) Region() *runtime.Region {
	if h.region == nil {
		h.resolve()
	}
	return h.region
}

// Name returns the region name.
func (h *RegionHandle[T]) Name() string {
	return h.name
}

// resolve initializes the typed region after the pipeline is built.
func (h *RegionHandle[T]) resolve() {
	if h.pipe.arena == nil {
		panic(fmt.Sprintf("RegionHandle[%s]: pipeline not built yet — call Run() first", h.name))
	}
	if h.typed != nil {
		return
	}
	tr, err := runtime.NewTypedRegion[T](h.pipe.arena, h.name)
	if err != nil {
		panic(fmt.Sprintf("RegionHandle[%s]: %v", h.name, err))
	}
	h.typed = tr
	h.region = tr.Region()
}

// layoutToHeaderFields converts a RegionLayout's fields to HeaderField slice.
func layoutToHeaderFields(layout *runtime.RegionLayout) []*runtime.HeaderField {
	hf := make([]*runtime.HeaderField, len(layout.Fields))
	for i, f := range layout.Fields {
		hf[i] = &runtime.HeaderField{
			Name:   f.Name,
			Type:   f.Type,
			Offset: f.Offset,
			Size:   f.Size,
		}
	}
	return hf
}

// suppress unused import warning during development
var _ = log.Printf
