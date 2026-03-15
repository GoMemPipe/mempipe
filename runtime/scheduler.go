// Package runtime provides the scheduler for cell execution
package runtime

import (
"fmt"
"log"
"sync"
)

// AsyncTask represents an asynchronous task to be executed
type AsyncTask func()

// CellFunc is the function type for cell execution logic
type CellFunc func()

// CellSpec defines a computational cell with Go closure execution
type CellSpec struct {
	Name    string
	Inputs  []string // Region names this cell reads from
	Outputs []string // Region names this cell writes to
	Fn      CellFunc // Go closure to execute
}

// SchedulePolicy controls how cells are dispatched.
type SchedulePolicy int

const (
// ScheduleSequential executes cells one at a time in topological order.
ScheduleSequential SchedulePolicy = iota
// ScheduleParallel executes independent cells concurrently.
ScheduleParallel
)

// Scheduler manages cell execution with memory-native infrastructure.
// It keeps no legacy RegionManager; all region memory lives in an Arena
// which is owned by the Pipeline that wraps this scheduler.
type Scheduler struct {
	cells      []*CellSpec
	cellMap    map[string]*CellSpec
	committed  map[string]bool // tracks which regions have been committed

	policy     SchedulePolicy
	maxWorkers int

	// Async support
	asyncQueue   chan AsyncTask
	asyncEnabled bool

	// Memory-native infrastructure
	clock       *RuntimeClock
	pipeManager *PipeManager
	logger      *MemoryLogger
	enableOSLog bool
}

// NewScheduler creates a new scheduler with memory-native infrastructure.
func NewScheduler() *Scheduler {
	clock := NewRuntimeClock(1)
	pipeManager := NewPipeManager()
	logger := NewMemoryLogger(clock, 10000)

	return &Scheduler{
		cells:      make([]*CellSpec, 0),
		cellMap:    make(map[string]*CellSpec),
		committed:  make(map[string]bool),
		policy:     ScheduleSequential,
		maxWorkers: 4,
		asyncQueue: make(chan AsyncTask, 64),
		clock:      clock,
		pipeManager: pipeManager,
		logger:     logger,
		enableOSLog: false,
	}
}

// AddCell registers a cell with the scheduler.
func (s *Scheduler) AddCell(cell *CellSpec) {
	s.cells = append(s.cells, cell)
	s.cellMap[cell.Name] = cell
}

// SetPolicy sets the scheduling policy.
func (s *Scheduler) SetPolicy(p SchedulePolicy, workers int) {
	s.policy = p
	if workers > 0 {
		s.maxWorkers = workers
	}
}

// EnableAsync starts the background async task processor.
func (s *Scheduler) EnableAsync() {
	if !s.asyncEnabled {
		s.asyncEnabled = true
		go s.loopAsync()
	}
}

// DisableAsync stops the async loop.
func (s *Scheduler) DisableAsync() {
	s.asyncEnabled = false
}

// RunAsync queues an asynchronous task.
func (s *Scheduler) RunAsync(task AsyncTask) {
	if s.asyncEnabled {
		select {
		case s.asyncQueue <- task:
		default:
			task() // queue full – run inline
		}
	} else {
		task()
	}
}

func (s *Scheduler) loopAsync() {
	for s.asyncEnabled {
		task := <-s.asyncQueue
		func() {
			defer func() {
				if r := recover(); r != nil {
					s.log("[ASYNC] Task panicked: %v", r)
				}
			}()
			task()
		}()
	}
}

// MarkCommitted marks a region name as having valid data.
func (s *Scheduler) MarkCommitted(name string) {
	s.committed[name] = true
}

// MarkAllCommitted marks all listed region names as committed.
func (s *Scheduler) MarkAllCommitted(names []string) {
	for _, n := range names {
		s.committed[n] = true
	}
}

// ResetCommitted clears commit state for the next iteration.
func (s *Scheduler) ResetCommitted() {
	for k := range s.committed {
		delete(s.committed, k)
	}
}

// ============================================================================
// Topological Sort
// ============================================================================

// topoOrder returns cells in topological order based on input/output regions.
// If there is a cycle the original registration order is returned.
func (s *Scheduler) topoOrder() []*CellSpec {
	if len(s.cells) == 0 {
		return nil
	}

	// Build producer map: region → cell that produces it
	producer := make(map[string]string) // region name → cell name
	for _, c := range s.cells {
		for _, out := range c.Outputs {
			producer[out] = c.Name
		}
	}

	// Build adjacency: cell → set of cells it depends on
	inDeg := make(map[string]int)
	deps := make(map[string][]string) // cell → cells that depend on it

	for _, c := range s.cells {
		if _, ok := inDeg[c.Name]; !ok {
			inDeg[c.Name] = 0
		}
		for _, inp := range c.Inputs {
			prod, ok := producer[inp]
			if !ok || prod == c.Name {
				continue
			}
			deps[prod] = append(deps[prod], c.Name)
			inDeg[c.Name]++
		}
	}

	// Kahn's algorithm
queue := make([]string, 0, len(s.cells))
for _, c := range s.cells {
if inDeg[c.Name] == 0 {
queue = append(queue, c.Name)
}
}

order := make([]*CellSpec, 0, len(s.cells))
for len(queue) > 0 {
name := queue[0]
queue = queue[1:]
order = append(order, s.cellMap[name])
for _, dep := range deps[name] {
inDeg[dep]--
if inDeg[dep] == 0 {
queue = append(queue, dep)
}
}
}

if len(order) != len(s.cells) {
// Cycle detected – fall back to registration order
return s.cells
}
return order
}

// levelize groups topologically-sorted cells into concurrent levels.
func (s *Scheduler) levelize() [][]*CellSpec {
producer := make(map[string]string)
for _, c := range s.cells {
for _, out := range c.Outputs {
producer[out] = c.Name
}
}

level := make(map[string]int)
// BFS-style level assignment
order := s.topoOrder()
for _, c := range order {
maxLvl := 0
for _, inp := range c.Inputs {
if p, ok := producer[inp]; ok {
if l, ok2 := level[p]; ok2 && l+1 > maxLvl {
maxLvl = l + 1
}
}
}
level[c.Name] = maxLvl
}

// Group by level
maxLevel := 0
for _, l := range level {
if l > maxLevel {
maxLevel = l
}
}
levels := make([][]*CellSpec, maxLevel+1)
for _, c := range order {
l := level[c.Name]
levels[l] = append(levels[l], c)
}
return levels
}

// ============================================================================
// Execution
// ============================================================================

// Run executes all registered cells once in dependency order.
func (s *Scheduler) Run() error {
s.clock.Tick()

switch s.policy {
case ScheduleParallel:
return s.runParallel()
default:
return s.runSequential()
}
}

// RunIterations executes N iterations with an optional data-prep callback.
func (s *Scheduler) RunIterations(n int, prepare func(iter int) error) error {
for i := 0; i < n; i++ {
if prepare != nil {
if err := prepare(i); err != nil {
return fmt.Errorf("iteration %d prepare: %w", i, err)
}
}
// Mark source regions committed
s.markSources()
if err := s.Run(); err != nil {
return fmt.Errorf("iteration %d: %w", i, err)
}
s.ResetCommitted()
}
return nil
}

func (s *Scheduler) runSequential() error {
order := s.topoOrder()
for _, cell := range order {
if !s.canExecute(cell) {
continue
}
s.log("[EXEC] %s", cell.Name)
cell.Fn()
for _, out := range cell.Outputs {
s.committed[out] = true
}
s.clock.Tick()
}
return nil
}

func (s *Scheduler) runParallel() error {
levels := s.levelize()

for _, lvl := range levels {
if len(lvl) == 1 {
cell := lvl[0]
s.log("[EXEC] %s", cell.Name)
cell.Fn()
for _, out := range cell.Outputs {
s.committed[out] = true
}
} else {
var wg sync.WaitGroup
errCh := make(chan error, len(lvl))
sem := make(chan struct{}, s.maxWorkers)

for _, cell := range lvl {
wg.Add(1)
sem <- struct{}{}
go func(c *CellSpec) {
defer wg.Done()
defer func() { <-sem }()
defer func() {
if r := recover(); r != nil {
errCh <- fmt.Errorf("cell %s panicked: %v", c.Name, r)
}
}()
s.log("[EXEC] %s (parallel)", c.Name)
c.Fn()
}(cell)
}
wg.Wait()
close(errCh)

// Check errors
for err := range errCh {
return err
}

// Commit outputs
for _, cell := range lvl {
for _, out := range cell.Outputs {
s.committed[out] = true
}
}
}
s.clock.Tick()
}
return nil
}

func (s *Scheduler) canExecute(cell *CellSpec) bool {
for _, inp := range cell.Inputs {
if !s.committed[inp] {
return false
}
}
return true
}

func (s *Scheduler) markSources() {
outputs := make(map[string]bool)
for _, c := range s.cells {
for _, out := range c.Outputs {
outputs[out] = true
}
}
for _, c := range s.cells {
for _, inp := range c.Inputs {
if !outputs[inp] {
s.committed[inp] = true
}
}
}
}

// ============================================================================
// Accessors
// ============================================================================

// Clock returns the runtime clock.
func (s *Scheduler) Clock() *RuntimeClock { return s.clock }

// PipeManager returns the pipe manager.
func (s *Scheduler) PipeManager() *PipeManager { return s.pipeManager }

// Logger returns the memory logger.
func (s *Scheduler) Logger() *MemoryLogger { return s.logger }

// Stdout returns stdout pipe contents.
func (s *Scheduler) Stdout() string {
p, err := s.pipeManager.GetPipe("/sys/stdout")
if err != nil {
return ""
}
return p.Read()
}

// Stderr returns stderr pipe contents.
func (s *Scheduler) Stderr() string {
p, err := s.pipeManager.GetPipe("/sys/stderr")
if err != nil {
return ""
}
return p.Read()
}

// EnableOSLogging enables/disables OS log output.
func (s *Scheduler) EnableOSLogging(enabled bool) {
s.enableOSLog = enabled
}

func (s *Scheduler) log(format string, args ...interface{}) {
s.logger.Info(format, args...)
if s.enableOSLog {
log.Printf(format, args...)
}
}
