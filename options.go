package mempipe

// Option configures a Pipeline at construction time.
type Option func(*Pipeline)

// RegionOption configures a region at registration time.
type RegionOption func(*regionOpts)

type regionOpts struct {
	mode     string // "stream", "ring", etc.
	sizeHint int64  // explicit size hint (0 = auto)
}

// WithWorkers sets the number of parallel workers for cell execution.
// Workers <= 1 means sequential execution.
func WithWorkers(n int) Option {
	return func(p *Pipeline) {
		if n < 1 {
			n = 1
		}
		p.workers = n
	}
}

// WithArenaSizeHint provides a total arena size hint (bytes).
// If 0 the arena is auto-sized from region layouts.
func WithArenaSizeHint(bytes int64) Option {
	return func(p *Pipeline) {
		p.arenaSizeHint = bytes
	}
}

// WithRegionMode sets the region mode (stream, ring, slab, windowed, append).
func WithRegionMode(mode string) RegionOption {
	return func(o *regionOpts) {
		o.mode = mode
	}
}

// WithRegionSize sets an explicit region size in bytes.
func WithRegionSize(bytes int64) RegionOption {
	return func(o *regionOpts) {
		o.sizeHint = bytes
	}
}
