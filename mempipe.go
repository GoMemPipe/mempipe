// Package mempipe provides a zero-allocation pipeline engine for real-time
// data processing.  Pipelines are composed of Regions (typed arena-backed
// memory) and Cells (pure-Go closures that read/write regions).
//
// Quick start:
//
//	pipe := mempipe.NewPipeline()
//	r := mempipe.AddRegion[MyStruct](pipe, "sensor")
//	pipe.Cell("process", func() { v := r.Get(); v.Temp += 1; r.Set(v) })
//	pipe.Run(100)
package mempipe

// Version is the single source of truth for the mempipe library version.
const Version = "1.0.0"
