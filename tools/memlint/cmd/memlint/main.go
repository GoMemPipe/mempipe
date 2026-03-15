// Command memlint is a go/analysis-based static analyzer that enforces
// MemPipe's zero-allocation invariants.
//
// Usage:
//
//	go install github.com/GoMemPipe/mempipe/tools/memlint/cmd/memlint@latest
//	go vet -vettool=$(which memlint) ./...
//
// Or run directly:
//
//	memlint ./...
package main

import (
	"github.com/GoMemPipe/mempipe/tools/memlint"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() {
	singlechecker.Main(memlint.Analyzer)
}
