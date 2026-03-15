package memlint_test

import (
	"testing"

	"github.com/GoMemPipe/mempipe/tools/memlint"
	"golang.org/x/tools/go/analysis/analysistest"
)

func TestAll(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, memlint.Analyzer, "a")
}
