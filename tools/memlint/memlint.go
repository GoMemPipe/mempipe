// Package memlint implements a static analyzer that enforces MemPipe's
// zero-allocation invariants on functions annotated with //mem:hot and
// //mem:nogc directives.
//
// Rules:
//
//	ML001: No interface{}/any parameters or variables in //mem:hot functions
//	ML002: No make/new/append calls in //mem:nogc functions
//	ML003: No reflect usage in //mem:hot functions
//	ML004: No closures or defer in //mem:nogc functions (likely heap escape)
//	ML005: No ReadField/WriteField calls (use typed API instead)
package memlint

import (
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

// Analyzer is the memlint analysis pass.
var Analyzer = &analysis.Analyzer{
	Name:     "memlint",
	Doc:      "enforces MemPipe zero-allocation invariants on //mem:hot and //mem:nogc functions",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// directive represents a parsed //mem: comment directive.
type directive struct {
	hot  bool // //mem:hot — performance-critical path
	nogc bool // //mem:nogc — must have zero heap allocations
}

func run(pass *analysis.Pass) (interface{}, error) {
	insp := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Pre-scan: collect all //mem: directives attached to function declarations.
	funcDirectives := make(map[*ast.FuncDecl]directive)
	for _, file := range pass.Files {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Doc == nil {
				continue
			}
			d := parseDirectives(fn.Doc)
			if d.hot || d.nogc {
				funcDirectives[fn] = d
			}
		}
	}
	if len(funcDirectives) == 0 {
		return nil, nil
	}

	// Walk each annotated function's body.
	for fn, dir := range funcDirectives {
		if fn.Body == nil {
			continue
		}
		checkFuncBody(pass, fn, dir)
	}

	// ML005: file-wide check — ReadField/WriteField anywhere (not just in annotated funcs).
	nodeFilter := []ast.Node{(*ast.CallExpr)(nil)}
	insp.Preorder(nodeFilter, func(n ast.Node) {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		checkML005(pass, call)
	})

	return nil, nil
}

// parseDirectives extracts //mem:hot and //mem:nogc from a comment group.
func parseDirectives(doc *ast.CommentGroup) directive {
	var d directive
	for _, c := range doc.List {
		text := strings.TrimSpace(c.Text)
		switch text {
		case "//mem:hot":
			d.hot = true
		case "//mem:nogc":
			d.nogc = true
		}
	}
	return d
}

// checkFuncBody walks a function body and reports violations.
func checkFuncBody(pass *analysis.Pass, fn *ast.FuncDecl, dir directive) {
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if n == nil {
			return true
		}

		switch node := n.(type) {
		case *ast.CallExpr:
			if dir.nogc {
				checkML002(pass, node)
			}
			if dir.hot {
				checkML003Call(pass, node)
			}

		case *ast.FuncLit:
			if dir.nogc {
				pass.Reportf(node.Pos(), "ML004: closure in //mem:nogc function may cause heap escape")
			}
			return false // don't recurse into closure body

		case *ast.DeferStmt:
			if dir.nogc {
				pass.Reportf(node.Pos(), "ML004: defer in //mem:nogc function may cause heap escape")
			}

		case *ast.InterfaceType:
			if dir.hot {
				pass.Reportf(node.Pos(), "ML001: interface type in //mem:hot function")
			}

		case *ast.ValueSpec:
			if dir.hot {
				checkML001ValueSpec(pass, node)
			}

		case *ast.Field:
			if dir.hot {
				checkML001Field(pass, node)
			}
		}

		return true
	})

	// Also check function signature for interface{}/any params
	if dir.hot && fn.Type != nil {
		checkML001FuncType(pass, fn.Type)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// ML001: No interface{}/any in //mem:hot functions
// ────────────────────────────────────────────────────────────────────────────

func checkML001FuncType(pass *analysis.Pass, ft *ast.FuncType) {
	checkFieldList(pass, ft.Params, "parameter")
	checkFieldList(pass, ft.Results, "return value")
}

func checkFieldList(pass *analysis.Pass, fl *ast.FieldList, kind string) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		if isInterfaceType(f.Type) {
			pass.Reportf(f.Pos(), "ML001: interface{}/any %s in //mem:hot function", kind)
		}
	}
}

func checkML001ValueSpec(pass *analysis.Pass, spec *ast.ValueSpec) {
	if spec.Type != nil && isInterfaceType(spec.Type) {
		pass.Reportf(spec.Pos(), "ML001: interface{}/any variable in //mem:hot function")
	}
}

func checkML001Field(pass *analysis.Pass, field *ast.Field) {
	if isInterfaceType(field.Type) {
		pass.Reportf(field.Pos(), "ML001: interface{}/any field in //mem:hot function")
	}
}

// isInterfaceType returns true for `interface{}` and `any`.
func isInterfaceType(expr ast.Expr) bool {
	switch t := expr.(type) {
	case *ast.InterfaceType:
		return t.Methods == nil || len(t.Methods.List) == 0
	case *ast.Ident:
		return t.Name == "any"
	}
	return false
}

// ────────────────────────────────────────────────────────────────────────────
// ML002: No make/new/append in //mem:nogc functions
// ────────────────────────────────────────────────────────────────────────────

func checkML002(pass *analysis.Pass, call *ast.CallExpr) {
	name := callName(call)
	switch name {
	case "make", "new", "append":
		pass.Reportf(call.Pos(), "ML002: %s() in //mem:nogc function causes heap allocation", name)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// ML003: No reflect in //mem:hot functions
// ────────────────────────────────────────────────────────────────────────────

func checkML003Call(pass *analysis.Pass, call *ast.CallExpr) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	ident, ok := sel.X.(*ast.Ident)
	if !ok {
		return
	}
	if ident.Name == "reflect" {
		pass.Reportf(call.Pos(), "ML003: reflect.%s() in //mem:hot function", sel.Sel.Name)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// ML005: No ReadField/WriteField (use typed API instead)
// ────────────────────────────────────────────────────────────────────────────

func checkML005(pass *analysis.Pass, call *ast.CallExpr) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	switch sel.Sel.Name {
	case "ReadField", "WriteField":
		pass.Reportf(call.Pos(), "ML005: %s() is deprecated; use TypedRegion[T].Get()/Set() instead", sel.Sel.Name)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

// callName extracts the function name from a CallExpr.
// Returns "" for method calls and other complex expressions.
func callName(call *ast.CallExpr) string {
	switch fn := call.Fun.(type) {
	case *ast.Ident:
		return fn.Name
	case *ast.SelectorExpr:
		return fn.Sel.Name
	}
	return ""
}

// Ensure token.Pos is used (suppress unused import warning).
var _ token.Pos
