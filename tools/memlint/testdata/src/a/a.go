package a

import "reflect"

// ════════════════════════════════════════════════════════════════════════════
// ML001: No interface{}/any in //mem:hot functions
// ════════════════════════════════════════════════════════════════════════════

//mem:hot
func hotFuncGood(x int, y float32) int { // OK
	return x + int(y)
}

//mem:hot
func hotFuncBadParam(x interface{}) int { // want `ML001: interface\{\}/any parameter in //mem:hot function`
	return 0
}

//mem:hot
func hotFuncBadAnyParam(x any) int { // want `ML001: interface\{\}/any parameter in //mem:hot function`
	return 0
}

//mem:hot
func hotFuncBadReturn() interface{} { // want `ML001: interface\{\}/any return value in //mem:hot function`
	return nil
}

// Not annotated — should be fine.
func normalFunc(x interface{}) interface{} {
	return x
}

// ════════════════════════════════════════════════════════════════════════════
// ML002: No make/new/append in //mem:nogc functions
// ════════════════════════════════════════════════════════════════════════════

//mem:nogc
func nogcFuncGood() {
	x := 42
	_ = x
}

//mem:nogc
func nogcFuncBadMake() {
	_ = make([]int, 10) // want `ML002: make\(\) in //mem:nogc function causes heap allocation`
}

//mem:nogc
func nogcFuncBadNew() {
	_ = new(int) // want `ML002: new\(\) in //mem:nogc function causes heap allocation`
}

//mem:nogc
func nogcFuncBadAppend() {
	s := []int{1}
	s = append(s, 2) // want `ML002: append\(\) in //mem:nogc function causes heap allocation`
	_ = s
}

// Not annotated — should be fine.
func normalAlloc() {
	_ = make([]int, 10)
	_ = new(int)
}

// ════════════════════════════════════════════════════════════════════════════
// ML003: No reflect in //mem:hot functions
// ════════════════════════════════════════════════════════════════════════════

//mem:hot
func hotFuncBadReflect(x int) int {
	_ = reflect.TypeOf(x) // want `ML003: reflect.TypeOf\(\) in //mem:hot function`
	return x
}

//mem:hot
func hotFuncNoReflect(x int) int { // OK
	return x + 1
}

// Not annotated — reflect is fine.
func normalReflect(x int) {
	_ = reflect.TypeOf(x)
}

// ════════════════════════════════════════════════════════════════════════════
// ML004: No closures or defer in //mem:nogc functions
// ════════════════════════════════════════════════════════════════════════════

//mem:nogc
func nogcFuncBadClosure() {
	f := func() {} // want `ML004: closure in //mem:nogc function may cause heap escape`
	_ = f
}

//mem:nogc
func nogcFuncBadDefer() {
	defer func() {}() // want `ML004: closure in //mem:nogc function may cause heap escape` `ML004: defer in //mem:nogc function may cause heap escape`
}

//mem:nogc
func nogcFuncGoodNoClosure() { // OK
	x := 42
	_ = x
}

// Not annotated — closures and defer are fine.
func normalClosure() {
	f := func() {}
	defer f()
}

// ════════════════════════════════════════════════════════════════════════════
// ML005: No ReadField/WriteField (use typed API instead)
// ════════════════════════════════════════════════════════════════════════════

type FakeRegion struct{}

func (r *FakeRegion) ReadField(name string) int       { return 0 }
func (r *FakeRegion) WriteField(name string, val int) {}

func badReadField() {
	r := &FakeRegion{}
	r.ReadField("x") // want `ML005: ReadField\(\) is deprecated; use TypedRegion\[T\].Get\(\)/Set\(\) instead`
}

func badWriteField() {
	r := &FakeRegion{}
	r.WriteField("x", 42) // want `ML005: WriteField\(\) is deprecated; use TypedRegion\[T\].Get\(\)/Set\(\) instead`
}

// ════════════════════════════════════════════════════════════════════════════
// Combined: //mem:hot + //mem:nogc
// ════════════════════════════════════════════════════════════════════════════

//mem:hot
//mem:nogc
func hotNogcCombined(x int) int {
	_ = make([]int, 1) // want `ML002: make\(\) in //mem:nogc function causes heap allocation`
	return x
}

//mem:hot
//mem:nogc
func hotNogcClean(x int) int { // OK — no violations
	y := x + 1
	return y
}
