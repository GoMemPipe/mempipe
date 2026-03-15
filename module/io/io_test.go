//go:build !wasm && !embedded
// +build !wasm,!embedded

package io

import (
	"strings"
	"testing"

	"github.com/GoMemPipe/mempipe/runtime"
)

func newTestIOModule(t *testing.T) (*IOModule, *runtime.PipeManager) {
	t.Helper()
	pm := runtime.NewPipeManager()
	m := NewIOModule(pm)
	return m, pm
}

func TestPipeWrite(t *testing.T) {
	pipe := runtime.NewMemoryPipe("test", 4096)

	err := pipe.Write("Hello")
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	result := pipe.Read()
	if result != "Hello" {
		t.Errorf("Expected 'Hello', got '%s'", result)
	}
}

func TestPipeWriteln(t *testing.T) {
	pipe := runtime.NewMemoryPipe("test", 4096)

	err := pipe.Writeln("Hello")
	if err != nil {
		t.Fatalf("Writeln failed: %v", err)
	}

	result := pipe.Read()
	if result != "Hello\n" {
		t.Errorf("Expected 'Hello\\n', got '%s'", result)
	}
}

func TestPipeMultipleWrites(t *testing.T) {
	pipe := runtime.NewMemoryPipe("test", 4096)

	pipe.Write("Line 1")
	pipe.Write("\n")
	pipe.Write("Line 2")

	result := pipe.Read()
	expected := "Line 1\nLine 2"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}
}

func TestIOModulePrint(t *testing.T) {
	m, _ := newTestIOModule(t)

	err := m.Println("Hello", "World")
	if err != nil {
		t.Fatalf("Println failed: %v", err)
	}

	result := m.ReadStdout()
	if !strings.Contains(result, "Hello") || !strings.Contains(result, "World") {
		t.Errorf("Expected 'Hello World', got '%s'", result)
	}
}

func TestIOModulePrintln(t *testing.T) {
	m, _ := newTestIOModule(t)

	err := m.Println("Test", "Message")
	if err != nil {
		t.Fatalf("Println failed: %v", err)
	}

	result := m.ReadStdout()
	if !strings.HasSuffix(result, "\n") {
		t.Errorf("Expected newline at end, got '%s'", result)
	}
}

func TestIOModuleWrite(t *testing.T) {
	m, _ := newTestIOModule(t)

	err := m.Write("Test")
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	result := m.ReadStdout()
	if result != "Test" {
		t.Errorf("Expected 'Test', got '%s'", result)
	}
}

func TestIOModuleWriteln(t *testing.T) {
	m, _ := newTestIOModule(t)

	err := m.Writeln("Test")
	if err != nil {
		t.Fatalf("Writeln failed: %v", err)
	}

	result := m.ReadStdout()
	if result != "Test\n" {
		t.Errorf("Expected 'Test\\n', got '%s'", result)
	}
}

func TestIOModuleEprint(t *testing.T) {
	m, _ := newTestIOModule(t)

	err := m.Eprint("Error", "Message")
	if err != nil {
		t.Fatalf("Eprint failed: %v", err)
	}

	result := m.ReadStderr()
	if !strings.Contains(result, "Error") || !strings.Contains(result, "Message") {
		t.Errorf("Expected 'Error Message', got '%s'", result)
	}
}

func TestPipeConcurrency(t *testing.T) {
	pipe := runtime.NewMemoryPipe("test", 4096)

	done := make(chan bool)

	for i := 0; i < 10; i++ {
		go func(n int) {
			pipe.Write(string(rune('A' + n)))
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	result := pipe.Read()
	if len(result) != 10 {
		t.Errorf("Expected 10 characters, got %d", len(result))
	}
}
