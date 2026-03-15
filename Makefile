# MemPipe Makefile
# ────────────────────────────────────────────────────────────────────────────
# Targets:
#   make build        — Build native binary
#   make test         — Run all tests
#   make bench        — Run benchmarks
#   make wasm         — Build WASM module + copy JS assets to dist/
#   make embedded     — Build for embedded target (requires TinyGo)
#   make clean        — Remove build artifacts
#   make all          — Build native + WASM
# ────────────────────────────────────────────────────────────────────────────

# Go toolchain
GO       ?= go
TINYGO   ?= tinygo
GOFLAGS  ?=

# Output directories
DIST     := dist
BIN      := bin

# WASM settings
WASM_OUT   := $(DIST)/mempipe.wasm
WASM_JS    := $(DIST)/mempipe.js
WASM_DTS   := $(DIST)/mempipe.d.ts
WASM_EXEC  := $(DIST)/wasm_exec.js
WASM_PKG   := ./platform/wasm/
WASM_LDFLAGS := -s -w

# Embedded settings
TINYGO_TARGET ?= arduino-nano33
EMBEDDED_OUT  := $(BIN)/mempipe-embedded

# ────────────────────────────────────────────────────────────────────────────
# Default
# ────────────────────────────────────────────────────────────────────────────

.PHONY: all
all: build wasm

# ────────────────────────────────────────────────────────────────────────────
# Native build
# ────────────────────────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "==> Building native binary..."
	@mkdir -p $(BIN)
	$(GO) build $(GOFLAGS) -o $(BIN)/mempipe ./...

# ────────────────────────────────────────────────────────────────────────────
# Test & Bench
# ────────────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(GO) test ./... -count=1

.PHONY: bench
bench:
	$(GO) test ./... -bench=. -benchmem -count=1

.PHONY: test-race
test-race:
	$(GO) test ./... -race -count=1

.PHONY: test-cover
test-cover:
	$(GO) test ./... -coverprofile=coverage.out
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# ────────────────────────────────────────────────────────────────────────────
# WASM build
# ────────────────────────────────────────────────────────────────────────────

.PHONY: wasm
wasm: $(WASM_OUT) $(WASM_JS) $(WASM_EXEC)
	@echo ""
	@echo "==> WASM build complete:"
	@ls -lh $(DIST)/mempipe.wasm
	@echo "    $(WASM_OUT)"
	@echo "    $(WASM_JS)"
	@echo "    $(WASM_DTS)"
	@echo "    $(WASM_EXEC)"

$(WASM_OUT):
	@echo "==> Building WASM module..."
	@mkdir -p $(DIST)
	GOOS=js GOARCH=wasm $(GO) build \
		-ldflags="$(WASM_LDFLAGS)" \
		$(GOFLAGS) \
		-o $(WASM_OUT) \
		$(WASM_PKG)

$(WASM_JS): platform/wasm/js/mempipe.js
	@mkdir -p $(DIST)
	cp platform/wasm/js/mempipe.js $(WASM_JS)
	cp platform/wasm/js/mempipe.d.ts $(WASM_DTS)

$(WASM_EXEC):
	@echo "==> Copying wasm_exec.js from Go installation..."
	@mkdir -p $(DIST)
	cp "$$($(GO) env GOROOT)/misc/wasm/wasm_exec.js" $(WASM_EXEC)

# WASM size report
.PHONY: wasm-size
wasm-size: $(WASM_OUT)
	@echo "==> WASM binary size:"
	@ls -lh $(WASM_OUT)
	@echo ""
	@echo "Breakdown (top symbols):"
	@wasm-objdump -h $(WASM_OUT) 2>/dev/null || echo "(install wasm-objdump for detailed breakdown)"

# ────────────────────────────────────────────────────────────────────────────
# Embedded build (TinyGo)
# ────────────────────────────────────────────────────────────────────────────

.PHONY: embedded
embedded:
	@echo "==> Building for embedded target: $(TINYGO_TARGET)..."
	@mkdir -p $(BIN)
	$(TINYGO) build \
		-target=$(TINYGO_TARGET) \
		-tags embedded \
		-o $(EMBEDDED_OUT) \
		$(WASM_PKG)

# TinyGo WASM (alternative to Go's WASM, much smaller binary)
.PHONY: tinygo-wasm
tinygo-wasm:
	@echo "==> Building TinyGo WASM module..."
	@mkdir -p $(DIST)
	$(TINYGO) build \
		-target=wasm \
		-tags embedded \
		-no-debug \
		-o $(DIST)/mempipe-tiny.wasm \
		$(WASM_PKG)
	@echo "==> TinyGo WASM build complete:"
	@ls -lh $(DIST)/mempipe-tiny.wasm

# ────────────────────────────────────────────────────────────────────────────
# Vet & lint
# ────────────────────────────────────────────────────────────────────────────

.PHONY: vet
vet:
	$(GO) vet ./...

.PHONY: lint
lint:
	@which golangci-lint > /dev/null 2>&1 || \
		echo "Install golangci-lint: https://golangci-lint.run/usage/install/"
	golangci-lint run ./...

.PHONY: memlint
memlint:
	@echo "==> Installing memlint..."
	@cd tools/memlint && $(GO) install ./cmd/memlint/
	@echo "==> Running memlint..."
	$(GO) vet -vettool=$$(which memlint) ./...

# ────────────────────────────────────────────────────────────────────────────
# Examples
# ────────────────────────────────────────────────────────────────────────────

.PHONY: examples
examples:
	@echo "==> Building all examples..."
	$(GO) build ./examples/...
	@echo "==> All examples compile OK"

# ────────────────────────────────────────────────────────────────────────────
# Clean
# ────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(DIST) $(BIN)
	rm -f coverage.out coverage.html

# ────────────────────────────────────────────────────────────────────────────
# Help
# ────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "MemPipe Build Targets:"
	@echo ""
	@echo "  build        Build native binary"
	@echo "  test         Run all tests"
	@echo "  bench        Run benchmarks with -benchmem"
	@echo "  test-race    Run tests with race detector"
	@echo "  test-cover   Generate HTML coverage report"
	@echo "  wasm         Build WASM module to dist/"
	@echo "  wasm-size    Report WASM binary size"
	@echo "  embedded     Build for embedded target (TinyGo)"
	@echo "  tinygo-wasm  Build tiny WASM via TinyGo"
	@echo "  vet          Run go vet"
	@echo "  lint         Run golangci-lint"
	@echo "  memlint      Run memlint zero-alloc analyzer"
	@echo "  examples     Build all examples"
	@echo "  clean        Remove build artifacts"
	@echo "  all          Build native + WASM"
	@echo ""
	@echo "Variables:"
	@echo "  TINYGO_TARGET  Embedded target board (default: arduino-nano33)"
	@echo "  WASM_LDFLAGS   Extra linker flags for WASM (default: -s -w)"
