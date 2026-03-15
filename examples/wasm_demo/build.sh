#!/usr/bin/env bash
set -euo pipefail

# Build script for MemPipe WASM Demo
# Compiles the Go WASM binary, copies wasm_exec.js, and optionally starts a server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR"

echo "═══ MemPipe WASM Demo — Build ═══"
echo "  Root:   $ROOT_DIR"
echo "  Output: $OUT_DIR"
echo ""

# 1. Compile WASM binary
echo "→ Building demo.wasm …"
cd "$ROOT_DIR"
GOOS=js GOARCH=wasm go build -o "$OUT_DIR/demo.wasm" ./examples/wasm_demo/
WASM_SIZE=$(wc -c < "$OUT_DIR/demo.wasm")
echo "  demo.wasm: $(numfmt --to=iec $WASM_SIZE 2>/dev/null || echo "$WASM_SIZE bytes")"

# 2. Copy wasm_exec.js
WASM_EXEC="$(go env GOROOT)/lib/wasm/wasm_exec.js"
if [[ ! -f "$WASM_EXEC" ]]; then
  echo "ERROR: wasm_exec.js not found at $WASM_EXEC"
  exit 1
fi
cp "$WASM_EXEC" "$OUT_DIR/wasm_exec.js"
echo "→ Copied wasm_exec.js"

# 3. Symlink gpt2.mpmodel for the server to serve
if [[ -f "$ROOT_DIR/gpt2.mpmodel" ]] && [[ ! -e "$OUT_DIR/gpt2.mpmodel" ]]; then
  ln -s "$ROOT_DIR/gpt2.mpmodel" "$OUT_DIR/gpt2.mpmodel"
  echo "→ Symlinked gpt2.mpmodel"
elif [[ -e "$OUT_DIR/gpt2.mpmodel" ]]; then
  echo "→ gpt2.mpmodel already present"
else
  echo "⚠ gpt2.mpmodel not found at $ROOT_DIR/gpt2.mpmodel — load manually via file picker"
fi

echo ""
echo "═══ Build complete ═══"
echo ""

# 4. Optionally start a server
if [[ "${1:-}" == "--serve" ]]; then
  PORT="${2:-8080}"
  echo "Starting HTTP server on http://localhost:$PORT"
  echo "  Press Ctrl+C to stop."
  echo ""
  cd "$OUT_DIR"

  # Write a tiny Go file-server as last-resort fallback
  _GO_SERVER=$(mktemp /tmp/mempipe_serve_XXXX.go)
  cat > "$_GO_SERVER" <<'EOF'
package main

import (
	"fmt"
	"net/http"
	"os"
)

func main() {
	port := os.Args[1]
	fmt.Printf("Go file server on http://localhost:%s\n", port)
	http.ListenAndServe("localhost:"+port, http.FileServer(http.Dir(".")))
}
EOF

  python3 -m http.server --bind localhost "$PORT" || \
  python -m http.server --bind localhost "$PORT" 2>/dev/null || \
  go run "$_GO_SERVER" "$PORT"

  rm -f "$_GO_SERVER" 2>/dev/null
fi
