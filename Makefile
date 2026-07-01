# Lohalloc root Makefile
#
# Convenience entry points for common workflows. The actual build is driven
# by Cargo (workspace at the repo root) and the shim's own Makefile at
# shim/Makefile.
#
# Usage:
#   make help                 — show available targets
#   make all                  — build everything (shim + workspace + GUI)
#   make shim                 — build the C telemetry shim
#   make example-sink         — build lohalloc-example with the shim sink
#   make demo-sink            — build lohalloc-demo with the shim sink
#   make dev                  — start the server (release) + Vite dev server
#   make test                 — run all Rust tests
#   make e2e                  — run Playwright e2e tests
#   make clean                — cargo clean + remove shim/build/

# ---- Paths ----
SHIM_DIR        := shim
SHIM_BUILD      := $(SHIM_DIR)/build/liblohalloc_obs$(if $(filter Darwin,$(UNAME_S)),.dylib,.so)
GUI_DIR         := gui
TARGET_DIR      := target
EXAMPLE_BIN     := $(TARGET_DIR)/release/lohalloc-example
DEMO_BIN        := $(TARGET_DIR)/release/lohalloc-demo

UNAME_S := $(shell uname -s)

# On macOS, use DYLD_INSERT_LIBRARIES; on Linux, LD_PRELOAD.
ifeq ($(UNAME_S),Darwin)
    LD_PRELOAD_VAR := DYLD_INSERT_LIBRARIES
else
    LD_PRELOAD_VAR := LD_PRELOAD
endif

# ---- Phony targets ----
.PHONY: help all shim example-sink demo-sink binaries \
        server server-debug gui dev test e2e lint fmt clean

help:
	@echo "Lohalloc — available targets:"
	@echo "  make shim           — build the C telemetry shim (shim/build/liblohalloc_obs.{so,dylib})"
	@echo "  make example-sink   — build lohalloc-example with install-shim-sink feature"
	@echo "  make demo-sink      — build lohalloc-demo with install-shim-sink feature"
	@echo "  make binaries       — build all cargo binaries (debug)"
	@echo "  make all            — shim + example-sink + demo-sink + cargo build --release"
	@echo "  make server         — run lohalloc-server (release) on :3000"
	@echo "  make server-debug   — run lohalloc-server (debug) on :3000"
	@echo "  make gui            — start Vite dev server (port 5173)"
	@echo "  make dev            — shim + binaries + server + gui"
	@echo "  make test           — cargo test (all crates)"
	@echo "  make e2e            — Playwright e2e tests (assumes server + gui running)"
	@echo "  make lint           — cargo clippy --all-targets --workspace"
	@echo "  make fmt            — cargo fmt --all"
	@echo "  make clean          — cargo clean + remove shim/build/"

# ---- Builds ----
shim:
	@$(MAKE) -C $(SHIM_DIR)

example-sink: shim
	cargo build --release -p lohalloc-example --features install-shim-sink
	@echo "[ok] $(EXAMPLE_BIN) built with shim sink support"

demo-sink: shim
	cargo build --release -p lohalloc-demo --features install-shim-sink
	@echo "[ok] $(DEMO_BIN) built with shim sink support"

binaries:
	cargo build --workspace

all: shim example-sink demo-sink
	cargo build --release --workspace

# ---- Run ----
server:
	cargo run --release -p lohalloc-server -- --port 3000

server-debug:
	cargo run -p lohalloc-server -- --port 3000

gui:
	cd $(GUI_DIR) && npm run dev

# Build the pieces, then run server + GUI together.
dev: all
	@echo "Starting server (release) and Vite dev server in parallel."
	@echo "Use Ctrl-C to stop both."
	@trap 'kill 0' INT TERM EXIT; \
	( cargo run --release -p lohalloc-server -- --port 3000 & ) ; \
	( cd $(GUI_DIR) && npm run dev & ) ; \
	wait

# ---- Quality ----
test:
	cargo test --workspace

e2e:
	cd $(GUI_DIR) && npx playwright test

lint:
	cargo clippy --all-targets --workspace -- -D warnings

fmt:
	cargo fmt --all

# ---- Clean ----
clean:
	cargo clean
	rm -rf $(SHIM_DIR)/build
	rm -rf $(GUI_DIR)/dist
	rm -rf $(GUI_DIR)/test-results
	rm -rf $(GUI_DIR)/playwright-report
