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
        server server-debug gui dev test e2e lint fmt clean \
        bench bench-native bench-cache bench-native-host bench-cache-host bench-report

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
	@echo "  make bench          — Rust criterion + latency_profile hypothesis suite (Phase 6)"
	@echo "  make bench-native   — native C/C++ cross-allocator timing (Docker, Linux LD_PRELOAD)"
	@echo "  make bench-cache    — native cachegrind cache-miss pass (Docker)"
	@echo "  make bench-report   — consolidate results/raw/*.json into results/<timestamp>/ (report + graphs)"

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

# ---- Phase 6: hypothesis-validation benchmarking ----
RESULTS_DIR := results

# Layer 1 (forced-routing) criterion benches, Layer 2 hypothesis benches,
# and per-op latency profiles across every workload x mode combination.
# Cross-allocator `comparison` runs once per allocator so criterion can
# diff baselines (--save-baseline). See crates/lohalloc-bench.
bench:
	mkdir -p $(RESULTS_DIR)/raw
	cargo bench -p lohalloc-bench --bench backend_micro
	cargo bench -p lohalloc-bench --bench hypothesis
	cargo bench -p lohalloc-bench --bench inference_overhead
	cargo bench -p lohalloc-bench --bench comparison -- --save-baseline system
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-lohalloc -- --save-baseline lohalloc
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-jemalloc -- --save-baseline jemalloc
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-mimalloc -- --save-baseline mimalloc
	@for workload in slab arena buddy system adv-mixed; do \
		for mode in training inference baseline forced:slab forced:buddy forced:system forced:arena; do \
			out="$(RESULTS_DIR)/raw/rust_$${workload}_$$(echo $$mode | tr ':' '-').json"; \
			echo "latency_profile $$workload $$mode -> $$out"; \
			cargo run -p lohalloc-bench --bin latency_profile --release -- \
				--workload "$$workload" --mode "$$mode" --ops 100000 --out "$$out" || exit 1; \
		done; \
	done
	$(MAKE) bench-report

# Native (C/C++) cross-allocator wall-time comparison via LD_PRELOAD —
# Linux-only, run inside Docker (see docker/Dockerfile.bench) even on a
# Linux host, for a consistent, isolated environment.
bench-native:
	docker build -f docker/Dockerfile.bench -t lohalloc-bench .
	mkdir -p $(RESULTS_DIR)/raw
	docker run --rm -v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/results" lohalloc-bench

# Cache-miss simulation (cachegrind) for the same native harness — much
# slower than bench-native, so run separately and on demand.
bench-cache:
	docker build -f docker/Dockerfile.bench -t lohalloc-bench .
	mkdir -p $(RESULTS_DIR)/raw
	docker run --rm -v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/results" \
		--entrypoint bash lohalloc-bench bench/run_native.sh --cachegrind

# Host variants (no Docker) — for the CI runners in infra/remote_bench.sh,
# which already run natively on the provisioned Linux EC2 instances.
bench-native-host:
	cargo build -p lohalloc-cabi --release
	make -C bench/native
	mkdir -p $(RESULTS_DIR)/raw
	bash bench/run_native.sh

bench-cache-host:
	cargo build -p lohalloc-cabi --release
	make -C bench/native
	mkdir -p $(RESULTS_DIR)/raw
	bash bench/run_native.sh --cachegrind

# Consolidation step: moves results/raw/*.json into a fresh
# results/<timestamp>/raw/, writes bench-report.{json,md} beside it, and
# renders graphs into results/<timestamp>/graphs/ via a Python venv.
bench-report:
	cargo run -p lohalloc-bench --bin aggregate --release -- --results-dir $(RESULTS_DIR)

# ---- Clean ----
clean:
	cargo clean
	rm -rf $(SHIM_DIR)/build
	rm -rf $(GUI_DIR)/dist
	rm -rf $(GUI_DIR)/test-results
	rm -rf $(GUI_DIR)/playwright-report
	rm -rf bench/native/build
	rm -rf $(RESULTS_DIR)
