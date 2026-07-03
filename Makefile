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
        bench-all bench-all-host bench-image bench-latency bench graphs \
        bench-native bench-cache bench-native-host bench-cache-host bench-rust-bins bench-report

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
	@echo "  make bench-all      — ONE command: run every benchmark + build ALL graphs into results/<timestamp>/ (Docker)"
	@echo "  make bench-all-host — same, but native passes run on the host (no Docker)"
	@echo "  make bench          — full Rust criterion suite + latency profiles + report (Phase 6)"
	@echo "  make bench-native   — native C/C++/Rust cross-allocator timing (Docker, Linux LD_PRELOAD)"
	@echo "  make bench-cache    — native cachegrind cache-miss pass (Docker)"
	@echo "  make bench-report   — aggregate results/<timestamp>/raw into report + graphs (use RUN_DIR=...)"
	@echo "  make graphs RUN_DIR=results/<ts>  — re-render graphs for an existing run"

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

# ONE timestamped run directory per `make` invocation. Every producer writes
# straight into $(RUN_DIR)/raw and the aggregator writes the report + graphs
# beside it — no staging dir, no move. Computed once (`:=`, so the shell runs
# a single time), overridable so separate invocations can share a run:
#     make bench-native RUN_DIR=results/20260101T000000
#     make bench-report RUN_DIR=results/20260101T000000
# `make bench-all` sets it once for the whole pipeline, so you never have to.
RUN_DIR := $(RESULTS_DIR)/$(shell date +%Y%m%dT%H%M%S)

# ============================================================================
# THE single command: run every graph-producing benchmark (native C/C++/Rust
# timing, cachegrind cache metrics, and Rust per-op latency profiles) into one
# $(RUN_DIR), then aggregate + render every graph. Everything lands in
# $(RUN_DIR)/{raw,graphs} + bench-report.{json,md}. Needs Docker for the
# native/cache passes; `bench-all-host` is the no-Docker variant.
# ============================================================================
bench-all: bench-image
	@echo "==> Benchmark run directory: $(RUN_DIR)"
	@mkdir -p $(RUN_DIR)/raw
	@echo "==> [1/4] Rust per-op latency profiles"
	@$(MAKE) --no-print-directory bench-latency RUN_DIR=$(RUN_DIR)
	@echo "==> [2/4] Native timing (C / C++ / Rust)"
	docker run --rm -e RAW_DIR=/lohalloc/$(RUN_DIR)/raw \
		-v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/$(RESULTS_DIR)" lohalloc-bench
	@echo "==> [3/4] Native cache metrics (cachegrind)"
	docker run --rm -e RAW_DIR=/lohalloc/$(RUN_DIR)/raw \
		-v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/$(RESULTS_DIR)" \
		--entrypoint bash lohalloc-bench bench/run_native.sh --cachegrind
	@echo "==> [4/4] Aggregate + render graphs"
	@$(MAKE) --no-print-directory bench-report RUN_DIR=$(RUN_DIR)
	@echo "==> Done. Report + graphs in $(RUN_DIR)"

# No-Docker variant of bench-all (native passes run on the host directly) —
# for a Linux box / CI runner that already has jemalloc/mimalloc/valgrind.
bench-all-host:
	@echo "==> Benchmark run directory: $(RUN_DIR)"
	@mkdir -p $(RUN_DIR)/raw
	@$(MAKE) --no-print-directory bench-latency RUN_DIR=$(RUN_DIR)
	@$(MAKE) --no-print-directory bench-native-host RUN_DIR=$(RUN_DIR)
	@$(MAKE) --no-print-directory bench-cache-host RUN_DIR=$(RUN_DIR)
	@$(MAKE) --no-print-directory bench-report RUN_DIR=$(RUN_DIR)
	@echo "==> Done. Report + graphs in $(RUN_DIR)"

# Build the Docker image once (bench-all reuses it across the timing + cache
# passes instead of rebuilding for each).
bench-image:
	docker build -f docker/Dockerfile.bench -t lohalloc-bench .

# Rust per-op latency profiles (hdrhistogram) across every workload x mode ->
# $(RUN_DIR)/raw. Feeds the rust-latency-p99 graph.
bench-latency:
	@mkdir -p $(RUN_DIR)/raw
	@for workload in slab arena buddy system adv-mixed; do \
		for mode in training inference baseline forced:slab forced:buddy forced:system forced:arena; do \
			out="$(RUN_DIR)/raw/rust_$${workload}_$$(echo $$mode | tr ':' '-').json"; \
			echo "latency_profile $$workload $$mode -> $$out"; \
			cargo run -p lohalloc-bench --bin latency_profile --release -- \
				--workload "$$workload" --mode "$$mode" --ops 100000 --out "$$out" || exit 1; \
		done; \
	done

# Full Rust suite: criterion micro/hypothesis/comparison benches (the
# --save-baseline runs feed criterion's own HTML diff, not the report graphs)
# + latency profiles + aggregate. `bench-all` is usually what you want; this
# is for the criterion baselines specifically.
bench: bench-latency
	cargo bench -p lohalloc-bench --bench backend_micro
	cargo bench -p lohalloc-bench --bench hypothesis
	cargo bench -p lohalloc-bench --bench inference_overhead
	cargo bench -p lohalloc-bench --bench comparison -- --save-baseline system
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-lohalloc -- --save-baseline lohalloc
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-jemalloc -- --save-baseline jemalloc
	cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-mimalloc -- --save-baseline mimalloc
	@$(MAKE) --no-print-directory bench-report RUN_DIR=$(RUN_DIR)

# Native (C/C++/Rust) cross-allocator wall-time comparison via LD_PRELOAD —
# Linux-only, run inside Docker (see docker/Dockerfile.bench). Writes straight
# into $(RUN_DIR)/raw. Standalone use must share RUN_DIR with bench-report
# (bench-all does this for you); the echo prints the exact follow-up command.
bench-native: bench-image
	@mkdir -p $(RUN_DIR)/raw
	docker run --rm -e RAW_DIR=/lohalloc/$(RUN_DIR)/raw \
		-v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/$(RESULTS_DIR)" lohalloc-bench
	@echo "Raw in $(RUN_DIR)/raw — build the report with: make bench-report RUN_DIR=$(RUN_DIR)"

# Cache-miss simulation (cachegrind) for the same native harness — much
# slower than bench-native, so run separately and on demand.
bench-cache: bench-image
	@mkdir -p $(RUN_DIR)/raw
	docker run --rm -e RAW_DIR=/lohalloc/$(RUN_DIR)/raw \
		-v "$(CURDIR)/$(RESULTS_DIR):/lohalloc/$(RESULTS_DIR)" \
		--entrypoint bash lohalloc-bench bench/run_native.sh --cachegrind
	@echo "Raw in $(RUN_DIR)/raw — build the report with: make bench-report RUN_DIR=$(RUN_DIR)"

# Host variants (no Docker) — for the CI runners in infra/remote_bench.sh,
# which already run natively on the provisioned Linux EC2 instances.
# Rust entries in the native matrix: one native_workload binary per
# allocator feature (cargo reuses the bin name across feature sets, hence
# the copy-out to distinct names).
bench-rust-bins:
	@for feat in system lohalloc jemalloc mimalloc; do \
		cargo build -p lohalloc-bench --bin native_workload --release --features alloc-$$feat || exit 1; \
		mkdir -p bench/native/build; \
		cp target/release/native_workload bench/native/build/native_workload_$$feat; \
	done

bench-native-host: bench-rust-bins
	cargo build -p lohalloc-cabi --release
	make -C bench/native
	@mkdir -p $(RUN_DIR)/raw
	RAW_DIR="$(CURDIR)/$(RUN_DIR)/raw" bash bench/run_native.sh

bench-cache-host: bench-rust-bins
	cargo build -p lohalloc-cabi --release
	make -C bench/native
	@mkdir -p $(RUN_DIR)/raw
	RAW_DIR="$(CURDIR)/$(RUN_DIR)/raw" bash bench/run_native.sh --cachegrind

# Aggregate $(RUN_DIR)/raw into $(RUN_DIR)/bench-report.{json,md} and render
# graphs into $(RUN_DIR)/graphs. No staging, no move — reads and writes the
# same run dir. `make graphs RUN_DIR=results/<ts>` regenerates the graphs for
# an existing run without re-benchmarking.
bench-report graphs:
	cargo run -p lohalloc-bench --bin aggregate --release -- --run-dir $(RUN_DIR)

# ---- Clean ----
clean:
	cargo clean
	rm -rf $(SHIM_DIR)/build
	rm -rf $(GUI_DIR)/dist
	rm -rf $(GUI_DIR)/test-results
	rm -rf $(GUI_DIR)/playwright-report
	rm -rf bench/native/build
	rm -rf $(RESULTS_DIR)
