# Native (C/C++) benchmark harness. Mirrors shim/Makefile's style: no
# generator, hand-written, portable cc/c++ flags.
CC ?= cc
CXX ?= c++
CFLAGS ?= -O2 -Wall -Wextra
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17
# The mt-* workloads (workload_mt_slab_churn/workload_mt_adversarial_mixed/
# workload_mt_xfree) use pthreads directly.
LDLIBS := -lpthread
BUILD_DIR := build

.PHONY: all clean

all: $(BUILD_DIR)/bench_main_c $(BUILD_DIR)/bench_main_cpp

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/workloads_c.o: workloads.c workloads.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c workloads.c -o $@

$(BUILD_DIR)/bench_main_c: bench_main.c $(BUILD_DIR)/workloads_c.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) bench_main.c $(BUILD_DIR)/workloads_c.o $(LDLIBS) -o $@

$(BUILD_DIR)/bench_main_cpp: bench_main.cpp $(BUILD_DIR)/workloads_c.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) bench_main.cpp $(BUILD_DIR)/workloads_c.o $(LDLIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)
