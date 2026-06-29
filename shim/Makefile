CC ?= cc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -pthread -std=c11
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
    LIB_FLAGS := -dynamiclib -undefined dynamic_lookup
    LD_PRELOAD_VAR := DYLD_INSERT_LIBRARIES
else
    LIB_EXT := so
    LIB_FLAGS := -shared
    LD_PRELOAD_VAR := LD_PRELOAD
endif

BUILD_DIR := build
LIB := $(BUILD_DIR)/liblohalloc_obs.$(LIB_EXT)
SRC := lohalloc_obs.c
OBJ := $(BUILD_DIR)/lohalloc_obs.o
TEST_BIN := $(BUILD_DIR)/test_ring
TEST_OBJ := $(BUILD_DIR)/lohalloc_obs_test.o

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(OBJ)
	$(CC) $(LIB_FLAGS) -o $@ $< -pthread

$(OBJ): $(SRC) lohalloc_obs.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): test_ring.c lohalloc_obs.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DLOHALLOC_OBS_TESTING -c lohalloc_obs.c -o $(TEST_OBJ)
	$(CC) $(CFLAGS) -DLOHALLOC_OBS_TESTING -o $@ test_ring.c $(TEST_OBJ) -pthread

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test clean