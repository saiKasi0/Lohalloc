// Native (C++) benchmark driver: runs one workload, once, then exits.
// Reuses the C workload generators (raw malloc/free — exercised so the raw
// allocation shape is comparable across languages) and adds a couple of
// genuinely C++ allocation patterns (std::vector growth, std::string
// building) so the comparison isn't purely "C called from a .cpp file".
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "workloads.h"

namespace {

// C++ container equivalent of W-SLAB: repeated std::vector<char> churn at a
// fixed small size, bounded live window -- exercises the allocator through
// libstdc++/libc++'s allocator machinery, not raw malloc/free directly.
void cpp_vector_churn(size_t ops) {
    std::vector<std::vector<char>> window;
    window.reserve(64);
    for (size_t i = 0; i < ops; i++) {
        std::vector<char> v(SMALL_FIXED_REQUEST, 'x');
        if (window.size() == 64) {
            window.erase(window.begin());
        }
        window.push_back(std::move(v));
    }
}

// C++ std::string build/grow pattern -- repeated append forces reallocation
// through the allocator at a variety of transient sizes.
void cpp_string_build(size_t ops) {
    for (size_t i = 0; i < ops; i++) {
        std::string s;
        for (int j = 0; j < 64; j++) {
            s += "0123456789abcdef";
        }
        // Prevent the optimizer from eliding the whole loop.
        if (s.empty()) {
            std::abort();
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <slab|arena|buddy|system|adv-mixed|mt-slab-tN|mt-mixed-tN|mt-xfree-tN|mt-interfere-tN"
                "|cpp-vector|cpp-string> [ops]\n",
                argv[0]);
        return 2;
    }
    const std::string workload = argv[1];
    size_t ops = argc >= 3 ? static_cast<size_t>(strtoull(argv[2], nullptr, 10)) : 50000;

    if (workload == "cpp-vector") {
        cpp_vector_churn(ops);
    } else if (workload == "cpp-string") {
        cpp_string_build(ops);
    } else if (!dispatch_workload(workload.c_str(), ops)) {
        fprintf(stderr, "unknown workload '%s'\n", workload.c_str());
        return 2;
    }
    return 0;
}
