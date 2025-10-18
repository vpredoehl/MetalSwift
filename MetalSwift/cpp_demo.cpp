// Deprecated: C++ demo disabled. Use Swift demo in CppDemo.swift instead.
#if 0
#include <cstdio>
#include <vector>

extern "C" {
#include "MetalBridge.h"
}

int run_cpp_demo() {
    std::vector<float> a{1, 2, 3, 4, 5};
    std::vector<float> b{10, 20, 30, 40, 50};
    std::vector<float> out(a.size());

    char errorBuf[256] = {0};

    if (metal_vector_add(a.data(), b.data(), out.data(), (unsigned int)a.size(), errorBuf, 256)) {
        printf("C++ GPU result: [");
        for (size_t i = 0; i < out.size(); ++i) {
            printf("%g", out[i]);
            if (i + 1 < out.size()) {
                printf(", ");
            }
        }
        printf("]\n");
    } else {
        printf("C++ Metal error: %s\n", errorBuf);
    }

    return 0;
}
#endif // disabled
