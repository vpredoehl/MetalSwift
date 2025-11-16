#include <vector>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <mutex>
#include "CppGpuBridge.h"
#include "CppGpuMatMulAdd.h"

// Forward declarations for new bridge functions (to be implemented in Objective-C++):
extern "C" int cpp_gpu_vector_add_shared(const float *a, const float *b, float *out, size_t count);
extern "C" int cpp_gpu_matmul_add_shared(const float* A, size_t rowsA, size_t colsA,
                       const float* B, size_t rowsB, size_t colsB,
                       const float* D, float* C);
extern "C" void cpp_gpu_bridge_init_shared();

static void runVectorAddDemo() {
    const size_t size = 10000;
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> out(size, 0.0f);

    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    int status = cpp_gpu_vector_add_shared(a.data(), b.data(), out.data(), size);
    if (status != 0) {
        std::cerr << "cpp_gpu_vector_add_shared failed with status: " << status << std::endl;
        return;
    }

    std::printf("VectorAdd: [");
    // Print first 5 elements
    for (size_t i = 0; i < 5; ++i) {
        std::printf("%g, ", out[i]);
    }
    std::printf("... ");
    // Print last 5 elements
    for (size_t i = size - 5; i < size; ++i) {
        std::printf("%g%s", out[i], (i + 1 < size) ? ", " : "]\n");
    }
}

static void runMatMulAddDemo() {
    // Larger square matrix multiply-add demo: 256x256 * 256x256
    const size_t rowsA = 256, colsA = 256;
    const size_t rowsB = 256, colsB = 256;
    const size_t rowsC = rowsA, colsC = colsB;

    std::vector<float> A(rowsA * colsA);
    std::vector<float> B(rowsB * colsB);
    std::vector<float> D(rowsC * colsC);
    std::vector<float> C(rowsC * colsC, 0.0f);

    for (size_t i = 0; i < rowsA * colsA; ++i) {
        A[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < rowsB * colsB; ++i) {
        B[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < rowsC * colsC; ++i) {
        D[i] = static_cast<float>(i) * 0.001f;  // small values for D
    }

    int status = cpp_gpu_matmul_add_shared(A.data(), rowsA, colsA,
                                          B.data(), rowsB, colsB,
                                          D.data(),
                                          C.data());
    if (status != 0) {
        std::cerr << "cpp_gpu_matmul_add_shared failed with status: " << status << std::endl;
        return;
    }

    // Print top-left 3x3 block of C
    std::printf("MatMul+Add C top-left 3x3 block:\n");
    for (size_t r = 0; r < 3; ++r) {
        std::printf("[");
        for (size_t c = 0; c < 3; ++c) {
            std::printf("%g%s", C[r * colsC + c], (c + 1 < 3) ? ", " : "]\n");
        }
    }

    // Print last row of C
    std::printf("MatMul+Add C last row [%zu]: [", rowsC - 1);
    for (size_t c = 0; c < colsC; ++c) {
        std::printf("%g%s", C[(rowsC - 1) * colsC + c], (c + 1 < colsC) ? ", " : "]\n");
    }
}

// Updated stress test to use shared bridge functions without device/queue pointers
static void runGpuStressTest(double seconds) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    size_t iterations = 0;

    // Prepare large vector add data once to avoid repeated allocation
    const size_t largeVecSize = 1000000;
    std::vector<float> a(largeVecSize);
    std::vector<float> b(largeVecSize);
    std::vector<float> out(largeVecSize, 0.0f);

    for (size_t i = 0; i < largeVecSize; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Prepare large matrices once for matmul+add
    const size_t rowsA = 256, colsA = 256;
    const size_t rowsB = 256, colsB = 256;
    const size_t rowsC = rowsA, colsC = colsB;

    std::vector<float> A(rowsA * colsA);
    std::vector<float> B(rowsB * colsB);
    std::vector<float> D(rowsC * colsC);
    std::vector<float> C(rowsC * colsC, 0.0f);

    for (size_t i = 0; i < rowsA * colsA; ++i) {
        A[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < rowsB * colsB; ++i) {
        B[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < rowsC * colsC; ++i) {
        D[i] = static_cast<float>(i) * 0.001f;
    }

    while (true) {
        // Run large vector add with shared bridge function
        int status = cpp_gpu_vector_add_shared(a.data(), b.data(), out.data(), largeVecSize);
        if (status != 0) {
            std::cerr << "cpp_gpu_vector_add_shared failed during stress test with status: " << status << std::endl;
            break;
        }

        // Run large matmul+add with shared bridge function
        status = cpp_gpu_matmul_add_shared(A.data(), rowsA, colsA,
                                          B.data(), rowsB, colsB,
                                          D.data(),
                                          C.data());
        if (status != 0) {
            std::cerr << "cpp_gpu_matmul_add_shared failed during stress test with status: " << status << std::endl;
            break;
        }

        ++iterations;

        auto now = clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (elapsed.count() >= seconds) {
            std::printf("GPU Stress Test completed %zu iterations in %.3f seconds.\n", iterations, elapsed.count());
            break;
        }
    }
}

extern "C" void cpp_run_gpu_demos() {
    cpp_gpu_bridge_init_shared();

    runVectorAddDemo();
    runMatMulAddDemo();

    // Run fixed-time stress test of all GPU cores for 60 seconds
    runGpuStressTest(10.0);
}
