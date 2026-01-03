#include <vector>
#include <iostream>
#include <cstdio>
#include "CppGpuBridge.h"
#include "CppGpuMatMulAdd.h"

static void printVector(const char* name, const std::vector<float>& v) {
    std::printf("%s: [", name);
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%g%s", v[i], (i + 1 < v.size()) ? ", " : "]\n");
    }
}

static void printMatrix(const char* name, const std::vector<float>& m, size_t rows, size_t cols) {
    std::printf("%s (%zux%zu):\n", name, rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        std::printf("  %s[%zu]: [", name, r);
        for (size_t c = 0; c < cols; ++c) {
            std::printf("%g%s", m[r * cols + c], (c + 1 < cols) ? ", " : "]\n");
        }
    }
}

static void printDotProductBreakdown(const std::vector<float>& A, size_t rowsA, size_t colsA,
                                     const std::vector<float>& B, size_t rowsB, size_t colsB,
                                     size_t r, size_t c) {
    // Computes C[r,c] = sum_{k=0..colsA-1} A[r,k] * B[k,c]
    std::printf("  C[%zu,%zu] = ", r, c);
    for (size_t k = 0; k < colsA; ++k) {
        float a = A[r * colsA + k];
        float b = B[k * colsB + c];
        std::printf("%s(%g*%g)", (k==0?"":" + "), a, b);
    }
}

static void printTensor3DSample(const char* name, const float* data, int B, int W, int F, int maxB = 2, int maxW = 2, int maxF = 4) {
    if (!data) { std::printf("%s: <null>\n", name); return; }
    if (B <= 0 || W <= 0 || F <= 0) { std::printf("%s: <empty>\n", name); return; }
    std::printf("%s sample (B=%d, W=%d, F=%d):\n", name, B, W, F);
    int sb = (B < maxB) ? B : maxB;
    int sw = (W < maxW) ? W : maxW;
    int sf = (F < maxF) ? F : maxF;
    for (int b = 0; b < sb; ++b) {
        for (int w = 0; w < sw; ++w) {
            std::printf("  %s[b=%d,w=%d]: [", name, b, w);
            for (int f = 0; f < sf; ++f) {
                size_t idx = (size_t)b * (size_t)W * (size_t)F + (size_t)w * (size_t)F + (size_t)f;
                std::printf("%s%.6g", (f==0?"":" "), data[idx]);
            }
            if (F > sf) std::printf(" ...");
            std::printf("]\n");
        }
        if (W > sw) std::printf("  ... (more w)\n");
    }
    if (B > sb) std::printf("  ... (more b)\n");
}

void printTensor3DSample(const char* name, const std::vector<float>& data, int B, int W, int F, int maxB = 2, int maxW = 2, int maxF = 4) {
    const float* ptr = data.empty() ? nullptr : data.data();
    printTensor3DSample(name, ptr, B, W, F, maxB, maxW, maxF);
}

static void printMatrixSample(const char* name, const float* data, int rows, int cols, int maxRows = 4, int maxCols = 4) {
    if (!data) { std::printf("%s: <null>\n", name); return; }
    if (rows <= 0 || cols <= 0) { std::printf("%s: <empty>\n", name); return; }
    std::printf("%s sample (rows=%d, cols=%d):\n", name, rows, cols);
    int sr = (rows < maxRows) ? rows : maxRows;
    int sc = (cols < maxCols) ? cols : maxCols;
    for (int r = 0; r < sr; ++r) {
        std::printf("  %s[r=%d]: [", name, r);
        for (int c = 0; c < sc; ++c) {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            std::printf("%s%.6g", (c==0?"":" "), data[idx]);
        }
        if (cols > sc) std::printf(" ...");
        std::printf("]\n");
    }
    if (rows > sr) std::printf("  ... (more rows)\n");
}

void printMatrixSample(const char* name, const std::vector<float>& data, int rows, int cols, int maxRows = 4, int maxCols = 4) {
    const float* ptr = data.empty() ? nullptr : data.data();
    printMatrixSample(name, ptr, rows, cols, maxRows, maxCols);
}

static void runVectorAddDemo() {
    std::vector<float> a = {1, 2, 3, 4, 5};
    std::vector<float> b = {10, 20, 30, 40, 50};
    std::vector<float> out(a.size(), 0.0f);

    printVector("a", a);
    printVector("b", b);

    int status = cpp_gpu_vector_add(a.data(), b.data(), out.data(), a.size());
    if (status != 0) {
        std::cerr << "cpp_gpu_vector_add failed with status: " << status << std::endl;
        return;
    }

    // Show element-wise addition breakdown
    std::printf("VectorAdd breakdown (out[i] = a[i] + b[i]):\n");
    for (size_t i = 0; i < out.size(); ++i) {
        std::printf("  out[%zu] = %g + %g = %g\n", i, a[i], b[i], out[i]);
    }
}

static void runMatMulAddDemo() {
    const size_t rowsA = 2, colsA = 3;
    const size_t rowsB = 3, colsB = 2;
    const size_t rowsC = rowsA, colsC = colsB;

    std::vector<float> A = { 1, 2, 3, 4, 5, 6 };
    std::vector<float> B = { 7, 8, 9, 10, 11, 12 };
    std::vector<float> D = { 0.5f, 1.5f, 2.5f, 3.5f };
    std::vector<float> C(rowsC * colsC, 0.0f);

    printMatrix("A", A, rowsA, colsA);
    printMatrix("B", B, rowsB, colsB);
    printVector("D", D);

    int status = cpp_gpu_matmul_add(A.data(), rowsA, colsA,
                                    B.data(), rowsB, colsB,
                                    D.data(),
                                    C.data());
    if (status != 0) {
        std::cerr << "cpp_gpu_matmul_add failed with status: " << status << std::endl;
        return;
    }

    // Show per-element breakdown: C[r,c] = sum_k A[r,k]*B[k,c] + D[r*colsC + c]
    std::printf("MatMul+Add breakdown (C[r,c] = A[r,:]·B[:,c] + D[r,c]):\n");
    for (size_t r = 0; r < rowsC; ++r) {
        for (size_t c = 0; c < colsC; ++c) {
            printDotProductBreakdown(A, rowsA, colsA, B, rowsB, colsB, r, c);
            float sum = 0.0f;
            for (size_t k = 0; k < colsA; ++k) {
                sum += A[r * colsA + k] * B[k * colsB + c];
            }
            float d = D[r * colsC + c];
            std::printf(" + %g = %g\n", d, sum + d);
        }
    }

    // Print final C as a matrix
    printMatrix("C", C, rowsC, colsC);
}

static void runTensordotDemoCPU() {
    // Small demo sizes
    const int B = 2, W = 2, F = 3, O = 2;
    // bwo shaped (B,W,F)
    std::vector<float> bwo = {
        // b=0
        1, 2, 3,   // w=0, f=0..2
        4, 5, 6,   // w=1
        // b=1
        7, 8, 9,   // w=0
        10, 11, 12 // w=1
    };
    // fo shaped (F,O)
    std::vector<float> fo = {
        0.5f, 1.0f,
        1.5f, 2.0f,
        2.5f, 3.0f
    };

    // Output (B,W,O)
    std::vector<float> C((size_t)B * (size_t)W * (size_t)O, 0.0f);

    // Print input samples
    printTensor3DSample("bwo", bwo, B, W, F);
    printMatrixSample("fo", fo, F, O);

    std::printf("Tensordot breakdown (C[b,w,o] = sum_f bwo[b,w,f]*fo[f,o]):\n");
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            for (int o = 0; o < O; ++o) {
                std::printf("  C[b=%d,w=%d,o=%d] = ", b, w, o);
                float sum = 0.0f;
                for (int f = 0; f < F; ++f) {
                    size_t idxBWO = (size_t)b * (size_t)W * (size_t)F + (size_t)w * (size_t)F + (size_t)f;
                    size_t idxFO  = (size_t)f * (size_t)O + (size_t)o;
                    float a = bwo[idxBWO];
                    float bval = fo[idxFO];
                    sum += a * bval;
                    std::printf("%s(%g*%g)", (f==0?"":" + "), a, bval);
                }
                size_t idxOut = (size_t)b * (size_t)W * (size_t)O + (size_t)w * (size_t)O + (size_t)o;
                C[idxOut] = sum;
                std::printf(" = %g\n", sum);
            }
        }
    }

    // Print final C as (B,W,O) rows per (b,w)
    std::printf("C (B=%d, W=%d, O=%d) sample:\n", B, W, O);
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            std::printf("  C[b=%d,w=%d]: [", b, w);
            for (int o = 0; o < O; ++o) {
                size_t idxOut = (size_t)b * (size_t)W * (size_t)O + (size_t)w * (size_t)O + (size_t)o;
                std::printf("%s%g", (o==0?"":" "), C[idxOut]);
            }
            std::printf("]\n");
        }
    }
}

extern "C" void cpp_run_gpu_demos() {
    runVectorAddDemo();
    runMatMulAddDemo();
    runTensordotDemoCPU();
}
