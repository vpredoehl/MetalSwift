#import "CppGpuMatMulAdd.h"
#import <limits.h>
#include <stdint.h>

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <vector>
#include <iostream>

#include <iostream>
#include <iomanip>

void PrintMatrix(const char* name,
                 id<MTLBuffer> buffer,
                 int rows,
                 int cols)
{
    float* data = (float*)buffer.contents;

    std::cout << name << " (" << rows << "x" << cols << ")\n";

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            std::cout << std::setw(10)
                      << std::setprecision(5)
                      << data[r * cols + c]
                      << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}

extern "C" void runCppMatMulAddDemoMetal()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (!device || !queue)
        {
            std::cout << "Metal not available\n";
            return;
        }

        const int M = 4;
        const int K = 8;
        const int N = 3;

        size_t aCount = M * K;
        size_t bCount = K * N;
        size_t cCount = M * N;
        size_t biasCount = N;

        // Allocate GPU buffers (shared for demo simplicity)
        id<MTLBuffer> aBuf = [device newBufferWithLength:aCount*sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> bBuf = [device newBufferWithLength:bCount*sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> cBuf = [device newBufferWithLength:cCount*sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> biasBuf = [device newBufferWithLength:biasCount*sizeof(float)
                                                    options:MTLResourceStorageModeShared];

        float* A = (float*)aBuf.contents;
        float* B = (float*)bBuf.contents;
        float* C = (float*)cBuf.contents;
        float* bias = (float*)biasBuf.contents;

        // Fill demo data
        for (size_t i = 0; i < aCount; ++i) A[i] = float(i % 7) * 0.1f;
        for (size_t i = 0; i < bCount; ++i) B[i] = float((i * 3) % 11) * 0.05f;
        for (size_t i = 0; i < biasCount; ++i) bias[i] = 0.01f * float(i + 1);
        for (size_t i = 0; i < cCount; ++i) C[i] = 0.0f;
        
        PrintMatrix("A", aBuf, M, K);
        PrintMatrix("B", bBuf, K, N);
        PrintMatrix("Bias", biasBuf, 1, N);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];

        // ---- MPS Matrix Multiply ----

        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                  columns:K
                                                 rowBytes:K*sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                  columns:N
                                                 rowBytes:N*sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                  columns:N
                                                 rowBytes:N*sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:descA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:descB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:descC];

        MPSMatrixMultiplication* gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                               transposeLeft:NO
                                              transposeRight:NO
                                                 resultRows:M
                                              resultColumns:N
                                            interiorColumns:K
                                                      alpha:1.0
                                                       beta:0.0];

        [gemm encodeToCommandBuffer:cmd
                          leftMatrix:mA
                         rightMatrix:mB
                        resultMatrix:mC];

        [cmd commit];
        [cmd waitUntilCompleted];

        // ---- CPU Add Bias (for now, keep simple) ----
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c)
                C[r*N + c] += bias[c];

        // Print result
        std::cout << "Result C:\n";
        for (int r = 0; r < M; ++r)
        {
            for (int c = 0; c < N; ++c)
                std::cout << C[r*N + c] << " ";
            std::cout << "\n";
        }
    }
}

// Declare the Swift thunk symbol
extern "C" int32_t swift_gpu_matmul_add(const float* A, int32_t rowsA, int32_t colsA,
                                        const float* B, int32_t rowsB, int32_t colsB,
                                        const float* D,
                                        float* C);


int cpp_gpu_matmul_add(const float* A, size_t rowsA, size_t colsA,
                       const float* B, size_t rowsB, size_t colsB,
                       const float* D,
                       float* C) {
    if (!A || !B || !C || !D) return 9;
    if (colsA != rowsB) return 4;
    if (rowsA > INT32_MAX || colsA > INT32_MAX || rowsB > INT32_MAX || colsB > INT32_MAX) return 3;

    int32_t status = swift_gpu_matmul_add(A, (int32_t)rowsA, (int32_t)colsA,
                                          B, (int32_t)rowsB, (int32_t)colsB,
                                          D,
                                          C);
    return (int)status;
}

