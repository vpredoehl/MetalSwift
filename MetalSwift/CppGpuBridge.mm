#import <Metal/Metal.h>
#include "CppGpuBridge.h"
#include <mutex>

static id<MTLDevice> gSharedDevice = nil;
static id<MTLCommandQueue> gSharedQueue = nil;
static std::once_flag gInitFlag;

static void initMetalObjects() {
    gSharedDevice = MTLCreateSystemDefaultDevice();
    if (gSharedDevice) {
        gSharedQueue = [gSharedDevice newCommandQueue];
    }
}

extern "C" void cpp_gpu_bridge_init_shared() {
    std::call_once(gInitFlag, initMetalObjects);
}

extern "C" int cpp_gpu_vector_add_shared(const float *a, const float *b, float *out, size_t count) {
    std::call_once(gInitFlag, initMetalObjects);
    id<MTLDevice> device = gSharedDevice;
    id<MTLCommandQueue> queue = gSharedQueue;
    if (!device || !queue) return 2;
    NSError *error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) return 3;
    id<MTLFunction> fn = [library newFunctionWithName:@"vector_add"];
    if (!fn) return 4;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) return 5;
    id<MTLBuffer> aBuf = [device newBufferWithBytes:a length:sizeof(float)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuf = [device newBufferWithBytes:b length:sizeof(float)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> oBuf = [device newBufferWithLength:sizeof(float)*count options:MTLResourceStorageModeShared];
    if (!aBuf || !bBuf || !oBuf) return 6;
    uint32_t uCount = (uint32_t)count;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) return 7;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return 8;
    [enc setComputePipelineState:pso];
    [enc setBuffer:aBuf offset:0 atIndex:0];
    [enc setBuffer:bBuf offset:0 atIndex:1];
    [enc setBuffer:oBuf offset:0 atIndex:2];
    [enc setBytes:&uCount length:sizeof(uint32_t) atIndex:3];
    MTLSize tgSize = MTLSizeMake(pso.threadExecutionWidth, 1, 1);
    MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    memcpy(out, [oBuf contents], sizeof(float)*count);
    return 0;
}

extern "C" int cpp_gpu_matmul_add_shared(const float* A, size_t rowsA, size_t colsA, const float* B, size_t rowsB, size_t colsB, const float* D, float* C) {
    std::call_once(gInitFlag, initMetalObjects);
    id<MTLDevice> device = gSharedDevice;
    id<MTLCommandQueue> queue = gSharedQueue;
    if (!device || !queue) return 2;
    extern int32_t swift_gpu_matmul_add(const float* A, int32_t rowsA, int32_t colsA, const float* B, int32_t rowsB, int32_t colsB, const float* D, float* C);
    return swift_gpu_matmul_add(A, (int32_t)rowsA, (int32_t)colsA, B, (int32_t)rowsB, (int32_t)colsB, D, C);
}
