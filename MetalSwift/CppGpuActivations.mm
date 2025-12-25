#import "CppGpuActivations.h"
#import <Metal/Metal.h>
#include <vector>
#include <cstring>

static int runMetalActivation(const float *input, float *output, size_t count, const char *kernelName, const char *kernelSrc) {
    if (count == 0) return 0;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return 2;
    NSError *err = nil;
    NSString *srcStr = [NSString stringWithUTF8String:kernelSrc];
    id<MTLLibrary> lib = [device newLibraryWithSource:srcStr options:nil error:&err];
    if (!lib) return 3;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
    if (!fn) return 4;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) return 5;
    id<MTLBuffer> inBuf = [device newBufferWithBytes:input length:sizeof(float)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:sizeof(float)*count options:MTLResourceStorageModeShared];
    if (!inBuf || !outBuf) return 6;
    uint32_t cnt = (uint32_t)count;
    id<MTLBuffer> cntBuf = [device newBufferWithBytes:&cnt length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!cntBuf) return 7;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) return 8;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return 9;
    [enc setComputePipelineState:pso];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    [enc setBuffer:cntBuf offset:0 atIndex:2];
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger grid = count;
    MTLSize tg = MTLSizeMake(w, 1, 1);
    MTLSize gr = MTLSizeMake(grid, 1, 1);
    [enc dispatchThreads:gr threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) return 10;
    memcpy(output, [outBuf contents], sizeof(float)*count);
    return 0;
}

extern "C" int cpp_gpu_vector_sigmoid(const float *input, float *output, size_t count) {
    static const char *src = R"METAL(
        #include <metal_stdlib>
        using namespace metal;
        kernel void vec_sigmoid(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            if (id >= count) return;
            out[id] = 1.0f / (1.0f + exp(-in[id]));
        }
    )METAL";
    return runMetalActivation(input, output, count, "vec_sigmoid", src);
}

extern "C" int cpp_gpu_vector_tanh(const float *input, float *output, size_t count) {
    static const char *src = R"METAL(
        #include <metal_stdlib>
        using namespace metal;
        kernel void vec_tanh(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            if (id >= count) return;
            out[id] = tanh(in[id]);
        }
    )METAL";
    return runMetalActivation(input, output, count, "vec_tanh", src);
}
