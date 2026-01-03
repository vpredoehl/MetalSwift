// CppTensordotRunGPU.mm
// macOS Metal implementation of tensordot_bwo_run_gpu

#import <Metal/Metal.h>
#include <cstdint>
#include <cstddef>

extern "C" int tensordot_bwo_run_gpu(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */) {
    if (!bwo || !fo || !out) return -1;
    if (B <= 0 || W <= 0 || F <= 0 || O <= 0) return -2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -10;

    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) return -11;

    id<MTLFunction> function = [library newFunctionWithName:@"tensordot_bwo"];
    if (!function) return -12;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return -13;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return -14;

    // Create buffers
    const size_t bwoCount = (size_t)B * (size_t)W * (size_t)F;
    const size_t foCount  = (size_t)F * (size_t)O;
    const size_t outCount = (size_t)B * (size_t)W * (size_t)O;

    id<MTLBuffer> bwoBuf = [device newBufferWithBytes:bwo length:bwoCount * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> foBuf  = [device newBufferWithBytes:fo  length:foCount  * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:outCount * sizeof(float) options:MTLResourceStorageModeShared];
    if (!bwoBuf || !foBuf || !outBuf) return -15;

    struct TDParams { int B; int W; int F; int O; int bwoStrideBF; int bwoStrideWF; int outStrideBO; int outStrideWO; };
    TDParams p;
    p.B = B; p.W = W; p.F = F; p.O = O;
    p.bwoStrideBF = W * F;
    p.bwoStrideWF = F;
    p.outStrideBO = W * O;
    p.outStrideWO = O;

    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&p length:sizeof(TDParams) options:MTLResourceStorageModeShared];
    if (!paramsBuf) return -16;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) return -17;

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return -18;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bwoBuf offset:0 atIndex:0];
    [enc setBuffer:foBuf  offset:0 atIndex:1];
    [enc setBuffer:outBuf offset:0 atIndex:2];
    [enc setBuffer:paramsBuf offset:0 atIndex:3];

    // Thread grid: one thread per (b,w,o)
    MTLSize grid = MTLSizeMake((NSUInteger)B, (NSUInteger)W, (NSUInteger)O);

    // Choose a reasonable threadgroup size. We'll flatten z into y when needed, but Metal allows 3D grids.
    NSUInteger tgX = MIN(pipeline.maxTotalThreadsPerThreadgroup, 8);
    NSUInteger tgY = MIN(pipeline.maxTotalThreadsPerThreadgroup / tgX, 8);
    NSUInteger tgZ = MIN(pipeline.maxTotalThreadsPerThreadgroup / (tgX * tgY), 1);
    if (tgX == 0) tgX = 1;
    if (tgY == 0) tgY = 1;
    if (tgZ == 0) tgZ = 1;

    MTLSize threadsPerGroup = MTLSizeMake(tgX, tgY, tgZ);

    // Align grid to threadgroup sizes
    auto ceilDiv = ^(NSUInteger a, NSUInteger b) { return (a + b - 1) / b; };
    MTLSize groups = MTLSizeMake(ceilDiv(grid.width, threadsPerGroup.width),
                                 ceilDiv(grid.height, threadsPerGroup.height),
                                 ceilDiv(grid.depth, threadsPerGroup.depth));

    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy results back
    memcpy(out, outBuf.contents, outCount * sizeof(float));

    return 0;
}
