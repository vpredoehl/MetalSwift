// CppTensordotRunGPU.mm
// macOS Metal implementation of tensordot_bwo_run_gpu

#import <Metal/Metal.h>
#include <cstdint>
#include <cstddef>

extern "C" int tensordot_gpu(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */) {
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

    // Debug: show all input data and calculation steps for all outputs before GPU dispatch
    {
        printf("tensordot_bwo_run_gpu: Input tensors before GPU dispatch\n");
        // Print bwo as [B][W][F]
        printf("bwo [B=%d, W=%d, F=%d]:\n", B, W, F);
        for (int b = 0; b < B; ++b) {
            for (int w = 0; w < W; ++w) {
                printf("  bwo[b=%d,w=%d,:] = [", b, w);
                for (int f = 0; f < F; ++f) {
                    size_t idxBWO = (size_t)b * (size_t)W * (size_t)F + (size_t)w * (size_t)F + (size_t)f;
                    printf("%s%g", (f==0?"":" ,"), bwo[idxBWO]);
                }
                printf("]\n");
            }
        }
        // Print fo as [F][O]
        printf("fo [F=%d, O=%d]:\n", F, O);
        for (int f = 0; f < F; ++f) {
            printf("  fo[f=%d,:] = [", f);
            for (int o = 0; o < O; ++o) {
                size_t idxFO  = (size_t)f * (size_t)O + (size_t)o;
                printf("%s%g", (o==0?"":" ,"), fo[idxFO]);
            }
            printf("]\n");
        }

        // Full breakdown of C[b,w,o] = sum_f bwo[b,w,f] * fo[f,o]
        printf("tensordot_bwo_run_gpu breakdown (C[b,w,o] = sum_f bwo[b,w,f]*fo[f,o]):\n");
        for (int b = 0; b < B; ++b) {
            for (int w = 0; w < W; ++w) {
                for (int o = 0; o < O; ++o) {
                    printf("  C[b=%d,w=%d,o=%d] = ", b, w, o);
                    float sum = 0.0f;
                    for (int f = 0; f < F; ++f) {
                        size_t idxBWO = (size_t)b * (size_t)W * (size_t)F + (size_t)w * (size_t)F + (size_t)f;
                        size_t idxFO  = (size_t)f * (size_t)O + (size_t)o;
                        float a = bwo[idxBWO];
                        float bval = fo[idxFO];
                        sum += a * bval;
                        printf("%s(%g*%g)", (f==0?"":" + "), a, bval);
                    }
                    printf(" = %g\n", sum);
                }
            }
        }
    }

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

    // Debug: show all outputs after GPU tensordot
    {
        printf("tensordot_bwo_run_gpu: Output tensor after GPU dispatch\n");
        // Print full out as [B][W][O]
        printf("out [B=%d, W=%d, O=%d]:\n", B, W, O);
        for (int b = 0; b < B; ++b) {
            for (int w = 0; w < W; ++w) {
                printf("  out[b=%d,w=%d,:] = [", b, w);
                for (int o = 0; o < O; ++o) {
                    size_t idxOUT = (size_t)b * (size_t)W * (size_t)O + (size_t)w * (size_t)O + (size_t)o;
                    printf("%s%g", (o==0?"":" ,"), out[idxOUT]);
                }
                printf("]\n");
            }
        }
        // Sample line for [b=0,w=0,*]
        if (B > 0 && W > 0) {
            printf("TensorDot output sample [b=0,w=0,*]: ");
            for (int o = 0; o < O; ++o) {
                size_t idxOUT = (size_t)0 * (size_t)W * (size_t)O + (size_t)0 * (size_t)O + (size_t)o;
                printf("%s%g", (o==0?"":" , "), out[idxOUT]);
            }
            printf("\n");
        }
    }

    return 0;
}

