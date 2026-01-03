// CppTensordotRun.cpp
// Prefer GPU implementation of cpp_gpu_tensordot_bwo_run; fall back to CPU for correctness.
// Computes Y[b,w,o] = sum_f X[b,w,f] * W[f,o]

#include <cstdint>
#include <cstddef>

// GPU entry point (implemented elsewhere, e.g., via Metal). Should return 0 on success.
extern "C" int tensordot_bwo_run_gpu(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */);

// CPU fallback implementation
static int tensordot_bwo_cpu(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */) {
    if (!bwo || !fo || !out) return -1;
    if (B <= 0 || W <= 0 || F <= 0 || O <= 0) return -2;

    const int bwoStrideBF = W * F; // elements per batch
    const int bwoStrideWF = F;     // elements per width row
    const int outStrideBO = W * O; // elements per batch in output
    const int outStrideWO = O;     // elements per width row in output

    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            for (int o = 0; o < O; ++o) {
                float acc = 0.0f;
                for (int f = 0; f < F; ++f) {
                    const int xIdx = (b * bwoStrideBF) + (w * bwoStrideWF) + f; // (b,w,f)
                    const int wIdx = f * O + o;                                  // (f,o)
                    acc += bwo[xIdx] * fo[wIdx];
                }
                const int yIdx = (b * outStrideBO) + (w * outStrideWO) + o;      // (b,w,o)
                out[yIdx] = acc;
            }
        }
    }

    return 0;
}
extern "C" int cpp_gpu_tensordot_bwo_run(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */) {
    // Basic validation shared by both paths
    if (!bwo || !fo || !out) return -1;
    if (B <= 0 || W <= 0 || F <= 0 || O <= 0) return -2;

    // Try GPU first. If the GPU symbol is linked and returns success, we're done.
    // Note: If the GPU implementation is not linked, this call will still resolve at link time;
    // provide the symbol in your GPU module. Non-zero return indicates GPU failure.
    int gpuStatus = tensordot_bwo_run_gpu(bwo, B, W, F, fo, O, out);
    if (gpuStatus == 0) {
        return 0;
    }

    // Fall back to CPU for correctness
    return tensordot_bwo_cpu(bwo, B, W, F, fo, O, out);
}

