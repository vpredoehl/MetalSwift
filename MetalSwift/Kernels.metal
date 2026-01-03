#include <metal_stdlib>
using namespace metal;

struct VecParams {
    uint count;
};

kernel void vec_add(const device float* a   [[buffer(0)]],
                    const device float* b   [[buffer(1)]],
                    device float* out       [[buffer(2)]],
                    constant VecParams& p   [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    out[gid] = a[gid] + b[gid];
}


struct TDParams {
    int B;
    int W;
    int F;
    int O;
    int bwoStrideBF; // W * F
    int bwoStrideWF; // F
    int outStrideBO; // W * O
    int outStrideWO; // O
};

kernel void tensordot_bwo(const device float* bwo   [[ buffer(0) ]],
                          const device float* fo    [[ buffer(1) ]],
                          device float* out         [[ buffer(2) ]],
                          constant TDParams& p      [[ buffer(3) ]],
                          uint3 gid                 [[ thread_position_in_grid ]]) {
    int b = int(gid.x);
    int w = int(gid.y);
    int o = int(gid.z);

    // Guard against over-dispatch
    if (b >= p.B || w >= p.W || o >= p.O) {
        return;
    }

    float acc = 0.0f;
    // Sum over F: X[b,w,f] * W[f,o]
    for (int f = 0; f < p.F; ++f) {
        int xIdx = (b * p.bwoStrideBF) + (w * p.bwoStrideWF) + f; // (b,w,f)
        int wIdx = f * p.O + o;                                   // (f,o)
        acc += bwo[xIdx] * fo[wIdx];
    }

    int yIdx = (b * p.outStrideBO) + (w * p.outStrideWO) + o;     // (b,w,o)
    out[yIdx] = acc;
}
