#include <iostream>
#include <vector>
#include <cmath>
#include "CppGpuActivations.h"

extern "C" void cpp_run_matmul_add_demo();
extern "C" void cpp_run_gpu_demos();
extern "C" void cpp_gpu_tensordot_bwo_demo(int B, int W, int F, int O);
extern "C" int cpp_gpu_tensordot_bwo_run(const float* bwo, int B, int W, int F, const float* fo, int O, float* out /* B*W*O */);

void printTensor3DSample(const char* name, const std::vector<float>& data, int B, int W, int F, int maxB = 2, int maxW = 2, int maxF = 4);
void printMatrixSample(const char* name, const std::vector<float>& data, int rows, int cols, int maxRows = 4, int maxCols = 4);

// This project previously used Swift as the entry point. To use this C++ main instead:
// 1) Remove or exclude any Swift files that declare an entry (e.g., a type annotated with `@main`
//    or a SwiftUI `App` conformer) from the build target.
// 2) Ensure your target compiles and links this file (main.cpp) as part of the target's Sources.
// 3) If you still need to call into Swift code, expose C-compatible functions via `@_cdecl` in Swift
//    and declare them here with `extern "C"`.
// 4) For app bundles (iOS/macOS app), make sure your Info.plist and app lifecycle align with a C/C++
//    entry point. For command-line tools, this file alone is sufficient.

// Example of calling a Swift function (if provided):
// extern "C" void SwiftEntryPoint();

int main(int argc, char* argv[]) {
    std::cout << "C++ main running. argc=" << argc << std::endl;

    std::cout << "-- Running C++ GPU MatMul+Add demo --" << std::endl;
    cpp_run_matmul_add_demo();

    std::cout << "-- Running additional GPU demos --" << std::endl;
    cpp_run_gpu_demos();

    std::cout << "-- Running TensorDot (B,W,F)·(F,O) demo (7,5,4)·(4,4) --" << std::endl;
    {
        const int B = 7, W = 5, F = 4, O = 4;
        std::vector<float> bwo(B * W * F);
        std::vector<float> fo(F * O);
        std::vector<float> bwo_fo_out(B * W * O, 0.0f);

        // Fill (B,W,F) tensor with a simple pattern: value = b*100 + w*10 + f
        for (int b = 0; b < B; ++b) {
            for (int w = 0; w < W; ++w) {
                for (int f = 0; f < F; ++f) {
                    int idx = (b * W * F) + (w * F) + f;
                    bwo[idx] = static_cast<float>(b * 100 + w * 10 + f);
                }
            }
        }
        // Fill (F,O) matrix with value = f*10 + o
        for (int f = 0; f < F; ++f) {
            for (int o = 0; o < O; ++o) {
                int idx = f * O + o;
                fo[idx] = static_cast<float>(f * 10 + o);
            }
        }

        // Assuming you have pointers and sizes:
        // const float* bwo; int B, W, F;
        // const float* fo;  int O;

        printTensor3DSample("bwo", bwo, B, W, F); // shows (b,w) slices and first few Fs
        printMatrixSample("fo", fo, F, O);        // shows first few rows/cols
        
        int tdStatus = cpp_gpu_tensordot_bwo_run(bwo.data(), B, W, F, fo.data(), O, bwo_fo_out.data());
        if (tdStatus != 0) {
            std::cerr << "cpp_gpu_tensordot_bwo_run failed (status=" << tdStatus << "). Falling back to cpp_gpu_tensordot_bwo_demo()" << std::endl;
            cpp_gpu_tensordot_bwo_demo(7, 5, 4, 4);
        } else {
            std::cout << "TensorDot output sample [b=0,w=0,*]: ";
            for (int o = 0; o < O; ++o) {
                int idx = (0 * W * O) + (0 * O) + o;
                std::cout << bwo_fo_out[idx] << (o + 1 < O ? ", " : "\n");
            }
        }
    }

    // --- Activation Function Demos (Metal GPU) ---
    std::vector<float> activationInput = { -2, -1, 0, 1, 2 };
    std::vector<float> sigmoidOut(activationInput.size(), 0.0f), tanhOut(activationInput.size(), 0.0f);
    int sigStatus = cpp_gpu_vector_sigmoid(activationInput.data(), sigmoidOut.data(), activationInput.size());
    int tanhStatus = cpp_gpu_vector_tanh(activationInput.data(), tanhOut.data(), activationInput.size());
    if (sigStatus != 0) std::cerr << "Metal sigmoid failed: " << sigStatus << std::endl;
    if (tanhStatus != 0) std::cerr << "Metal tanh failed: " << tanhStatus << std::endl;
    std::cout << "Sigmoid(";
    for (float v : activationInput) std::cout << v << (v!=activationInput.back() ? ", " : ") = ");
    for (float v : sigmoidOut) std::cout << v << (v!=sigmoidOut.back() ? ", " : "\n");
    std::cout << "Tanh(";
    for (float v : activationInput) std::cout << v << (v!=activationInput.back() ? ", " : ")     = ");
    for (float v : tanhOut) std::cout << v << (v!=tanhOut.back() ? ", " : "\n");

    return 0;
}

