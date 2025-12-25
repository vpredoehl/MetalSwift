#include <iostream>
#include <vector>
#include <cmath>
#include "CppGpuActivations.h"

extern "C" void cpp_run_matmul_add_demo();
extern "C" void cpp_run_gpu_demos();

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
