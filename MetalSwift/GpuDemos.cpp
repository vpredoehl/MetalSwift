/*
 GpuDemos.cpp

 This file previously contained executable demo code (vector add and matmul+add)
 and an exported entry point `cpp_run_gpu_demos()`. To convert this target into
 a macOS C++ library, side-effectful demo code and entry points have been
 removed. The library should expose only reusable APIs declared elsewhere
 (e.g., in CppGpuBridge.h / CppGpuMatMulAdd.h) and be linked by clients.

 If you still need sample usage, move it to a separate sample app or test target.
*/
