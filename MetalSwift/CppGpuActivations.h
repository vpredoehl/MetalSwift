#ifndef CPP_GPU_ACTIVATIONS_H
#define CPP_GPU_ACTIVATIONS_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// Runs vector sigmoid on GPU using Metal. Returns 0 on success, non-zero on failure.
// - input: input array of length `count`
// - output: output array of length `count`
// - count: number of elements
int cpp_gpu_vector_sigmoid(const float *input, float *output, size_t count);

// Runs vector tanh on GPU using Metal. Returns 0 on success, non-zero on failure.
// - input: input array of length `count`
// - output: output array of length `count`
// - count: number of elements
int cpp_gpu_vector_tanh(const float *input, float *output, size_t count);

#ifdef __cplusplus
}
#endif

#endif // CPP_GPU_ACTIVATIONS_H
