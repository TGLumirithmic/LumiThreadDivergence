#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#if TCNN_HALF_PRECISION
#include <cuda_fp16.h>
#endif

namespace debug_utils {

// Print statistics about a device buffer
void print_buffer_stats(const float* d_buffer, size_t size, const std::string& name,
                       int max_print = 10, cudaStream_t stream = 0);

// Check for NaN/Inf values in a device buffer
bool check_for_nan_inf(const float* d_buffer, size_t size, const std::string& name,
                      cudaStream_t stream = 0);

// Print first N values from a device buffer
void print_buffer_values(const float* d_buffer, size_t size, const std::string& name,
                        int max_print = 10, cudaStream_t stream = 0);

// Check encoding weights for NaN/Inf
template<typename T>
bool check_encoding_params(void* encoding_ptr, const std::string& name);

// CUDA kernels for debugging
__global__ void check_nan_inf_kernel(const float* input, int* has_nan, int* has_inf, size_t size);
__global__ void print_values_kernel(const float* input, size_t size, int max_print);

// Print device values directly from kernel (synchronous)
void print_device_values(const float* d_buffer, size_t size, const std::string& name, int max_print = 10);

// Print half precision device values
#if TCNN_HALF_PRECISION
void print_device_values_half(const __half* d_buffer, size_t size, const std::string& name, int max_print = 10);
#endif

} // namespace debug_utils
