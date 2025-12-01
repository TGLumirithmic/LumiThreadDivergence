#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// CUDA kernel declarations for network inference

// Float precision versions (for sm_52 compatibility)
__global__ void pad_direction_kernel_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t input_dims,
    uint32_t padded_dims
);

__global__ void concatenate_encodings_kernel_float(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t dims1,
    uint32_t dims2
);

// Half precision versions (legacy - not used with sm_52)
__global__ void concatenate_encodings_to_half_kernel(
    const float* __restrict__ pos_encoding,
    const float* __restrict__ dir_encoding,
    __half* __restrict__ output,
    uint32_t batch_size,
    uint32_t pos_dims,
    uint32_t dir_dims
);

__global__ void convert_float_to_half_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    uint32_t size
);

__global__ void convert_and_pad_direction_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    uint32_t batch_size,
    uint32_t input_dims,
    uint32_t padded_dims
);
