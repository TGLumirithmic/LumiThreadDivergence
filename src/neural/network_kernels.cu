#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ========== Float precision kernels (for sm_52 compatibility) ==========

__global__ void pad_direction_kernel_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t input_dims,
    uint32_t padded_dims
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint32_t input_offset = idx * input_dims;
    uint32_t output_offset = idx * padded_dims;

    // Copy actual input dimensions
    for (uint32_t i = 0; i < input_dims; ++i) {
        output[output_offset + i] = input[input_offset + i];
    }

    // Pad remaining dimensions with ones (matching training)
    for (uint32_t i = input_dims; i < padded_dims; ++i) {
        output[output_offset + i] = 1.0f;
    }
}

__global__ void concatenate_encodings_kernel_float(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t dims1,
    uint32_t dims2
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint32_t total_dims = dims1 + dims2;
    uint32_t input1_offset = idx * dims1;
    uint32_t input2_offset = idx * dims2;
    uint32_t output_offset = idx * total_dims;

    // Copy first input
    for (uint32_t i = 0; i < dims1; ++i) {
        output[output_offset + i] = input1[input1_offset + i];
    }

    // Copy second input
    for (uint32_t i = 0; i < dims2; ++i) {
        output[output_offset + dims1 + i] = input2[input2_offset + i];
    }
}

// ========== Half precision kernels (legacy - not used with sm_52) ==========

// CUDA kernel for concatenating position and direction encodings and converting to half precision
__global__ void concatenate_encodings_to_half_kernel(
    const float* __restrict__ pos_encoding,
    const float* __restrict__ dir_encoding,
    __half* __restrict__ output,
    uint32_t batch_size,
    uint32_t pos_dims,
    uint32_t dir_dims
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint32_t total_dims = pos_dims + dir_dims;
    uint32_t output_offset = idx * total_dims;
    uint32_t pos_offset = idx * pos_dims;
    uint32_t dir_offset = idx * dir_dims;

    // Copy position encoding and convert to half
    for (uint32_t i = 0; i < pos_dims; ++i) {
        output[output_offset + i] = __float2half(pos_encoding[pos_offset + i]);
    }

    // Copy direction encoding and convert to half
    for (uint32_t i = 0; i < dir_dims; ++i) {
        output[output_offset + pos_dims + i] = __float2half(dir_encoding[dir_offset + i]);
    }
}

// CUDA kernel for converting float buffer to half precision (for position-only case)
__global__ void convert_float_to_half_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    uint32_t size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = __float2half(input[idx]);
}

// CUDA kernel for converting and padding direction input from 3D to 16D (padded with zeros)
__global__ void convert_and_pad_direction_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    uint32_t batch_size,
    uint32_t input_dims,
    uint32_t padded_dims
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint32_t input_offset = idx * input_dims;
    uint32_t output_offset = idx * padded_dims;

    // Copy actual input dimensions and convert to half
    for (uint32_t i = 0; i < input_dims; ++i) {
        output[output_offset + i] = __float2half(input[input_offset + i]);
    }

    // Pad remaining dimensions with ones (matching training)
    for (uint32_t i = input_dims; i < padded_dims; ++i) {
        output[output_offset + i] = __float2half(1.0f);
    }
}
