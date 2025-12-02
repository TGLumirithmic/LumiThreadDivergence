#include "debug_utils.h"
#include "cuda_utils.h"
#include "error.h"
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_memory.h>

namespace debug_utils {

// Kernel to check for NaN and Inf values
__global__ void check_nan_inf_kernel(const float* input, int* has_nan, int* has_inf, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = input[idx];
    if (isnan(val)) {
        atomicAdd(has_nan, 1);
    }
    if (isinf(val)) {
        atomicAdd(has_inf, 1);
    }
}

// Kernel to print values (for debugging)
__global__ void print_values_kernel(const float* input, size_t size, int max_print) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_print || idx >= size) return;

    if (idx == 0) {
        printf("[DEVICE] First %d values:\n", max_print);
    }
    __syncthreads();

    printf("  [%d] = %.6f\n", idx, input[idx]);
}

// Kernel to print device values with more detail
__global__ void print_device_values_detailed_kernel(const float* input, size_t size, int max_print) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n[DEVICE-SIDE] Inspecting buffer at %p, size=%zu\n", input, size);

        int print_count = (max_print < size) ? max_print : size;

        // Print values with their addresses
        for (int i = 0; i < print_count; i++) {
            float val = input[i];
            printf("  [%d] @ %p = %.8f", i, &input[i], val);

            // Check for special values
            if (isnan(val)) printf(" (NaN)");
            if (isinf(val)) printf(" (Inf)");
            if (val == 0.0f) printf(" (Zero)");

            printf("\n");
        }
        printf("\n");
    }
}

// Host function to print buffer statistics
void print_buffer_stats(const float* d_buffer, size_t size, const std::string& name,
                       int max_print, cudaStream_t stream) {
    std::cout << "\n[DEBUG] " << name << " (size=" << size << ")" << std::endl;

    // Copy to host
    std::vector<float> h_buffer(size);
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_buffer, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;

    for (size_t i = 0; i < size; ++i) {
        float val = h_buffer[i];

        if (std::isnan(val)) {
            nan_count++;
        } else if (std::isinf(val)) {
            inf_count++;
        } else {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            if (val == 0.0f) zero_count++;
        }
    }

    std::cout << "  NaN count: " << nan_count << std::endl;
    std::cout << "  Inf count: " << inf_count << std::endl;
    std::cout << "  Zero count: " << zero_count << std::endl;

    if (nan_count == 0 && inf_count == 0 && size > 0) {
        float mean = sum / (size - nan_count - inf_count);
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
    }

    // Print first few values
    int print_count = std::min(max_print, (int)size);
    std::cout << "  First " << print_count << " values: ";
    for (int i = 0; i < print_count; ++i) {
        std::cout << h_buffer[i];
        if (i < print_count - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

// Check for NaN/Inf values
bool check_for_nan_inf(const float* d_buffer, size_t size, const std::string& name,
                      cudaStream_t stream) {
    int* d_has_nan;
    int* d_has_inf;
    CUDA_CHECK(cudaMalloc(&d_has_nan, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_has_inf, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_nan, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_inf, 0, sizeof(int)));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    check_nan_inf_kernel<<<blocks, threads, 0, stream>>>(d_buffer, d_has_nan, d_has_inf, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_has_nan, h_has_inf;
    CUDA_CHECK(cudaMemcpy(&h_has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_has_inf, d_has_inf, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_has_nan));
    CUDA_CHECK(cudaFree(d_has_inf));

    if (h_has_nan > 0 || h_has_inf > 0) {
        std::cerr << "[DEBUG] " << name << " contains ";
        if (h_has_nan > 0) std::cerr << h_has_nan << " NaN values ";
        if (h_has_inf > 0) std::cerr << h_has_inf << " Inf values";
        std::cerr << std::endl;
        return false;
    }

    std::cout << "[DEBUG] " << name << " - OK (no NaN/Inf)" << std::endl;
    return true;
}

// Print first N values
void print_buffer_values(const float* d_buffer, size_t size, const std::string& name,
                        int max_print, cudaStream_t stream) {
    std::cout << "\n[DEBUG] " << name << " values:" << std::endl;

    std::vector<float> h_buffer(std::min(max_print, (int)size));
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_buffer, h_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_buffer.size(); ++i) {
        std::cout << "  [" << i << "] = " << h_buffer[i] << std::endl;
    }
}

// Template specialization for checking encoding parameters
template<typename T>
bool check_encoding_params(void* encoding_ptr, const std::string& name) {
    auto* encoding = reinterpret_cast<tcnn::Encoding<T>*>(encoding_ptr);

    size_t n_params = encoding->n_params();
    std::cout << "\n[DEBUG] Checking " << name << " parameters (count=" << n_params << ")" << std::endl;

    // Get parameters pointer - returns T*
    T* params_ptr = encoding->params();

    // Copy to host and check
    std::vector<T> h_params(n_params);
    CUDA_CHECK(cudaMemcpy(h_params.data(), params_ptr, n_params * sizeof(T), cudaMemcpyDeviceToHost));

    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;

    for (size_t i = 0; i < n_params; ++i) {
        float val = (float)h_params[i];

        if (std::isnan(val)) {
            nan_count++;
        } else if (std::isinf(val)) {
            inf_count++;
        } else {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            if (val == 0.0f) zero_count++;
        }
    }

    std::cout << "  NaN count: " << nan_count << std::endl;
    std::cout << "  Inf count: " << inf_count << std::endl;
    std::cout << "  Zero count: " << zero_count << std::endl;

    if (nan_count == 0 && inf_count == 0 && n_params > 0) {
        float mean = sum / (n_params - nan_count - inf_count);
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
    }

    // Print first 20 values
    std::cout << "  First 20 values: ";
    for (int i = 0; i < std::min(20, (int)n_params); ++i) {
        std::cout << (float)h_params[i];
        if (i < 19 && i < (int)n_params - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    return (nan_count == 0 && inf_count == 0);
}

// Host function to print device values using kernel
void print_device_values(const float* d_buffer, size_t size, const std::string& name, int max_print) {
    std::cout << "\n[DEBUG] " << name << " - Device-side inspection:" << std::endl;

    // Launch kernel with single thread to print values
    print_device_values_detailed_kernel<<<1, 1>>>(d_buffer, size, max_print);

    // Synchronize to ensure print completes
    CUDA_CHECK(cudaDeviceSynchronize());
}

#if TCNN_HALF_PRECISION
// Kernel to print half precision values
__global__ void print_half_values_detailed_kernel(const __half* input, size_t size, int max_print) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n[DEVICE-SIDE HALF] Inspecting buffer at %p, size=%zu\n", input, size);

        int print_count = (max_print < size) ? max_print : size;

        // Print values with their addresses, converting to float for display
        for (int i = 0; i < print_count; i++) {
            __half val_half = input[i];
            float val = __half2float(val_half);
            printf("  [%d] @ %p = %.8f", i, &input[i], val);

            // Check for special values
            if (isnan(val)) printf(" (NaN)");
            if (isinf(val)) printf(" (Inf)");
            if (val == 0.0f) printf(" (Zero)");

            printf("\n");
        }
        printf("\n");
    }
}

// Host function to print half precision device values
void print_device_values_half(const __half* d_buffer, size_t size, const std::string& name, int max_print) {
    std::cout << "\n[DEBUG] " << name << " - Device-side inspection (HALF):" << std::endl;

    // Launch kernel with single thread to print values
    print_half_values_detailed_kernel<<<1, 1>>>(d_buffer, size, max_print);

    // Synchronize to ensure print completes
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif

// Explicit template instantiation
template bool check_encoding_params<float>(void* encoding_ptr, const std::string& name);

#if TCNN_HALF_PRECISION
#include <cuda_fp16.h>
template bool check_encoding_params<__half>(void* encoding_ptr, const std::string& name);
#endif

} // namespace debug_utils
