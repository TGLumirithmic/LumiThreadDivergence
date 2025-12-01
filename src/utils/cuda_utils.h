#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "error.h"

namespace cuda_utils {

// Device memory allocator wrapper
template<typename T>
T* allocate_device(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// Free device memory
template<typename T>
void free_device(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Copy host to device
template<typename T>
void copy_to_device(T* d_dest, const T* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dest, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy device to host
template<typename T>
void copy_to_host(T* h_dest, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dest, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Print device properties
inline void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
}

} // namespace cuda_utils
