#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA synchronization and error checking
#define CUDA_SYNC_CHECK() \
    do { \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)
