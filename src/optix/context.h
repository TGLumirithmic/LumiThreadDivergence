#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <string>

namespace optix {

// OptiX context wrapper
class Context {
public:
    Context();
    ~Context();

    // Initialize OptiX context
    bool initialize();

    // Get the OptiX context handle
    OptixDeviceContext get() const { return context_; }

    // Get CUDA context
    CUcontext get_cuda_context() const { return cuda_context_; }

    // Get CUDA stream
    cudaStream_t get_stream() const { return stream_; }

    // Check if initialized
    bool is_initialized() const { return initialized_; }

private:
    OptixDeviceContext context_ = nullptr;
    CUcontext cuda_context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;

    // OptiX log callback
    static void context_log_cb(
        unsigned int level,
        const char* tag,
        const char* message,
        void* /*cbdata*/);
};

} // namespace optix
