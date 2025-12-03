#include "context.h"
#include "utils/error.h"
#include <iostream>
#include <cuda.h>

namespace optix {

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                               \
        if (res != OPTIX_SUCCESS) {                                           \
            std::cerr << "OptiX call (" << #call << ") failed: "             \
                      << optixGetErrorName(res) << " - "                      \
                      << optixGetErrorString(res) << " ("                     \
                      << __FILE__ << ":" << __LINE__ << ")" << std::endl;     \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult res = call;                                                  \
        if (res != CUDA_SUCCESS) {                                            \
            const char* err_name;                                             \
            cuGetErrorName(res, &err_name);                                   \
            const char* err_str;                                              \
            cuGetErrorString(res, &err_str);                                  \
            std::cerr << "CUDA driver call (" << #call << ") failed: "       \
                      << err_name << " - " << err_str << " ("                 \
                      << __FILE__ << ":" << __LINE__ << ")" << std::endl;     \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

Context::Context() {}

Context::~Context() {
    if (context_) {
        optixDeviceContextDestroy(context_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void Context::context_log_cb(
    unsigned int level,
    const char* tag,
    const char* message,
    void* /*cbdata*/) {
    std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
}

bool Context::initialize() {
    if (initialized_) {
        return true;
    }

    // Initialize CUDA
    CU_CHECK(cuInit(0));

    // Get CUDA context
    CU_CHECK(cuCtxGetCurrent(&cuda_context_));
    if (!cuda_context_) {
        // Create a new CUDA context if none exists
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, 0));
        CU_CHECK(cuCtxCreate(&cuda_context_, 0, device));
    }

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX device context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &Context::context_log_cb;
    options.logCallbackLevel = 4; // Print all messages
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(
        cuda_context_,
        &options,
        &context_));

    std::cout << "OptiX context initialized successfully" << std::endl;

    initialized_ = true;
    return true;
}

} // namespace optix
