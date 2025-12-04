#include "tlas_builder.h"
#include "optix_stubs.h"
#include "utils/error.h"
#include <iostream>
#include <cstring>

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

namespace optix {

TLASBuilder::TLASBuilder(Context& context)
    : context_(context) {
}

TLASBuilder::~TLASBuilder() {
    if (d_instance_buffer_) cudaFree(d_instance_buffer_);
    if (d_tlas_buffer_) cudaFree(d_tlas_buffer_);
}

void TLASBuilder::add_instance(
    OptixTraversableHandle blas,
    uint32_t instance_id,
    uint32_t sbt_offset,
    const float transform[12],
    OptixInstanceFlags flags
) {
    OptixInstance instance = {};

    // Copy transform matrix (3x4 row-major)
    std::memcpy(instance.transform, transform, 12 * sizeof(float));

    // Set BLAS handle
    instance.traversableHandle = blas;

    // Set instance ID (used for geometry identification)
    instance.instanceId = instance_id;

    // Set SBT offset (routes to correct hit program)
    instance.sbtOffset = sbt_offset;

    // Set visibility mask (all rays can hit this instance)
    instance.visibilityMask = 255;

    // Set flags
    instance.flags = flags;

    instances_.push_back(instance);

    std::cout << "Added instance " << instance_id
              << " (BLAS=" << blas
              << ", SBT offset=" << sbt_offset << ")" << std::endl;
}

OptixTraversableHandle TLASBuilder::build() {
    if (instances_.empty()) {
        std::cerr << "Error: No instances added to TLAS builder" << std::endl;
        return 0;
    }

    std::cout << "Building TLAS with " << instances_.size() << " instances..." << std::endl;

    // Free old buffers if rebuilding
    if (d_instance_buffer_) {
        cudaFree(d_instance_buffer_);
        d_instance_buffer_ = nullptr;
    }
    if (d_tlas_buffer_) {
        cudaFree(d_tlas_buffer_);
        d_tlas_buffer_ = nullptr;
    }

    // Upload instances to device
    const size_t instance_buffer_size = instances_.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc(&d_instance_buffer_, instance_buffer_size));
    CUDA_CHECK(cudaMemcpy(
        d_instance_buffer_,
        instances_.data(),
        instance_buffer_size,
        cudaMemcpyHostToDevice
    ));

    // Setup build input for instances
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = (CUdeviceptr)d_instance_buffer_;
    build_input.instanceArray.numInstances = instances_.size();

    // Configure acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &build_input,
        1,  // num build inputs
        &buffer_sizes
    ));

    // Allocate temporary buffer
    void* d_temp_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, buffer_sizes.tempSizeInBytes));

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_tlas_buffer_, buffer_sizes.outputSizeInBytes));
    tlas_buffer_size_ = buffer_sizes.outputSizeInBytes;

    // Build the TLAS
    OptixTraversableHandle tlas_handle = 0;
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0,  // CUDA stream
        &accel_options,
        &build_input,
        1,  // num build inputs
        (CUdeviceptr)d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)d_tlas_buffer_,
        buffer_sizes.outputSizeInBytes,
        &tlas_handle,
        nullptr,  // emitted properties
        0         // num emitted properties
    ));

    // Wait for build to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_temp_buffer));

    std::cout << "TLAS built successfully (handle=" << tlas_handle << ")" << std::endl;

    return tlas_handle;
}

void TLASBuilder::clear() {
    instances_.clear();
}

} // namespace optix
