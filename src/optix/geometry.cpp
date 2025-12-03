#include "geometry.h"
#include "utils/error.h"
#include <iostream>

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

GeometryBuilder::GeometryBuilder(Context& context) : context_(context) {}

GeometryBuilder::~GeometryBuilder() {
    if (d_blas_output_buffer_) {
        cudaFree(d_blas_output_buffer_);
    }
    if (d_aabb_buffer_) {
        cudaFree(d_aabb_buffer_);
    }
}

OptixTraversableHandle GeometryBuilder::build_neural_asset_blas(
    const float3& min_bound,
    const float3& max_bound) {

    std::cout << "Building neural asset BLAS with bounds: ["
              << min_bound.x << ", " << min_bound.y << ", " << min_bound.z << "] to ["
              << max_bound.x << ", " << max_bound.y << ", " << max_bound.z << "]" << std::endl;

    // Allocate AABB buffer (6 floats: minx, miny, minz, maxx, maxy, maxz)
    float aabb[6] = {
        min_bound.x, min_bound.y, min_bound.z,
        max_bound.x, max_bound.y, max_bound.z
    };

    CUDA_CALL(cudaMalloc(&d_aabb_buffer_, sizeof(aabb)));
    CUDA_CALL(cudaMemcpy(d_aabb_buffer_, aabb, sizeof(aabb), cudaMemcpyHostToDevice));

    // Set up build input for custom primitives
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // Prepare flags array
    uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    CUdeviceptr aabb_ptr = (CUdeviceptr)d_aabb_buffer_;

    build_input.customPrimitiveArray.aabbBuffers = &aabb_ptr;
    build_input.customPrimitiveArray.numPrimitives = 1;
    build_input.customPrimitiveArray.flags = flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;

    // Configure acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &build_input,
        1,
        &buffer_sizes));

    // Allocate temporary buffers
    void* d_temp_buffer;
    CUDA_CALL(cudaMalloc(&d_temp_buffer, buffer_sizes.tempSizeInBytes));
    CUDA_CALL(cudaMalloc(&d_blas_output_buffer_, buffer_sizes.outputSizeInBytes));
    blas_buffer_size_ = buffer_sizes.outputSizeInBytes;

    // Build acceleration structure
    OptixTraversableHandle handle;
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        context_.get_stream(),
        &accel_options,
        &build_input,
        1,
        (CUdeviceptr)d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)d_blas_output_buffer_,
        buffer_sizes.outputSizeInBytes,
        &handle,
        nullptr,
        0));

    CUDA_CALL(cudaStreamSynchronize(context_.get_stream()));
    CUDA_CALL(cudaFree(d_temp_buffer));

    std::cout << "Neural asset BLAS built successfully" << std::endl;
    return handle;
}

} // namespace optix
