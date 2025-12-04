#include "triangle_geometry.h"
#include "optix_stubs.h"
#include "utils/error.h"
#include <vector>
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

// Hardcoded geometry definitions
namespace geometry {
    // Floor: Large horizontal quad at Y = -1, centered at origin, 10x10 units
    const Vertex floor_vertices[] = {
        {{-5.0f, -1.0f, -5.0f}, {0.0f, 1.0f, 0.0f}},  // Bottom-left
        {{ 5.0f, -1.0f, -5.0f}, {0.0f, 1.0f, 0.0f}},  // Bottom-right
        {{ 5.0f, -1.0f,  5.0f}, {0.0f, 1.0f, 0.0f}},  // Top-right
        {{-5.0f, -1.0f,  5.0f}, {0.0f, 1.0f, 0.0f}},  // Top-left
    };
    const uint3 floor_indices[] = {
        {0, 1, 2},  // Triangle 1
        {0, 2, 3}   // Triangle 2
    };
    const size_t floor_num_vertices = 4;
    const size_t floor_num_triangles = 2;

    // Back wall: Vertical quad at Z = -5, extends from Y=-1 to Y=5
    const Vertex wall_vertices[] = {
        {{-5.0f, -1.0f, -5.0f}, {0.0f, 0.0f, 1.0f}},  // Bottom-left
        {{ 5.0f, -1.0f, -5.0f}, {0.0f, 0.0f, 1.0f}},  // Bottom-right
        {{ 5.0f,  5.0f, -5.0f}, {0.0f, 0.0f, 1.0f}},  // Top-right
        {{-5.0f,  5.0f, -5.0f}, {0.0f, 0.0f, 1.0f}},  // Top-left
    };
    const uint3 wall_indices[] = {
        {0, 1, 2},  // Triangle 1
        {0, 2, 3}   // Triangle 2
    };
    const size_t wall_num_vertices = 4;
    const size_t wall_num_triangles = 2;
}

TriangleGeometry::TriangleGeometry(Context& context)
    : context_(context) {
}

TriangleGeometry::~TriangleGeometry() {
    // Free floor buffers
    if (d_floor_vertices_) cudaFree(d_floor_vertices_);
    if (d_floor_indices_) cudaFree(d_floor_indices_);
    if (d_floor_blas_buffer_) cudaFree(d_floor_blas_buffer_);

    // Free walls buffers
    if (d_wall_vertices_) cudaFree(d_wall_vertices_);
    if (d_wall_indices_) cudaFree(d_wall_indices_);
    if (d_walls_blas_buffer_) cudaFree(d_walls_blas_buffer_);
}

OptixTraversableHandle TriangleGeometry::build_floor_blas() {
    return build_triangle_blas(
        geometry::floor_vertices,
        geometry::floor_num_vertices,
        geometry::floor_indices,
        geometry::floor_num_triangles,
        &d_floor_vertices_,
        &d_floor_indices_,
        &d_floor_blas_buffer_,
        &floor_blas_size_
    );
}

OptixTraversableHandle TriangleGeometry::build_walls_blas() {
    return build_triangle_blas(
        geometry::wall_vertices,
        geometry::wall_num_vertices,
        geometry::wall_indices,
        geometry::wall_num_triangles,
        &d_wall_vertices_,
        &d_wall_indices_,
        &d_walls_blas_buffer_,
        &walls_blas_size_
    );
}

OptixTraversableHandle TriangleGeometry::build_triangle_blas(
    const Vertex* vertices,
    size_t num_vertices,
    const uint3* indices,
    size_t num_triangles,
    void** d_vertex_buffer_out,
    void** d_index_buffer_out,
    void** d_blas_buffer_out,
    size_t* blas_size_out
) {
    // Upload vertices to device
    const size_t vertex_buffer_size = num_vertices * sizeof(Vertex);
    CUDA_CHECK(cudaMalloc(d_vertex_buffer_out, vertex_buffer_size));
    CUDA_CHECK(cudaMemcpy(*d_vertex_buffer_out, vertices, vertex_buffer_size, cudaMemcpyHostToDevice));

    // Upload indices to device
    const size_t index_buffer_size = num_triangles * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(d_index_buffer_out, index_buffer_size));
    CUDA_CHECK(cudaMemcpy(*d_index_buffer_out, indices, index_buffer_size, cudaMemcpyHostToDevice));

    // Setup build input for triangles
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Vertex buffer
    CUdeviceptr d_vertices = (CUdeviceptr)*d_vertex_buffer_out;
    build_input.triangleArray.vertexBuffers = &d_vertices;
    build_input.triangleArray.numVertices = num_vertices;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);

    // Index buffer
    build_input.triangleArray.indexBuffer = (CUdeviceptr)*d_index_buffer_out;
    build_input.triangleArray.numIndexTriplets = num_triangles;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);

    // Flags (no special flags needed)
    uint32_t triangle_flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
    build_input.triangleArray.flags = triangle_flags;
    build_input.triangleArray.numSbtRecords = 1;

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
        1,  // num build inputs
        &buffer_sizes
    ));

    // Allocate temporary buffers
    void* d_temp_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, buffer_sizes.tempSizeInBytes));

    // Allocate output buffer (initially uncompacted)
    void* d_output_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_buffer, buffer_sizes.outputSizeInBytes));

    // Build the BLAS
    OptixTraversableHandle traversable_handle = 0;
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0,  // CUDA stream
        &accel_options,
        &build_input,
        1,  // num build inputs
        (CUdeviceptr)d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)d_output_buffer,
        buffer_sizes.outputSizeInBytes,
        &traversable_handle,
        nullptr,  // emitted properties
        0         // num emitted properties
    ));

    // Wait for build to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_temp_buffer));

    // Store output buffer and size
    *d_blas_buffer_out = d_output_buffer;
    *blas_size_out = buffer_sizes.outputSizeInBytes;

    std::cout << "Built triangle BLAS: " << num_triangles << " triangles, "
              << num_vertices << " vertices" << std::endl;

    return traversable_handle;
}

} // namespace optix
