#pragma once

#include "context.h"
#include <optix.h>
#include <cuda_runtime.h>

namespace optix {

// Vertex structure for triangle meshes
struct Vertex {
    float3 position;
    float3 normal;
};

// Triangle geometry builder and manager
class TriangleGeometry {
public:
    TriangleGeometry(Context& context);
    ~TriangleGeometry();

    // Build BLAS for floor geometry (large horizontal quad)
    OptixTraversableHandle build_floor_blas();

    // Build BLAS for walls geometry (vertical quads)
    OptixTraversableHandle build_walls_blas();

    // Get BLAS buffers (must keep alive for rendering)
    void* get_floor_blas_buffer() const { return d_floor_blas_buffer_; }
    void* get_walls_blas_buffer() const { return d_walls_blas_buffer_; }

private:
    Context& context_;

    // Floor geometry buffers
    void* d_floor_vertices_ = nullptr;
    void* d_floor_indices_ = nullptr;
    void* d_floor_blas_buffer_ = nullptr;
    size_t floor_blas_size_ = 0;

    // Walls geometry buffers
    void* d_wall_vertices_ = nullptr;
    void* d_wall_indices_ = nullptr;
    void* d_walls_blas_buffer_ = nullptr;
    size_t walls_blas_size_ = 0;

    // Helper: Generic triangle BLAS builder
    OptixTraversableHandle build_triangle_blas(
        const Vertex* vertices,
        size_t num_vertices,
        const uint3* indices,
        size_t num_triangles,
        void** d_vertex_buffer_out,
        void** d_index_buffer_out,
        void** d_blas_buffer_out,
        size_t* blas_size_out
    );
};

} // namespace optix
