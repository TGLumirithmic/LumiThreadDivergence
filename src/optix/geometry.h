#pragma once

#include "context.h"
#include <optix.h>
#include <cuda_runtime.h>

namespace optix {

// Geometry builder for neural assets (AABB-based custom primitives)
class GeometryBuilder {
public:
    GeometryBuilder(Context& context);
    ~GeometryBuilder();

    // Build BLAS for neural asset with custom AABB
    OptixTraversableHandle build_neural_asset_blas(
        const float3& min_bound,
        const float3& max_bound);

    // Get the BLAS buffer (must be kept alive)
    void* get_blas_buffer() const { return d_blas_output_buffer_; }
    size_t get_blas_buffer_size() const { return blas_buffer_size_; }

private:
    Context& context_;
    void* d_blas_output_buffer_ = nullptr;
    void* d_aabb_buffer_ = nullptr;
    size_t blas_buffer_size_ = 0;
};

} // namespace optix
