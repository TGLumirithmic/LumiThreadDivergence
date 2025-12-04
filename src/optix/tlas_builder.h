#pragma once

#include "context.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <vector>

namespace optix {

// Top-Level Acceleration Structure (TLAS) builder
class TLASBuilder {
public:
    TLASBuilder(Context& context);
    ~TLASBuilder();

    // Add an instance to the TLAS
    void add_instance(
        OptixTraversableHandle blas,
        uint32_t instance_id,           // For geometry type identification
        uint32_t sbt_offset,            // Routes to correct hit program
        const float transform[12],      // 3x4 row-major transform matrix
        OptixInstanceFlags flags = OPTIX_INSTANCE_FLAG_NONE
    );

    // Build the TLAS from all added instances
    OptixTraversableHandle build();

    // Clear all instances (for rebuilding)
    void clear();

    // Get number of instances
    size_t get_instance_count() const { return instances_.size(); }

    // Get TLAS buffer (must keep alive for rendering)
    void* get_tlas_buffer() const { return d_tlas_buffer_; }

private:
    Context& context_;
    std::vector<OptixInstance> instances_;

    void* d_instance_buffer_ = nullptr;
    void* d_tlas_buffer_ = nullptr;
    size_t tlas_buffer_size_ = 0;
};

} // namespace optix
