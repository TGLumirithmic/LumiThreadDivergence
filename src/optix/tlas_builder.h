#pragma once

#include "context.h"
#include "sbt.h"  // For InstanceMetadata
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
        GeometryType geometry_type,            // Routes to correct hit program
        const float transform[12],      // 3x4 row-major transform matrix
        OptixInstanceFlags flags = OPTIX_INSTANCE_FLAG_NONE
    );

    // Add an instance with metadata (for SBT construction)
    void add_instance_with_metadata(
        OptixTraversableHandle blas,
        const InstanceMetadata& metadata,
        const float transform[12],
        OptixInstanceFlags flags = OPTIX_INSTANCE_FLAG_NONE
    );

    // Build the TLAS from all added instances
    OptixTraversableHandle build();

    // Clear all instances (for rebuilding)
    void clear();

    // Get number of instances
    size_t get_instance_count() const { return instances_.size(); }

    // Get instance metadata for SBT construction
    const std::vector<InstanceMetadata>& get_instance_metadata() const { return instance_metadata_; }

    // Get TLAS buffer (must keep alive for rendering)
    void* get_tlas_buffer() const { return d_tlas_buffer_; }

private:
    Context& context_;
    std::vector<OptixInstance> instances_;
    std::vector<InstanceMetadata> instance_metadata_;

    void* d_instance_buffer_ = nullptr;
    void* d_tlas_buffer_ = nullptr;
    size_t tlas_buffer_size_ = 0;
};

} // namespace optix
