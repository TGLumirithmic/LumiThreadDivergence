#pragma once

#include "context.h"
#include "pipeline.h"
#include <optix.h>
#include <cuda_runtime.h>

namespace optix {

// Shader Binding Table record structures
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// Empty data for raygen and miss (no per-record data needed yet)
struct EmptyData {};

// Shader Binding Table manager
class ShaderBindingTable {
public:
    ShaderBindingTable(Context& context, Pipeline& pipeline);
    ~ShaderBindingTable();

    // Build the SBT
    bool build();

    // Get the OptiX SBT structure
    const OptixShaderBindingTable& get() const { return sbt_; }

private:
    Context& context_;
    Pipeline& pipeline_;
    OptixShaderBindingTable sbt_ = {};

    // Device buffers for SBT records
    void* d_raygen_record_ = nullptr;
    void* d_miss_record_ = nullptr;
    void* d_hitgroup_record_ = nullptr;

    template <typename T>
    void create_sbt_record(void** d_record, OptixProgramGroup program_group, const T& data);
};

} // namespace optix
