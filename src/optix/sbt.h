#pragma once

#include "context.h"
#include "pipeline.h"
#include "neural_types.h"
#include "vertex.h"  // Vertex struct definition
#include <optix.h>
#include <cuda_runtime.h>

namespace optix {

// Simple 3D vector structure (matches common.h)
struct float3_aligned {
    float x, y, z;
};

// Shader Binding Table record structures
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// Empty data for raygen and miss (no per-record data needed yet)
struct EmptyData {};

// Material data for hit groups
struct MaterialData {
    float3_aligned albedo;
    float roughness;

    // Neural network parameters (device pointer), nullptr for mesh instances
    NeuralNetworkParams* neural_params;

    // Triangle mesh data (device pointers), nullptr for neural assets
    Vertex* vertex_buffer;   // Device pointer to vertex array
    uint3* index_buffer;     // Device pointer to index array
};

// Instance metadata for SBT construction
enum class GeometryType {
    TRIANGLE_MESH,
    NEURAL_ASSET
};

struct InstanceMetadata {
    GeometryType type;
    uint32_t instance_id;
    uint32_t sbt_offset;

    // For neural assets: pointer to device-side neural network parameters
    NeuralNetworkParams* neural_params_device;

    // For meshes: material properties
    float3_aligned albedo;
    float roughness;

    // Triangle mesh buffers (device pointers), nullptr for neural assets
    void* vertex_buffer;  // Device pointer to Vertex array
    void* index_buffer;   // Device pointer to uint3 array
};

// Shader Binding Table manager
class ShaderBindingTable {
public:
    ShaderBindingTable(Context& context, Pipeline& pipeline);
    ~ShaderBindingTable();

    // Build the SBT with instance metadata
    bool build(const std::vector<InstanceMetadata>& instances);

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
