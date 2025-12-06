#include "sbt.h"
#include "utils/error.h"
#include <iostream>
#include <cstring>
#include <vector>

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

ShaderBindingTable::ShaderBindingTable(Context& context, Pipeline& pipeline)
    : context_(context), pipeline_(pipeline) {}

ShaderBindingTable::~ShaderBindingTable() {
    if (d_raygen_record_) cudaFree(d_raygen_record_);
    if (d_miss_record_) cudaFree(d_miss_record_);
    if (d_hitgroup_record_) cudaFree(d_hitgroup_record_);
}

template <typename T>
void ShaderBindingTable::create_sbt_record(
    void** d_record,
    OptixProgramGroup program_group,
    const T& data) {

    SbtRecord<T> record;
    OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &record));
    record.data = data;

    CUDA_CALL(cudaMalloc(d_record, sizeof(SbtRecord<T>)));
    CUDA_CALL(cudaMemcpy(
        *d_record,
        &record,
        sizeof(SbtRecord<T>),
        cudaMemcpyHostToDevice));
}

bool ShaderBindingTable::build(const std::vector<InstanceMetadata>& instances) {
    std::cout << "Building Shader Binding Table" << std::endl;

    EmptyData empty_data = {};

    // Create raygen record
    create_sbt_record(&d_raygen_record_, pipeline_.get_raygen_group(), empty_data);

    // Create miss records (2: primary miss, shadow miss)
    typedef SbtRecord<EmptyData> MissRecord;
    MissRecord miss_records[2];

    // Primary miss (background color)
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_miss_group(), &miss_records[0]));
    miss_records[0].data = empty_data;

    // Shadow miss (no occlusion - visible)
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_miss_group(), &miss_records[1]));
    miss_records[1].data = empty_data;

    CUDA_CALL(cudaMalloc(&d_miss_record_, 2 * sizeof(MissRecord)));
    CUDA_CALL(cudaMemcpy(d_miss_record_, miss_records, 2 * sizeof(MissRecord), cudaMemcpyHostToDevice));

    // Create hitgroup records for each instance
    const size_t num_instances = instances.size();
    typedef SbtRecord<MaterialData> HitgroupRecord;
    std::vector<HitgroupRecord> hitgroup_records(num_instances * 2);

    for (size_t i = 0; i < num_instances; ++i) {
        const auto& inst = instances[i];
        MaterialData mat = {};

        if (inst.type == GeometryType::TRIANGLE_MESH) {
            // Triangle mesh instance
            mat.albedo = inst.albedo;
            mat.roughness = inst.roughness;
            mat.neural_params = nullptr;

            // Set vertex/index buffer pointers
            mat.vertex_buffer = (Vertex*)inst.vertex_buffer;
            mat.index_buffer = (uint3*)inst.index_buffer;

            // Use appropriate program group based on sbt_offset
            OptixProgramGroup pg = (inst.sbt_offset == 0) ?
                pipeline_.get_triangle_hit_group() :
                pipeline_.get_triangle_shadow_group();
            OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_triangle_hit_group(), &hitgroup_records[i*2]));
            OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_triangle_shadow_group(), &hitgroup_records[i*2+1]));

        } else {
            std::cout << "Creating neural asset entry in SBT" << std::endl;
            // Neural asset instance
            mat.albedo = inst.albedo;
            mat.roughness = inst.roughness;
            mat.neural_params = inst.neural_params_device;

            // nullptr for neural assets (no triangle mesh buffers)
            mat.vertex_buffer = nullptr;
            mat.index_buffer = nullptr;

            // Use appropriate program group based on sbt_offset
            // OptixProgramGroup pg = (inst.sbt_offset == 2) ?
            //     pipeline_.get_neural_hit_group() :
                // OptixProgramGroup pg = pipeline_.get_neural_hit_group();

                // OptixProgramGroup pg = pipeline_.get_neural_shadow_group();
            OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_neural_hit_group(), &hitgroup_records[i*2]));
            OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_neural_shadow_group(), &hitgroup_records[i*2+1]));

        }
        hitgroup_records[i].data = mat;
    }

    CUDA_CALL(cudaMalloc(&d_hitgroup_record_, num_instances * 2 * sizeof(HitgroupRecord)));
    CUDA_CALL(cudaMemcpy(d_hitgroup_record_, hitgroup_records.data(), num_instances * 2 * sizeof(HitgroupRecord), cudaMemcpyHostToDevice));

    // Fill in SBT structure
    sbt_.raygenRecord = (CUdeviceptr)d_raygen_record_;

    sbt_.missRecordBase = (CUdeviceptr)d_miss_record_;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 2;

    sbt_.hitgroupRecordBase = (CUdeviceptr)d_hitgroup_record_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_.hitgroupRecordCount = num_instances*2;

    std::cout << "Shader Binding Table built successfully" << std::endl;
    std::cout << "  - Miss records: " << sbt_.missRecordCount << std::endl;
    std::cout << "  - Hitgroup records: " << sbt_.hitgroupRecordCount << std::endl;
    return true;
}

} // namespace optix
