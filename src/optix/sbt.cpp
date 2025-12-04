#include "sbt.h"
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

bool ShaderBindingTable::build() {
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

    // Create hitgroup records (4: triangle primary, triangle shadow, neural primary, neural shadow)
    typedef SbtRecord<MaterialData> HitgroupRecord;
    HitgroupRecord hitgroup_records[4];

    // Record 0: Triangle primary hit
    MaterialData triangle_material;
    triangle_material.albedo = {0.8f, 0.8f, 0.8f};
    triangle_material.roughness = 0.5f;
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_triangle_hit_group(), &hitgroup_records[0]));
    hitgroup_records[0].data = triangle_material;

    // Record 1: Triangle shadow hit
    MaterialData triangle_shadow_material;
    triangle_shadow_material.albedo = {0.0f, 0.0f, 0.0f};
    triangle_shadow_material.roughness = 0.0f;
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_triangle_shadow_group(), &hitgroup_records[1]));
    hitgroup_records[1].data = triangle_shadow_material;

    // Record 2: Neural primary hit
    MaterialData neural_material;
    neural_material.albedo = {1.0f, 1.0f, 1.0f};
    neural_material.roughness = 0.0f;
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_neural_hit_group(), &hitgroup_records[2]));
    hitgroup_records[2].data = neural_material;

    // Record 3: Neural shadow hit
    MaterialData neural_shadow_material;
    neural_shadow_material.albedo = {0.0f, 0.0f, 0.0f};
    neural_shadow_material.roughness = 0.0f;
    OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_.get_neural_shadow_group(), &hitgroup_records[3]));
    hitgroup_records[3].data = neural_shadow_material;

    CUDA_CALL(cudaMalloc(&d_hitgroup_record_, 4 * sizeof(HitgroupRecord)));
    CUDA_CALL(cudaMemcpy(d_hitgroup_record_, hitgroup_records, 4 * sizeof(HitgroupRecord), cudaMemcpyHostToDevice));

    // Fill in SBT structure
    sbt_.raygenRecord = (CUdeviceptr)d_raygen_record_;

    sbt_.missRecordBase = (CUdeviceptr)d_miss_record_;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 2;

    sbt_.hitgroupRecordBase = (CUdeviceptr)d_hitgroup_record_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_.hitgroupRecordCount = 4;

    std::cout << "Shader Binding Table built successfully" << std::endl;
    std::cout << "  - Miss records: " << sbt_.missRecordCount << std::endl;
    std::cout << "  - Hitgroup records: " << sbt_.hitgroupRecordCount << std::endl;
    return true;
}

} // namespace optix
