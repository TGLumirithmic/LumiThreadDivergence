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

    // Create raygen record
    EmptyData empty_data = {};
    create_sbt_record(&d_raygen_record_, pipeline_.get_raygen_group(), empty_data);

    // Create miss record
    create_sbt_record(&d_miss_record_, pipeline_.get_miss_group(), empty_data);

    // Create hitgroup record (for neural asset)
    create_sbt_record(&d_hitgroup_record_, pipeline_.get_neural_hit_group(), empty_data);

    // Fill in SBT structure
    sbt_.raygenRecord = (CUdeviceptr)d_raygen_record_;

    sbt_.missRecordBase = (CUdeviceptr)d_miss_record_;
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord<EmptyData>);
    sbt_.missRecordCount = 1;

    sbt_.hitgroupRecordBase = (CUdeviceptr)d_hitgroup_record_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<EmptyData>);
    sbt_.hitgroupRecordCount = 1;

    std::cout << "Shader Binding Table built successfully" << std::endl;
    return true;
}

} // namespace optix
