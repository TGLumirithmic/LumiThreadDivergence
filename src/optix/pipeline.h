#pragma once

#include "context.h"
#include <optix.h>
#include <string>
#include <vector>

namespace optix {

// OptiX pipeline wrapper
class Pipeline {
public:
    Pipeline(Context& context);
    ~Pipeline();

    // Build the OptiX pipeline from PTX files
    bool build(const std::string& ptx_dir);

    // Get pipeline handle
    OptixPipeline get() const { return pipeline_; }

    // Get module handles
    OptixModule get_raygen_module() const { return raygen_module_; }
    OptixModule get_miss_module() const { return miss_module_; }
    OptixModule get_neural_module() const { return neural_module_; }

    // Get program group handles
    OptixProgramGroup get_raygen_group() const { return raygen_group_; }
    OptixProgramGroup get_miss_group() const { return miss_group_; }
    OptixProgramGroup get_neural_hit_group() const { return neural_hit_group_; }

private:
    Context& context_;
    OptixPipeline pipeline_ = nullptr;

    // Modules
    OptixModule raygen_module_ = nullptr;
    OptixModule miss_module_ = nullptr;
    OptixModule neural_module_ = nullptr;

    // Program groups
    OptixProgramGroup raygen_group_ = nullptr;
    OptixProgramGroup miss_group_ = nullptr;
    OptixProgramGroup neural_hit_group_ = nullptr;

    // Pipeline compile options (needed for pipeline creation)
    OptixPipelineCompileOptions pipeline_compile_options_ = {};

    // Helper functions
    OptixModule create_module(const std::string& ptx_code);
    std::string load_ptx_file(const std::string& filename);
};

} // namespace optix
