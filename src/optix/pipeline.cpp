#include "pipeline.h"
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <iostream>
#include <fstream>
#include <sstream>

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

Pipeline::Pipeline(Context& context) : context_(context) {}

Pipeline::~Pipeline() {
    if (pipeline_) {
        optixPipelineDestroy(pipeline_);
    }
    if (raygen_group_) optixProgramGroupDestroy(raygen_group_);
    if (miss_group_) optixProgramGroupDestroy(miss_group_);
    if (neural_hit_group_) optixProgramGroupDestroy(neural_hit_group_);
    if (triangle_hit_group_) optixProgramGroupDestroy(triangle_hit_group_);
    if (triangle_shadow_group_) optixProgramGroupDestroy(triangle_shadow_group_);
    if (neural_shadow_group_) optixProgramGroupDestroy(neural_shadow_group_);
    if (raygen_module_) optixModuleDestroy(raygen_module_);
    if (miss_module_) optixModuleDestroy(miss_module_);
    if (neural_module_) optixModuleDestroy(neural_module_);
    if (triangle_module_) optixModuleDestroy(triangle_module_);
}

std::string Pipeline::load_ptx_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

OptixModule Pipeline::create_module(const std::string& ptx_code) {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixModule module = nullptr;
    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        context_.get(),
        &module_compile_options,
        &pipeline_compile_options_,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &log_size,
        &module));

    if (log_size > 1) {
        std::cout << "Module creation log:\n" << log << std::endl;
    }

    return module;
}

bool Pipeline::build(const std::string& ptx_dir) {
    std::cout << "Building OptiX pipeline from PTX files in: " << ptx_dir << std::endl;

    // Initialize pipeline compile options (used by all modules)
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options_.numPayloadValues = 26;  // color (3) + pos_norm (3) + dir (3) + pos_unnorm (3) + neural_cache (5) + divergence_counters (7) + geometry_type (1) + instance_id (1)
    pipeline_compile_options_.numAttributeValues = 3; // For triangles: 3 attributes (geometric normal), for custom: t, primitive_id
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";

    // Load PTX files
    std::string raygen_ptx = load_ptx_file(ptx_dir + "/raygen.ptx");
    std::string miss_ptx = load_ptx_file(ptx_dir + "/miss.ptx");
    std::string neural_ptx = load_ptx_file(ptx_dir + "/neural.ptx");
    std::string triangle_ptx = load_ptx_file(ptx_dir + "/triangle.ptx");

    if (raygen_ptx.empty() || miss_ptx.empty() || neural_ptx.empty() || triangle_ptx.empty()) {
        std::cerr << "Failed to load one or more PTX files" << std::endl;
        return false;
    }

    // Create modules
    raygen_module_ = create_module(raygen_ptx);
    miss_module_ = create_module(miss_ptx);
    neural_module_ = create_module(neural_ptx);
    triangle_module_ = create_module(triangle_ptx);

    // Create program groups
    OptixProgramGroupOptions program_group_options = {};

    // Raygen program group
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = raygen_module_;
        desc.raygen.entryFunctionName = "__raygen__rg";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &raygen_group_));

        if (log_size > 1) {
            std::cout << "Raygen program group log:\n" << log << std::endl;
        }
    }

    // Miss program group
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = miss_module_;
        desc.miss.entryFunctionName = "__miss__ms";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &miss_group_));

        if (log_size > 1) {
            std::cout << "Miss program group log:\n" << log << std::endl;
        }
    }

    // Neural hit group (intersection + closest hit)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = neural_module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__neural";
        desc.hitgroup.moduleCH = neural_module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__neural";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &neural_hit_group_));

        if (log_size > 1) {
            std::cout << "Neural hit group log:\n" << log << std::endl;
        }
    }

    // Triangle hit group (closest hit only, built-in intersection)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = triangle_module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &triangle_hit_group_));

        if (log_size > 1) {
            std::cout << "Triangle hit group log:\n" << log << std::endl;
        }
    }

    // Triangle shadow hit group (any-hit only for shadow rays)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = triangle_module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__triangle_shadow";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &triangle_shadow_group_));

        if (log_size > 1) {
            std::cout << "Triangle shadow hit group log:\n" << log << std::endl;
        }
    }

    // Neural shadow hit group (intersection + any-hit for shadow rays)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = neural_module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__neural";
        desc.hitgroup.moduleAH = neural_module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__neural_shadow";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_.get(),
            &desc,
            1,
            &program_group_options,
            log,
            &log_size,
            &neural_shadow_group_));

        if (log_size > 1) {
            std::cout << "Neural shadow hit group log:\n" << log << std::endl;
        }
    }

    // Link pipeline
    OptixProgramGroup program_groups[] = {
        raygen_group_,
        miss_group_,
        neural_hit_group_,
        triangle_hit_group_,
        triangle_shadow_group_,
        neural_shadow_group_
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;  // Primary ray + shadow ray

    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context_.get(),
        &pipeline_compile_options_,  // pipeline compile options
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &log_size,
        &pipeline_));

    if (log_size > 1) {
        std::cout << "Pipeline creation log:\n" << log << std::endl;
    }

    // Set stack sizes
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline_));
    }

    uint32_t max_trace_depth = 2;  // Primary ray + shadow ray
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        2));  // Primary ray + shadow ray

    std::cout << "OptiX pipeline created successfully" << std::endl;
    return true;
}

} // namespace optix
