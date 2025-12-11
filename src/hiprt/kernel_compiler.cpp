#include "kernel_compiler.h"
#include <iostream>
#include <sstream>

namespace hiprt {

KernelCompiler::KernelCompiler(const HIPRTContext& context)
    : context_(context) {
    // Add default HIPRT include paths for runtime compilation
    // These are needed for the device headers
    add_include_path("/opt/hiprt/include");
    add_include_path("/opt/hiprt/include/hiprt");
}

void KernelCompiler::add_include_path(const std::string& path) {
    include_paths_.push_back(path);
}

void KernelCompiler::add_header(const std::string& name, const std::string& source) {
    headers_.push_back({name, source});
}

CompiledKernel KernelCompiler::compile(
    const char* kernel_source,
    const char* kernel_name,
    const char* intersect_func_name,
    const char* filter_func_name,
    uint32_t num_geom_types,
    uint32_t num_ray_types,
    const std::vector<std::string>& compile_options
) {
    std::cout << "Compiling kernel: " << kernel_name << std::endl;
    if (intersect_func_name) {
        std::cout << "  Custom intersection function: " << intersect_func_name << std::endl;
    }
    std::cout << "  Geometry types: " << num_geom_types << std::endl;
    std::cout << "  Ray types: " << num_ray_types << std::endl;

    // Prepare function name sets for custom intersection/filter functions
    // Layout: funcNameSets[rayType * numGeomTypes + geomType]
    std::vector<hiprtFuncNameSet> func_name_sets(num_ray_types * num_geom_types);

    // Initialize all to nullptr (use built-in intersection)
    for (auto& fn : func_name_sets) {
        fn.intersectFuncName = nullptr;
        fn.filterFuncName = nullptr;
    }

    // Set custom intersection function for neural/custom geometry (geomType = 1)
    // Apply to all ray types. Neural assets use geomType=1 (GEOM_TYPE_NEURAL)
    // while triangles use geomType=0 (GEOM_TYPE_TRIANGLE) with built-in intersection
    if (intersect_func_name) {
        for (uint32_t ray_type = 0; ray_type < num_ray_types; ++ray_type) {
            // Custom geometry type is 1 (GEOM_TYPE_NEURAL)
            uint32_t custom_geom_type = 1;
            uint32_t index = ray_type * num_geom_types + custom_geom_type;
            func_name_sets[index].intersectFuncName = intersect_func_name;
            func_name_sets[index].filterFuncName = filter_func_name;
            std::cout << "  Setting funcNameSet[" << index << "] = " << intersect_func_name
                      << " (rayType=" << ray_type << ", geomType=" << custom_geom_type << ")" << std::endl;
        }
    }

    // Prepare compiler options
    std::vector<const char*> options_ptrs;

    // Add include paths
    std::vector<std::string> include_opts;
    for (const auto& path : include_paths_) {
        include_opts.push_back("-I" + path);
    }
    for (const auto& opt : include_opts) {
        options_ptrs.push_back(opt.c_str());
    }

    // Add user-specified options
    for (const auto& opt : compile_options) {
        options_ptrs.push_back(opt.c_str());
    }

    // Note: HIPRT internally adds -std=c++17 and other standard options
    // Don't add them here to avoid duplication errors

    // Prepare headers
    std::vector<const char*> header_sources;
    std::vector<const char*> header_names;
    for (const auto& header : headers_) {
        header_names.push_back(header.first.c_str());
        header_sources.push_back(header.second.c_str());
    }

    // Function names to compile
    const char* func_names[] = { kernel_name };

    // Output variables
    hiprtApiFunction function = nullptr;
    hiprtApiModule module = nullptr;

    std::cout << "Calling hiprtBuildTraceKernels..." << std::endl;

    // Build the kernel
    hiprtError err = hiprtBuildTraceKernels(
        context_.get_context(),
        1,                              // numFunctions
        func_names,                     // funcNames
        kernel_source,                  // src
        "render_kernel",                // moduleName
        static_cast<uint32_t>(headers_.size()),  // numHeaders
        header_sources.empty() ? nullptr : header_sources.data(),  // headers
        header_names.empty() ? nullptr : header_names.data(),      // includeNames
        static_cast<uint32_t>(options_ptrs.size()),  // numOptions
        options_ptrs.empty() ? nullptr : options_ptrs.data(),      // options
        num_geom_types,                 // numGeomTypes
        num_ray_types,                  // numRayTypes
        func_name_sets.data(),          // funcNameSets
        &function,                      // functionsOut
        &module,                        // moduleOut
        true                            // cache
    );

    if (err != hiprtSuccess) {
        std::cerr << "ERROR: hiprtBuildTraceKernels failed with error " << err << std::endl;
        return CompiledKernel();
    }

    std::cout << "Kernel compiled successfully" << std::endl;

    // Create function table for custom intersection
    hiprtFuncTable func_table = nullptr;

    if (intersect_func_name || filter_func_name) {
        std::cout << "Creating function table (" << num_geom_types << " geomTypes x "
                  << num_ray_types << " rayTypes)..." << std::endl;

        HIPRT_CHECK(hiprtCreateFuncTable(
            context_.get_context(),
            num_geom_types,
            num_ray_types,
            func_table
        ));

        std::cout << "  Function table created: " << (void*)func_table << std::endl;

        // Set empty data for all geomType/rayType combinations
        // The actual data (payload) is passed through the traversal, not here
        // But we still need to initialize the function table entries
        // NOTE: Caller should override geomType=1 entries with actual AABB data!
        for (uint32_t ray_type = 0; ray_type < num_ray_types; ++ray_type) {
            for (uint32_t geom_type = 0; geom_type < num_geom_types; ++geom_type) {
                hiprtFuncDataSet data_set;
                data_set.intersectFuncData = nullptr;  // Data comes from traversal payload
                data_set.filterFuncData = nullptr;

                std::cout << "  hiprtSetFuncTable(geomType=" << geom_type << ", rayType=" << ray_type
                          << ", intersectFuncData=nullptr) [INIT]" << std::endl;

                HIPRT_CHECK(hiprtSetFuncTable(
                    context_.get_context(),
                    func_table,
                    geom_type,
                    ray_type,
                    data_set
                ));
            }
        }
        std::cout << "  Function table initialized (caller should set actual data for custom geomTypes)" << std::endl;
    }

    return CompiledKernel(function, func_table, module, &context_);
}

} // namespace hiprt
