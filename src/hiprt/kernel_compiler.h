#pragma once

#include "hiprt_context.h"
#include <string>
#include <vector>

namespace hiprt {

/**
 * Compiled Kernel Handle
 *
 * Holds a compiled kernel function and associated resources.
 */
class CompiledKernel {
public:
    CompiledKernel() = default;
    CompiledKernel(hiprtApiFunction func, hiprtFuncTable table, hiprtApiModule module,
                   const HIPRTContext* ctx)
        : function_(func), func_table_(table), module_(module), context_(ctx) {}

    ~CompiledKernel() {
        cleanup();
    }

    // Move only
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    CompiledKernel(CompiledKernel&& other) noexcept
        : function_(other.function_)
        , func_table_(other.func_table_)
        , module_(other.module_)
        , context_(other.context_) {
        other.function_ = nullptr;
        other.func_table_ = nullptr;
        other.module_ = nullptr;
        other.context_ = nullptr;
    }

    CompiledKernel& operator=(CompiledKernel&& other) noexcept {
        if (this != &other) {
            cleanup();
            function_ = other.function_;
            func_table_ = other.func_table_;
            module_ = other.module_;
            context_ = other.context_;
            other.function_ = nullptr;
            other.func_table_ = nullptr;
            other.module_ = nullptr;
            other.context_ = nullptr;
        }
        return *this;
    }

    hiprtApiFunction get_function() const { return function_; }
    hiprtFuncTable get_func_table() const { return func_table_; }
    bool valid() const { return function_ != nullptr; }

    void cleanup() {
        if (func_table_ && context_) {
            hiprtDestroyFuncTable(context_->get_context(), func_table_);
        }
        // Module cleanup would go here if needed
        function_ = nullptr;
        func_table_ = nullptr;
        module_ = nullptr;
        context_ = nullptr;
    }

private:
    hiprtApiFunction function_ = nullptr;
    hiprtFuncTable func_table_ = nullptr;
    hiprtApiModule module_ = nullptr;
    const HIPRTContext* context_ = nullptr;
};

/**
 * Kernel Compiler
 *
 * Compiles CUDA/HIP kernel source at runtime using HIPRT's
 * hiprtBuildTraceKernels function.
 */
class KernelCompiler {
public:
    explicit KernelCompiler(const HIPRTContext& context);
    ~KernelCompiler() = default;

    // No copy
    KernelCompiler(const KernelCompiler&) = delete;
    KernelCompiler& operator=(const KernelCompiler&) = delete;

    /**
     * Compile render kernel with custom intersection functions
     *
     * @param kernel_source The kernel source code
     * @param kernel_name Name of the main kernel function
     * @param intersect_func_name Name of the custom intersection function (or nullptr)
     * @param filter_func_name Name of the custom filter function (or nullptr)
     * @param num_geom_types Number of geometry types (triangle=0, neural=1, etc.)
     * @param num_ray_types Number of ray types (primary=0, shadow=1, etc.)
     * @param compile_options Additional compiler options
     * @return Compiled kernel handle
     */
    CompiledKernel compile(
        const char* kernel_source,
        const char* kernel_name,
        const char* intersect_func_name = nullptr,
        const char* filter_func_name = nullptr,
        uint32_t num_geom_types = 2,
        uint32_t num_ray_types = 2,
        const std::vector<std::string>& compile_options = {}
    );

    /**
     * Add include path for kernel compilation
     */
    void add_include_path(const std::string& path);

    /**
     * Add header source for kernel compilation
     */
    void add_header(const std::string& name, const std::string& source);

private:
    const HIPRTContext& context_;
    std::vector<std::string> include_paths_;
    std::vector<std::pair<std::string, std::string>> headers_;
};

} // namespace hiprt
