#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include "kernel_source.h"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    std::cout << "=== HIPRT Neural Kernel Compilation Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    // Compile the full neural kernel
    std::cout << "1. Compiling neural render kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);

    // Add CUDA include path for cuda_runtime.h
    compiler.add_include_path("/usr/local/cuda/include");

    auto compiled = compiler.compile(
        getKernelSource(),
        getRenderKernelName(),
        getIntersectFuncName(),  // Custom intersection for neural assets
        nullptr,                  // No filter function
        2,                        // 2 geometry types (0=triangles, 1=neural)
        2                         // 2 ray types (0=primary, 1=shadow)
    );

    if (!compiled.valid()) {
        std::cerr << "FAILED: Could not compile neural kernel" << std::endl;
        return 1;
    }
    std::cout << "   Neural kernel compiled successfully!" << std::endl;

    // Build test scene with AABB geometry (simulating neural asset)
    std::cout << "\n2. Building test scene..." << std::endl;
    hiprt::GeometryBuilder geom_builder(context);

    std::vector<hiprt::AABB> aabbs = {
        {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f}  // Unit cube centered at origin
    };

    // geomType=1 for neural primitives (HIPRT transforms: stored = 1<<1 = 2, lookup = 2>>1 = 1)
    auto geom = geom_builder.build_aabb_geometry(aabbs, 1);
    if (!geom.valid()) {
        std::cerr << "FAILED: Could not build AABB geometry" << std::endl;
        return 1;
    }
    std::cout << "   AABB geometry built" << std::endl;

    // Build scene
    hiprt::SceneBuilder scene_builder(context);
    scene_builder.add_instance(geom.get(), 0);
    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "FAILED: Could not build scene" << std::endl;
        return 1;
    }
    std::cout << "   Scene built" << std::endl;

    std::cout << "\n=== Neural Kernel Compilation Test PASSED ===" << std::endl;
    context.cleanup();
    return 0;
}
