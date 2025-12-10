#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== HIPRT Scene Builder Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    hiprt::GeometryBuilder geom_builder(context);
    hiprt::SceneBuilder scene_builder(context);

    // Create a simple cube geometry
    std::vector<hiprt::Vertex> cube_vertices = {
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
    };
    std::vector<hiprt::Triangle> cube_triangles = {
        {0, 1, 2}, {0, 2, 3}, {5, 4, 7}, {5, 7, 6},
        {3, 2, 6}, {3, 6, 7}, {4, 5, 1}, {4, 1, 0},
        {1, 5, 6}, {1, 6, 2}, {4, 0, 3}, {4, 3, 7}
    };

    std::cout << "1. Building cube geometry..." << std::endl;
    auto cube_geom = geom_builder.build_triangle_geometry(cube_vertices, cube_triangles);
    if (!cube_geom.valid()) {
        std::cerr << "FAILED: Could not build cube geometry" << std::endl;
        return 1;
    }

    // Test 1: Single instance scene
    std::cout << "\n2. Testing single instance scene..." << std::endl;
    {
        scene_builder.clear();
        scene_builder.add_instance(cube_geom.get(), 0);
        auto scene = scene_builder.build();
        if (!scene.valid()) {
            std::cerr << "FAILED: Single instance scene is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Single instance scene built successfully\n" << std::endl;
    }

    // Test 2: Multiple instances with transforms
    std::cout << "3. Testing multiple instances with transforms..." << std::endl;
    {
        scene_builder.clear();

        // Instance at origin
        scene_builder.add_instance(cube_geom.get(), hiprt::identity_transform(), 0);

        // Instance translated to (3, 0, 0)
        scene_builder.add_instance(cube_geom.get(), hiprt::translation_transform(3.0f, 0.0f, 0.0f), 1);

        // Instance translated to (0, 3, 0)
        scene_builder.add_instance(cube_geom.get(), hiprt::translation_transform(0.0f, 3.0f, 0.0f), 2);

        // Instance translated to (0, 0, 3)
        scene_builder.add_instance(cube_geom.get(), hiprt::translation_transform(0.0f, 0.0f, 3.0f), 3);

        auto scene = scene_builder.build();
        if (!scene.valid()) {
            std::cerr << "FAILED: Multiple instance scene is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Multiple instance scene built successfully\n" << std::endl;
    }

    // Test 3: Mix of triangle and AABB geometries
    std::cout << "4. Testing mixed geometry types (triangles + AABBs)..." << std::endl;
    {
        // Create AABB geometry (for neural assets)
        std::vector<hiprt::AABB> aabbs = {
            {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f}
        };
        auto aabb_geom = geom_builder.build_aabb_geometry(aabbs, 1);  // geomType=1 for neural
        if (!aabb_geom.valid()) {
            std::cerr << "FAILED: Could not build AABB geometry" << std::endl;
            return 1;
        }

        scene_builder.clear();

        // Add triangle mesh instance
        scene_builder.add_instance(cube_geom.get(), hiprt::translation_transform(-2.0f, 0.0f, 0.0f), 0);

        // Add AABB (neural) instance
        scene_builder.add_instance(aabb_geom.get(), hiprt::translation_transform(2.0f, 0.0f, 0.0f), 1);

        auto scene = scene_builder.build();
        if (!scene.valid()) {
            std::cerr << "FAILED: Mixed geometry scene is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Mixed geometry scene built successfully\n" << std::endl;
    }

    // Test 4: Scaled instances
    std::cout << "5. Testing scaled instances..." << std::endl;
    {
        scene_builder.clear();

        // Normal scale
        scene_builder.add_instance(cube_geom.get(), hiprt::identity_transform(), 0);

        // 2x scale
        scene_builder.add_instance(cube_geom.get(), hiprt::scale_transform(2.0f, 2.0f, 2.0f), 1);

        // 0.5x scale
        scene_builder.add_instance(cube_geom.get(), hiprt::scale_transform(0.5f, 0.5f, 0.5f), 2);

        auto scene = scene_builder.build();
        if (!scene.valid()) {
            std::cerr << "FAILED: Scaled instance scene is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Scaled instance scene built successfully\n" << std::endl;
    }

    std::cout << "=== All scene tests PASSED ===" << std::endl;
    context.cleanup();
    return 0;
}
