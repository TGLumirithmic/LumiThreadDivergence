#include "hiprt_context.h"
#include "geometry_builder.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== HIPRT Geometry Builder Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    hiprt::GeometryBuilder builder(context);

    // Test 1: Build a simple triangle (single triangle)
    std::cout << "1. Testing single triangle geometry..." << std::endl;
    {
        std::vector<hiprt::Vertex> vertices = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.5f, 1.0f, 0.0f}
        };
        std::vector<hiprt::Triangle> triangles = {
            {0, 1, 2}
        };

        auto geom = builder.build_triangle_geometry(vertices, triangles, hiprt::BuildQuality::FAST);
        if (!geom.valid()) {
            std::cerr << "FAILED: Single triangle geometry is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Single triangle geometry built successfully\n" << std::endl;
    }

    // Test 2: Build a cube (12 triangles)
    std::cout << "2. Testing cube geometry (12 triangles)..." << std::endl;
    {
        std::vector<hiprt::Vertex> vertices = {
            // Front face
            {-0.5f, -0.5f,  0.5f}, // 0
            { 0.5f, -0.5f,  0.5f}, // 1
            { 0.5f,  0.5f,  0.5f}, // 2
            {-0.5f,  0.5f,  0.5f}, // 3
            // Back face
            {-0.5f, -0.5f, -0.5f}, // 4
            { 0.5f, -0.5f, -0.5f}, // 5
            { 0.5f,  0.5f, -0.5f}, // 6
            {-0.5f,  0.5f, -0.5f}, // 7
        };
        std::vector<hiprt::Triangle> triangles = {
            // Front
            {0, 1, 2}, {0, 2, 3},
            // Back
            {5, 4, 7}, {5, 7, 6},
            // Top
            {3, 2, 6}, {3, 6, 7},
            // Bottom
            {4, 5, 1}, {4, 1, 0},
            // Right
            {1, 5, 6}, {1, 6, 2},
            // Left
            {4, 0, 3}, {4, 3, 7}
        };

        auto geom = builder.build_triangle_geometry(vertices, triangles, hiprt::BuildQuality::FAST);
        if (!geom.valid()) {
            std::cerr << "FAILED: Cube geometry is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: Cube geometry built successfully\n" << std::endl;
    }

    // Test 3: Build AABB geometry (for neural assets)
    std::cout << "3. Testing AABB geometry (neural asset bounds)..." << std::endl;
    {
        std::vector<hiprt::AABB> aabbs = {
            {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f},   // Unit cube at origin
            {2.0f, 0.0f, 0.0f, 4.0f, 2.0f, 2.0f},      // Offset box
            {-3.0f, -1.0f, -1.0f, -2.0f, 1.0f, 1.0f}   // Another offset box
        };

        // geomType=1 for neural assets (custom intersection)
        auto geom = builder.build_aabb_geometry(aabbs, 1, hiprt::BuildQuality::FAST);
        if (!geom.valid()) {
            std::cerr << "FAILED: AABB geometry is invalid" << std::endl;
            return 1;
        }
        std::cout << "   PASSED: AABB geometry built successfully\n" << std::endl;
    }

    // Test 4: Test different build qualities
    std::cout << "4. Testing different build qualities..." << std::endl;
    {
        std::vector<hiprt::Vertex> vertices = {
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}
        };
        std::vector<hiprt::Triangle> triangles = {{0, 1, 2}};

        // Test BALANCED
        auto geom_balanced = builder.build_triangle_geometry(vertices, triangles, hiprt::BuildQuality::BALANCED);
        if (!geom_balanced.valid()) {
            std::cerr << "FAILED: BALANCED quality build failed" << std::endl;
            return 1;
        }
        std::cout << "   BALANCED quality: OK" << std::endl;

        // Test HIGH_QUALITY
        auto geom_hq = builder.build_triangle_geometry(vertices, triangles, hiprt::BuildQuality::HIGH_QUALITY);
        if (!geom_hq.valid()) {
            std::cerr << "FAILED: HIGH_QUALITY build failed" << std::endl;
            return 1;
        }
        std::cout << "   HIGH_QUALITY: OK" << std::endl;

        std::cout << "   PASSED: All build qualities work\n" << std::endl;
    }

    std::cout << "=== All geometry tests PASSED ===" << std::endl;
    context.cleanup();
    return 0;
}
