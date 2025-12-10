// Test isolated wall intersection to debug geometry/transform issues
#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

// Simple test kernel that fires rays straight at a wall and reports hit positions
const char* WALL_TEST_KERNEL_SOURCE = R"(
#include <hiprt/hiprt_device.h>

struct HitResult {
    float t;
    float hit_x, hit_y, hit_z;
    float normal_x, normal_y, normal_z;
    int has_hit;
};

extern "C" __global__ void wallTestKernel(
    hiprtScene scene,
    float ray_origin_x, float ray_origin_y, float ray_origin_z,
    float ray_dir_x, float ray_dir_y, float ray_dir_z,
    HitResult* results,
    uint32_t num_rays_x,
    uint32_t num_rays_y
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= num_rays_x || y >= num_rays_y) return;

    const uint32_t idx = y * num_rays_x + x;

    // Create ray - offset origin based on x,y to create a grid of rays
    // Rays are spaced 0.5 units apart in a grid pattern
    float offset_x = (float(x) - float(num_rays_x) / 2.0f) * 0.5f;
    float offset_y = (float(y) - float(num_rays_y) / 2.0f) * 0.5f;

    hiprtRay ray;
    ray.origin = hiprtFloat3{ray_origin_x + offset_x, ray_origin_y + offset_y, ray_origin_z};
    ray.direction = hiprtFloat3{ray_dir_x, ray_dir_y, ray_dir_z};
    ray.minT = 0.001f;
    ray.maxT = 1000.0f;

    // Trace ray
    hiprtSceneTraversalClosest traversal(
        scene,
        ray,
        hiprtFullRayMask,
        hiprtTraversalHintDefault,
        nullptr,
        nullptr,
        0,
        0.0f
    );

    hiprtHit hit = traversal.getNextHit();

    results[idx].has_hit = hit.hasHit() ? 1 : 0;
    if (hit.hasHit()) {
        results[idx].t = hit.t;
        results[idx].hit_x = ray.origin.x + hit.t * ray.direction.x;
        results[idx].hit_y = ray.origin.y + hit.t * ray.direction.y;
        results[idx].hit_z = ray.origin.z + hit.t * ray.direction.z;
        results[idx].normal_x = hit.normal.x;
        results[idx].normal_y = hit.normal.y;
        results[idx].normal_z = hit.normal.z;
    } else {
        results[idx].t = -1.0f;
        results[idx].hit_x = 0.0f;
        results[idx].hit_y = 0.0f;
        results[idx].hit_z = 0.0f;
        results[idx].normal_x = 0.0f;
        results[idx].normal_y = 0.0f;
        results[idx].normal_z = 0.0f;
    }
}
)";

struct HitResult {
    float t;
    float hit_x, hit_y, hit_z;
    float normal_x, normal_y, normal_z;
    int has_hit;
};

int main() {
    std::cout << "=== Wall Intersection Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }

    // Create wall geometry manually (same as walls.obj)
    // Wall at z=-5, x from -5 to 5, y from -1 to 5
    std::vector<hiprt::Vertex> vertices = {
        {-5.0f, -1.0f, -5.0f},  // v0
        { 5.0f, -1.0f, -5.0f},  // v1
        { 5.0f,  5.0f, -5.0f},  // v2
        {-5.0f,  5.0f, -5.0f},  // v3
    };

    std::vector<hiprt::Triangle> triangles = {
        {0, 1, 2},  // First triangle
        {0, 2, 3},  // Second triangle
    };

    std::cout << "\n1. Creating wall geometry..." << std::endl;
    std::cout << "   Vertices:" << std::endl;
    for (size_t i = 0; i < vertices.size(); ++i) {
        std::cout << "     v" << i << ": (" << vertices[i].x << ", "
                  << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
    }
    std::cout << "   Triangles:" << std::endl;
    for (size_t i = 0; i < triangles.size(); ++i) {
        std::cout << "     t" << i << ": v" << triangles[i].v0 << ", v"
                  << triangles[i].v1 << ", v" << triangles[i].v2 << std::endl;
    }

    hiprt::GeometryBuilder geom_builder(context);
    auto geom = geom_builder.build_triangle_geometry(vertices, triangles);
    if (!geom.valid()) {
        std::cerr << "FAILED: Could not build geometry" << std::endl;
        return 1;
    }

    // Build scene with identity transform
    std::cout << "\n2. Building scene with identity transform..." << std::endl;
    hiprt::SceneBuilder scene_builder(context);
    scene_builder.add_instance(geom.get(), 0);  // Identity transform
    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "FAILED: Could not build scene" << std::endl;
        return 1;
    }

    // Compile test kernel
    std::cout << "\n3. Compiling test kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);
    auto compiled = compiler.compile(
        WALL_TEST_KERNEL_SOURCE,
        "wallTestKernel",
        nullptr,  // No custom intersection
        nullptr,
        1,        // 1 geometry type
        1         // 1 ray type
    );
    if (!compiled.valid()) {
        std::cerr << "FAILED: Could not compile kernel" << std::endl;
        return 1;
    }

    // Test configuration
    const uint32_t num_rays_x = 16;
    const uint32_t num_rays_y = 16;
    const uint32_t total_rays = num_rays_x * num_rays_y;

    // Allocate results buffer
    void* d_results = nullptr;
    size_t results_size = total_rays * sizeof(HitResult);
    ORO_CHECK(oroMalloc(&d_results, results_size));

    // Test 1: Fire rays from z=0 towards z=-5 (should hit wall)
    std::cout << "\n4. Test 1: Rays from (0, 2, 0) in direction (0, 0, -1)" << std::endl;
    std::cout << "   Expected: Hit wall at z=-5, hit_t should be 5.0" << std::endl;

    float ray_origin_x = 0.0f, ray_origin_y = 2.0f, ray_origin_z = 0.0f;
    float ray_dir_x = 0.0f, ray_dir_y = 0.0f, ray_dir_z = -1.0f;

    hiprtScene scene_ptr = scene.get();
    void* kernel_args[] = {
        &scene_ptr,
        &ray_origin_x, &ray_origin_y, &ray_origin_z,
        &ray_dir_x, &ray_dir_y, &ray_dir_z,
        &d_results,
        (void*)&num_rays_x,
        (void*)&num_rays_y
    };

    oroFunction kernel_func = reinterpret_cast<oroFunction>(compiled.get_function());
    oroStream stream;
    ORO_CHECK(oroStreamCreate(&stream));

    unsigned int block_x = 8, block_y = 8;
    unsigned int grid_x = (num_rays_x + block_x - 1) / block_x;
    unsigned int grid_y = (num_rays_y + block_y - 1) / block_y;

    oroError err = oroModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, 1,
        block_x, block_y, 1,
        0, stream, kernel_args, nullptr
    );

    if (err != oroSuccess) {
        std::cerr << "FAILED: Kernel launch failed: " << err << std::endl;
        return 1;
    }

    ORO_CHECK(oroStreamSynchronize(stream));

    // Download results
    std::vector<HitResult> results(total_rays);
    ORO_CHECK(oroMemcpyDtoH(results.data(), (oroDeviceptr)d_results, results_size));

    // Analyze results
    int hit_count = 0;
    float min_t = 1e10f, max_t = -1e10f;
    float min_hit_z = 1e10f, max_hit_z = -1e10f;

    for (uint32_t i = 0; i < total_rays; ++i) {
        if (results[i].has_hit) {
            hit_count++;
            min_t = std::min(min_t, results[i].t);
            max_t = std::max(max_t, results[i].t);
            min_hit_z = std::min(min_hit_z, results[i].hit_z);
            max_hit_z = std::max(max_hit_z, results[i].hit_z);
        }
    }

    std::cout << "\n   Results:" << std::endl;
    std::cout << "     Hits: " << hit_count << "/" << total_rays << std::endl;
    if (hit_count > 0) {
        std::cout << "     t range: [" << min_t << ", " << max_t << "]" << std::endl;
        std::cout << "     hit_z range: [" << min_hit_z << ", " << max_hit_z << "]" << std::endl;
    }

    // Print a few sample hits
    std::cout << "\n   Sample hits:" << std::endl;
    int samples = 0;
    for (uint32_t y = 0; y < num_rays_y && samples < 5; y += 4) {
        for (uint32_t x = 0; x < num_rays_x && samples < 5; x += 4) {
            uint32_t idx = y * num_rays_x + x;
            if (results[idx].has_hit) {
                std::cout << "     Ray[" << x << "," << y << "]: t=" << results[idx].t
                          << " hit=(" << results[idx].hit_x << ", "
                          << results[idx].hit_y << ", " << results[idx].hit_z << ")"
                          << " normal=(" << results[idx].normal_x << ", "
                          << results[idx].normal_y << ", " << results[idx].normal_z << ")"
                          << std::endl;
                samples++;
            }
        }
    }

    // Visual representation of hits
    std::cout << "\n   Hit pattern (16x16 grid, '.' = miss, '#' = hit):" << std::endl;
    for (uint32_t y = 0; y < num_rays_y; ++y) {
        std::cout << "     ";
        for (uint32_t x = 0; x < num_rays_x; ++x) {
            uint32_t idx = y * num_rays_x + x;
            std::cout << (results[idx].has_hit ? '#' : '.');
        }
        std::cout << std::endl;
    }

    // Print hit positions for corner rays
    std::cout << "\n   Corner hit positions:" << std::endl;
    for (uint32_t y = 0; y < num_rays_y; y += num_rays_y - 1) {
        for (uint32_t x = 0; x < num_rays_x; x += num_rays_x - 1) {
            uint32_t idx = y * num_rays_x + x;
            float ray_x = (float(x) - float(num_rays_x) / 2.0f) * 0.5f;
            float ray_y = (float(y) - float(num_rays_y) / 2.0f) * 0.5f + 2.0f;
            std::cout << "     Ray origin (" << ray_x << ", " << ray_y << ", 0): ";
            if (results[idx].has_hit) {
                std::cout << "HIT at (" << results[idx].hit_x << ", "
                          << results[idx].hit_y << ", " << results[idx].hit_z << ")";
            } else {
                std::cout << "MISS";
            }
            std::cout << std::endl;
        }
    }

    // Test 2: Fire rays that should miss (above wall)
    std::cout << "\n5. Test 2: Rays from (0, 10, 0) in direction (0, 0, -1)" << std::endl;
    std::cout << "   Expected: All miss (y=10 is above wall which goes to y=5)" << std::endl;

    ray_origin_y = 10.0f;
    err = oroModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, 1,
        block_x, block_y, 1,
        0, stream, kernel_args, nullptr
    );
    ORO_CHECK(oroStreamSynchronize(stream));
    ORO_CHECK(oroMemcpyDtoH(results.data(), (oroDeviceptr)d_results, results_size));

    hit_count = 0;
    for (uint32_t i = 0; i < total_rays; ++i) {
        if (results[i].has_hit) hit_count++;
    }
    std::cout << "   Results: " << hit_count << "/" << total_rays << " hits" << std::endl;

    // Cleanup
    oroStreamDestroy(stream);
    oroFree((oroDeviceptr)d_results);

    std::cout << "\n=== Wall Intersection Test Complete ===" << std::endl;
    context.cleanup();
    return 0;
}
