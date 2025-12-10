// Test floor + wall intersection to debug why wall disappears
#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include <iostream>
#include <vector>
#include <cmath>

const char* TWO_PLANES_KERNEL = R"(
#include <hiprt/hiprt_device.h>

struct HitInfo {
    float t;
    float hit_x, hit_y, hit_z;
    float normal_x, normal_y, normal_z;
    int instance_id;
    int has_hit;
};

extern "C" __global__ void testTwoPlanesKernel(
    hiprtScene scene,
    float* ray_origins,  // 3 floats per ray
    float* ray_dirs,     // 3 floats per ray
    HitInfo* results,
    uint32_t num_rays
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    hiprtRay ray;
    ray.origin = hiprtFloat3{ray_origins[idx*3], ray_origins[idx*3+1], ray_origins[idx*3+2]};
    ray.direction = hiprtFloat3{ray_dirs[idx*3], ray_dirs[idx*3+1], ray_dirs[idx*3+2]};
    ray.minT = 0.001f;
    ray.maxT = 1000.0f;

    hiprtSceneTraversalClosest traversal(scene, ray, hiprtFullRayMask);
    hiprtHit hit = traversal.getNextHit();

    results[idx].has_hit = hit.hasHit() ? 1 : 0;
    results[idx].instance_id = hit.instanceID;
    if (hit.hasHit()) {
        results[idx].t = hit.t;
        results[idx].hit_x = ray.origin.x + hit.t * ray.direction.x;
        results[idx].hit_y = ray.origin.y + hit.t * ray.direction.y;
        results[idx].hit_z = ray.origin.z + hit.t * ray.direction.z;
        results[idx].normal_x = hit.normal.x;
        results[idx].normal_y = hit.normal.y;
        results[idx].normal_z = hit.normal.z;
    }
}
)";

struct HitInfo {
    float t;
    float hit_x, hit_y, hit_z;
    float normal_x, normal_y, normal_z;
    int instance_id;
    int has_hit;
};

int main() {
    std::cout << "=== Two Planes Intersection Test ===" << std::endl;

    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "Failed to initialize context" << std::endl;
        return 1;
    }

    hiprt::GeometryBuilder geom_builder(context);

    // Create floor (y=-1 plane)
    std::vector<hiprt::Vertex> floor_verts = {
        {-5.0f, -1.0f, -5.0f},
        { 5.0f, -1.0f, -5.0f},
        { 5.0f, -1.0f,  5.0f},
        {-5.0f, -1.0f,  5.0f},
    };
    std::vector<hiprt::Triangle> floor_tris = {{0, 1, 2}, {0, 2, 3}};

    std::cout << "\n1. Creating floor geometry (y=-1)..." << std::endl;
    auto floor_geom = geom_builder.build_triangle_geometry(floor_verts, floor_tris);

    // Create wall (z=-5 plane)
    // Swap winding order to make front face point towards +Z (towards camera)
    std::vector<hiprt::Vertex> wall_verts = {
        {-5.0f, -1.0f, -5.0f},
        { 5.0f, -1.0f, -5.0f},
        { 5.0f,  5.0f, -5.0f},
        {-5.0f,  5.0f, -5.0f},
    };
    // Original: {0, 1, 2}, {0, 2, 3} - CCW from front = normal points -Z
    // Swapped:  {0, 2, 1}, {0, 3, 2} - CW from front = normal points +Z
    std::vector<hiprt::Triangle> wall_tris = {{0, 2, 1}, {0, 3, 2}};

    std::cout << "2. Creating wall geometry (z=-5)..." << std::endl;
    auto wall_geom = geom_builder.build_triangle_geometry(wall_verts, wall_tris);

    // Build scene with both - keep geometry handles alive!
    std::cout << "\n3. Building scene with floor (instance 0) and wall (instance 1)..." << std::endl;

    // Store geometry handles to keep them alive during scene build
    std::vector<hiprt::GeometryHandle*> geom_handles;
    geom_handles.push_back(&floor_geom);
    geom_handles.push_back(&wall_geom);

    hiprt::SceneBuilder scene_builder(context);
    scene_builder.add_instance(floor_geom.get(), 0);  // Instance 0 = floor
    scene_builder.add_instance(wall_geom.get(), 1);   // Instance 1 = wall

    std::cout << "   Floor geometry ptr: " << (void*)floor_geom.get() << std::endl;
    std::cout << "   Wall geometry ptr: " << (void*)wall_geom.get() << std::endl;

    auto scene = scene_builder.build();

    // Geometry handles are still alive here
    std::cout << "   After build - Floor ptr: " << (void*)floor_geom.get() << std::endl;
    std::cout << "   After build - Wall ptr: " << (void*)wall_geom.get() << std::endl;

    // Compile kernel
    std::cout << "\n4. Compiling kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);
    auto compiled = compiler.compile(TWO_PLANES_KERNEL, "testTwoPlanesKernel", nullptr, nullptr, 1, 1);
    if (!compiled.valid()) {
        std::cerr << "Failed to compile kernel" << std::endl;
        return 1;
    }

    // Create test rays
    // Ray 1: Should hit floor only (pointing down)
    // Ray 2: Should hit wall only (pointing at wall, above floor level)
    // Ray 3: Should hit floor first, wall is behind (typical camera ray)

    float cam_pos[3] = {4.09576f, 7.73576f, 7.09406f};

    std::vector<float> ray_origins;
    std::vector<float> ray_dirs;
    std::vector<std::string> ray_names;

    // Ray 0: Straight down from above floor - should hit floor
    ray_origins.insert(ray_origins.end(), {0.0f, 5.0f, 0.0f});
    ray_dirs.insert(ray_dirs.end(), {0.0f, -1.0f, 0.0f});
    ray_names.push_back("Straight down (should hit floor)");

    // Ray 1: From camera towards wall area (screen 230, 180) - should hit wall
    ray_origins.insert(ray_origins.end(), {cam_pos[0], cam_pos[1], cam_pos[2]});
    ray_dirs.insert(ray_dirs.end(), {-0.5542f, -0.3170f, -0.7697f});  // Normalized
    ray_names.push_back("Camera ray to wall area (230,180)");

    // Ray 2: From camera towards floor area (screen 300, 400) - should hit floor
    ray_origins.insert(ray_origins.end(), {cam_pos[0], cam_pos[1], cam_pos[2]});
    ray_dirs.insert(ray_dirs.end(), {-0.3f, -0.6f, -0.74f});  // Approx normalized
    ray_names.push_back("Camera ray to floor area");

    // Ray 3: Horizontal towards wall at wall height - should hit wall
    ray_origins.insert(ray_origins.end(), {0.0f, 2.0f, 5.0f});
    ray_dirs.insert(ray_dirs.end(), {0.0f, 0.0f, -1.0f});
    ray_names.push_back("Horizontal to wall (z=-5)");

    uint32_t num_rays = ray_names.size();

    // Allocate GPU memory
    void *d_origins, *d_dirs, *d_results;
    ORO_CHECK(oroMalloc(&d_origins, ray_origins.size() * sizeof(float)));
    ORO_CHECK(oroMalloc(&d_dirs, ray_dirs.size() * sizeof(float)));
    ORO_CHECK(oroMalloc(&d_results, num_rays * sizeof(HitInfo)));

    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_origins, ray_origins.data(), ray_origins.size() * sizeof(float)));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_dirs, ray_dirs.data(), ray_dirs.size() * sizeof(float)));

    // Launch kernel
    hiprtScene scene_ptr = scene.get();
    void* args[] = {&scene_ptr, &d_origins, &d_dirs, &d_results, &num_rays};

    oroFunction func = reinterpret_cast<oroFunction>(compiled.get_function());
    oroStream stream;
    ORO_CHECK(oroStreamCreate(&stream));

    oroModuleLaunchKernel(func, 1, 1, 1, num_rays, 1, 1, 0, stream, args, nullptr);
    ORO_CHECK(oroStreamSynchronize(stream));

    // Download results
    std::vector<HitInfo> results(num_rays);
    ORO_CHECK(oroMemcpyDtoH(results.data(), (oroDeviceptr)d_results, num_rays * sizeof(HitInfo)));

    // Print results
    std::cout << "\n5. Results:" << std::endl;
    for (uint32_t i = 0; i < num_rays; ++i) {
        std::cout << "\n   Ray " << i << ": " << ray_names[i] << std::endl;
        std::cout << "     Origin: (" << ray_origins[i*3] << ", " << ray_origins[i*3+1] << ", " << ray_origins[i*3+2] << ")" << std::endl;
        std::cout << "     Dir: (" << ray_dirs[i*3] << ", " << ray_dirs[i*3+1] << ", " << ray_dirs[i*3+2] << ")" << std::endl;

        if (results[i].has_hit) {
            std::cout << "     HIT instance " << results[i].instance_id
                      << (results[i].instance_id == 0 ? " (FLOOR)" : " (WALL)") << std::endl;
            std::cout << "       t = " << results[i].t << std::endl;
            std::cout << "       pos = (" << results[i].hit_x << ", " << results[i].hit_y << ", " << results[i].hit_z << ")" << std::endl;
            std::cout << "       normal = (" << results[i].normal_x << ", " << results[i].normal_y << ", " << results[i].normal_z << ")" << std::endl;
        } else {
            std::cout << "     MISS" << std::endl;
        }
    }

    // Cleanup
    oroStreamDestroy(stream);
    oroFree((oroDeviceptr)d_origins);
    oroFree((oroDeviceptr)d_dirs);
    oroFree((oroDeviceptr)d_results);

    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
