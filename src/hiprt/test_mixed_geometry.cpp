#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>

// Write RGBA frame buffer to PPM file for inspection
void write_ppm(const char* filename, const unsigned char* data, uint32_t width, uint32_t height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    // PPM header (P6 = binary RGB)
    file << "P6\n" << width << " " << height << "\n255\n";

    // Write RGB data (skip alpha channel)
    for (uint32_t i = 0; i < width * height; ++i) {
        file.put(data[i * 4 + 0]);  // R
        file.put(data[i * 4 + 1]);  // G
        file.put(data[i * 4 + 2]);  // B
    }

    file.close();
    std::cout << "   Wrote " << filename << " (" << width << "x" << height << ")" << std::endl;
}

// Simple OBJ loader - loads vertices and faces from an OBJ file
bool load_obj(const std::string& filename,
              std::vector<hiprt::Vertex>& vertices,
              std::vector<hiprt::Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "ERROR: Could not open " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            hiprt::Vertex v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (prefix == "f") {
            // Parse face - supports "v", "v/vt", "v/vt/vn", "v//vn" formats
            std::string v1_str, v2_str, v3_str;
            iss >> v1_str >> v2_str >> v3_str;

            auto parse_vertex_index = [](const std::string& s) -> uint32_t {
                size_t pos = s.find('/');
                std::string idx_str = (pos != std::string::npos) ? s.substr(0, pos) : s;
                return static_cast<uint32_t>(std::stoi(idx_str) - 1);  // OBJ is 1-indexed
            };

            hiprt::Triangle tri;
            tri.v0 = parse_vertex_index(v1_str);
            tri.v1 = parse_vertex_index(v2_str);
            tri.v2 = parse_vertex_index(v3_str);
            triangles.push_back(tri);
        }
    }

    std::cout << "   Loaded " << filename << ": " << vertices.size() << " vertices, "
              << triangles.size() << " triangles" << std::endl;
    return !vertices.empty() && !triangles.empty();
}

// Kernel source with both triangle and custom AABB intersection
const char* MIXED_GEOMETRY_KERNEL_SOURCE = R"(
#include <hiprt/hiprt_device.h>

// Custom intersection function for AABB primitives
// Called only for geometry type 1 (custom/neural)
__device__ bool intersectCustomAABB(
    const hiprtRay& ray,
    const void* data,
    void* payload,
    hiprtHit& hit
) {
    // Data contains AABB bounds: [min_x, min_y, min_z, max_x, max_y, max_z] per primitive
    const float* aabb_data = reinterpret_cast<const float*>(data);
    const uint32_t prim_offset = hit.primID * 6;

    float min_x = aabb_data[prim_offset + 0];
    float min_y = aabb_data[prim_offset + 1];
    float min_z = aabb_data[prim_offset + 2];
    float max_x = aabb_data[prim_offset + 3];
    float max_y = aabb_data[prim_offset + 4];
    float max_z = aabb_data[prim_offset + 5];

    // Ray-AABB intersection using slab method
    float inv_dx = 1.0f / ray.direction.x;
    float inv_dy = 1.0f / ray.direction.y;
    float inv_dz = 1.0f / ray.direction.z;

    float t1 = (min_x - ray.origin.x) * inv_dx;
    float t2 = (max_x - ray.origin.x) * inv_dx;
    float t3 = (min_y - ray.origin.y) * inv_dy;
    float t4 = (max_y - ray.origin.y) * inv_dy;
    float t5 = (min_z - ray.origin.z) * inv_dz;
    float t6 = (max_z - ray.origin.z) * inv_dz;

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    if (tmax < 0.0f || tmin > tmax || tmin < ray.minT || tmin > ray.maxT) {
        return false;
    }

    hit.t = tmin;

    // Compute hit point for normal calculation
    float px = ray.origin.x + tmin * ray.direction.x;
    float py = ray.origin.y + tmin * ray.direction.y;
    float pz = ray.origin.z + tmin * ray.direction.z;

    // Determine which face was hit
    float eps = 1e-4f;
    if (fabsf(px - min_x) < eps) {
        hit.normal = hiprtFloat3{-1.0f, 0.0f, 0.0f};
    } else if (fabsf(px - max_x) < eps) {
        hit.normal = hiprtFloat3{1.0f, 0.0f, 0.0f};
    } else if (fabsf(py - min_y) < eps) {
        hit.normal = hiprtFloat3{0.0f, -1.0f, 0.0f};
    } else if (fabsf(py - max_y) < eps) {
        hit.normal = hiprtFloat3{0.0f, 1.0f, 0.0f};
    } else if (fabsf(pz - min_z) < eps) {
        hit.normal = hiprtFloat3{0.0f, 0.0f, -1.0f};
    } else {
        hit.normal = hiprtFloat3{0.0f, 0.0f, 1.0f};
    }

    hit.uv = hiprtFloat2{0.0f, 0.0f};
    return true;
}

// Render kernel for mixed geometry scene
extern "C" __global__ void mixedGeometryKernel(
    hiprtScene scene,
    hiprtFuncTable funcTable,
    float camera_pos_x, float camera_pos_y, float camera_pos_z,
    float camera_dir_x, float camera_dir_y, float camera_dir_z,
    float camera_up_x, float camera_up_y, float camera_up_z,
    float camera_right_x, float camera_right_y, float camera_right_z,
    unsigned char* frame_buffer,
    uint32_t width,
    uint32_t height
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint32_t pixel_idx = y * width + x;

    // Generate ray
    float u = (2.0f * ((float)x + 0.5f) / (float)width - 1.0f);
    float v = (1.0f - 2.0f * ((float)y + 0.5f) / (float)height);

    hiprtRay ray;
    ray.origin = hiprtFloat3{camera_pos_x, camera_pos_y, camera_pos_z};
    ray.direction = hiprtFloat3{
        camera_dir_x + u * camera_right_x + v * camera_up_x,
        camera_dir_y + u * camera_right_y + v * camera_up_y,
        camera_dir_z + u * camera_right_z + v * camera_up_z
    };
    ray.minT = 0.001f;
    ray.maxT = 1e16f;

    // Normalize direction
    float len = sqrtf(ray.direction.x * ray.direction.x +
                      ray.direction.y * ray.direction.y +
                      ray.direction.z * ray.direction.z);
    ray.direction.x /= len;
    ray.direction.y /= len;
    ray.direction.z /= len;

    // Trace ray with function table for custom intersection
    hiprtSceneTraversalClosest traversal(
        scene,
        ray,
        hiprtFullRayMask,
        hiprtTraversalHintDefault,
        nullptr,
        funcTable,
        0,
        0.0f
    );

    hiprtHit hit = traversal.getNextHit();

    // Shading - color based on instance ID
    unsigned char r, g, b, a;
    if (hit.hasHit()) {
        // Instance 0 = sphere mesh (blue-ish)
        // Instance 1 = custom AABB (orange-ish)
        float nx = hit.normal.x;
        float ny = hit.normal.y;
        float nz = hit.normal.z;
        float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
        if (nlen > 0.0f) {
            nx /= nlen; ny /= nlen; nz /= nlen;
        }

        // Simple diffuse lighting from camera direction
        float light_x = -camera_dir_x;
        float light_y = -camera_dir_y;
        float light_z = -camera_dir_z;
        float ndotl = fmaxf(0.0f, nx * light_x + ny * light_y + nz * light_z);
        float ambient = 0.2f;
        float intensity = ambient + 0.8f * ndotl;

        if (hit.instanceID == 0) {
            // Sphere mesh - blue tint
            r = (unsigned char)(intensity * 100.0f);
            g = (unsigned char)(intensity * 150.0f);
            b = (unsigned char)(intensity * 255.0f);
        } else {
            // Custom AABB - orange tint
            r = (unsigned char)(intensity * 255.0f);
            g = (unsigned char)(intensity * 150.0f);
            b = (unsigned char)(intensity * 50.0f);
        }
        a = 255;
    } else {
        // Background - dark gray
        r = 25; g = 25; b = 38; a = 255;
    }

    frame_buffer[pixel_idx * 4 + 0] = r;
    frame_buffer[pixel_idx * 4 + 1] = g;
    frame_buffer[pixel_idx * 4 + 2] = b;
    frame_buffer[pixel_idx * 4 + 3] = a;
}
)";

int main() {
    std::cout << "=== HIPRT Mixed Geometry Test ===" << std::endl;
    std::cout << "Testing scene with both triangle mesh and custom AABB geometry\n" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    hiprt::GeometryBuilder geom_builder(context);

    // 1. Load and build sphere mesh geometry (triangle-based)
    std::cout << "1. Loading sphere mesh..." << std::endl;
    std::vector<hiprt::Vertex> vertices;
    std::vector<hiprt::Triangle> triangles;

    if (!load_obj("data/obj/sphere.obj", vertices, triangles)) {
        std::cerr << "FAILED: Could not load sphere.obj" << std::endl;
        return 1;
    }

    std::cout << "   Building triangle geometry..." << std::endl;
    auto mesh_geom = geom_builder.build_triangle_geometry(vertices, triangles);
    if (!mesh_geom.valid()) {
        std::cerr << "FAILED: Could not build mesh geometry" << std::endl;
        return 1;
    }
    std::cout << "   Mesh geometry built successfully\n" << std::endl;

    // 2. Build AABB geometry (custom primitive)
    std::cout << "2. Building AABB geometry..." << std::endl;
    std::vector<hiprt::AABB> aabbs = {
        {-0.75f, -0.75f, -0.75f, 0.75f, 0.75f, 0.75f}  // Unit cube centered at origin
    };

    // geomType=1 for custom primitives
    auto aabb_geom = geom_builder.build_aabb_geometry(aabbs, 1);
    if (!aabb_geom.valid()) {
        std::cerr << "FAILED: Could not build AABB geometry" << std::endl;
        return 1;
    }
    std::cout << "   AABB geometry built successfully\n" << std::endl;

    // 3. Build scene with both geometries side by side
    std::cout << "3. Building mixed scene..." << std::endl;
    hiprt::SceneBuilder scene_builder(context);

    // Sphere mesh on the left (translated -2 units on X)
    auto mesh_transform = hiprt::translation_transform(-2.0f, 0.0f, 0.0f);
    scene_builder.add_instance(mesh_geom.get(), mesh_transform, 0);

    // AABB box on the right (translated +2 units on X)
    auto aabb_transform = hiprt::translation_transform(2.0f, 0.0f, 0.0f);
    scene_builder.add_instance(aabb_geom.get(), aabb_transform, 1);

    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "FAILED: Could not build scene" << std::endl;
        return 1;
    }
    std::cout << "   Scene built with " << scene_builder.instance_count() << " instances\n" << std::endl;

    // 4. Upload AABB data for custom intersection function
    std::vector<float> aabb_flat_data;
    for (const auto& aabb : aabbs) {
        aabb_flat_data.push_back(aabb.min_x);
        aabb_flat_data.push_back(aabb.min_y);
        aabb_flat_data.push_back(aabb.min_z);
        aabb_flat_data.push_back(aabb.max_x);
        aabb_flat_data.push_back(aabb.max_y);
        aabb_flat_data.push_back(aabb.max_z);
    }

    void* d_aabb_data = nullptr;
    size_t aabb_data_size = aabb_flat_data.size() * sizeof(float);
    ORO_CHECK(oroMalloc(&d_aabb_data, aabb_data_size));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_aabb_data,
                            const_cast<void*>(static_cast<const void*>(aabb_flat_data.data())),
                            aabb_data_size));

    // 5. Compile kernel
    std::cout << "4. Compiling render kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);

    auto compiled = compiler.compile(
        MIXED_GEOMETRY_KERNEL_SOURCE,
        "mixedGeometryKernel",
        "intersectCustomAABB",  // Custom intersection function
        nullptr,                 // No filter function
        2,                       // 2 geometry types (0=triangles, 1=custom)
        2                        // 2 ray types
    );

    if (!compiled.valid()) {
        std::cerr << "FAILED: Could not compile kernel" << std::endl;
        oroFree((oroDeviceptr)d_aabb_data);
        return 1;
    }
    std::cout << "   Kernel compiled successfully\n" << std::endl;

    // Set up function table with AABB data
    hiprtFuncTable func_table = compiled.get_func_table();
    if (func_table) {
        hiprtFuncDataSet data_set;
        data_set.intersectFuncData = d_aabb_data;
        data_set.filterFuncData = nullptr;

        // Set for geomType=1 (custom), rayType=0
        hiprtError err = hiprtSetFuncTable(context.get_context(), func_table, 1, 0, data_set);
        if (err != hiprtSuccess) {
            std::cerr << "WARNING: Failed to set function table data: " << err << std::endl;
        }
    }

    // 6. Render
    const uint32_t width = 512;
    const uint32_t height = 512;

    std::cout << "5. Rendering " << width << "x" << height << " test image..." << std::endl;

    void* d_frame_buffer = nullptr;
    size_t buffer_size = width * height * 4;
    ORO_CHECK(oroMalloc(&d_frame_buffer, buffer_size));

    // Camera parameters - pulled back to see both objects
    float camera_pos_x = 0.0f, camera_pos_y = 0.0f, camera_pos_z = 8.0f;
    float camera_dir_x = 0.0f, camera_dir_y = 0.0f, camera_dir_z = -1.0f;
    float camera_up_x = 0.0f, camera_up_y = 1.0f, camera_up_z = 0.0f;
    float camera_right_x = 1.0f, camera_right_y = 0.0f, camera_right_z = 0.0f;

    unsigned int block_x = 8, block_y = 8, block_z = 1;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    unsigned int grid_y = (height + block_y - 1) / block_y;
    unsigned int grid_z = 1;

    hiprtScene scene_ptr = scene.get();

    void* kernel_args[] = {
        &scene_ptr,
        &func_table,
        &camera_pos_x, &camera_pos_y, &camera_pos_z,
        &camera_dir_x, &camera_dir_y, &camera_dir_z,
        &camera_up_x, &camera_up_y, &camera_up_z,
        &camera_right_x, &camera_right_y, &camera_right_z,
        &d_frame_buffer,
        (void*)&width,
        (void*)&height
    };

    oroFunction kernel_func = reinterpret_cast<oroFunction>(compiled.get_function());

    oroStream stream;
    ORO_CHECK(oroStreamCreate(&stream));

    std::cout << "   Launching kernel..." << std::endl;

    oroError launch_err = oroModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,
        stream,
        kernel_args,
        nullptr
    );

    if (launch_err != oroSuccess) {
        std::cerr << "FAILED: Kernel launch failed with error " << launch_err << std::endl;
        oroFree((oroDeviceptr)d_frame_buffer);
        oroFree((oroDeviceptr)d_aabb_data);
        oroStreamDestroy(stream);
        return 1;
    }

    ORO_CHECK(oroStreamSynchronize(stream));

    // Download and verify
    std::vector<unsigned char> frame_buffer(width * height * 4);
    ORO_CHECK(oroMemcpyDtoH(
        frame_buffer.data(),
        (oroDeviceptr)d_frame_buffer,
        buffer_size
    ));

    // Save output image
    write_ppm("mixed_geometry_test.ppm", frame_buffer.data(), width, height);

    // Count pixels by instance
    int mesh_pixels = 0;   // Blue-ish (instance 0)
    int aabb_pixels = 0;   // Orange-ish (instance 1)
    int bg_pixels = 0;

    for (uint32_t i = 0; i < width * height; ++i) {
        unsigned char r = frame_buffer[i * 4 + 0];
        unsigned char g = frame_buffer[i * 4 + 1];
        unsigned char b = frame_buffer[i * 4 + 2];

        if (r == 25 && g == 25 && b == 38) {
            bg_pixels++;
        } else if (b > r) {
            mesh_pixels++;  // Blue dominant = mesh
        } else {
            aabb_pixels++;  // Red dominant = AABB
        }
    }

    std::cout << "\n   Pixel counts:" << std::endl;
    std::cout << "     Mesh (blue):  " << mesh_pixels << std::endl;
    std::cout << "     AABB (orange): " << aabb_pixels << std::endl;
    std::cout << "     Background:   " << bg_pixels << std::endl;

    bool passed = (mesh_pixels > 0 && aabb_pixels > 0);
    if (passed) {
        std::cout << "\n   PASSED: Both geometry types rendered successfully\n" << std::endl;
    } else {
        std::cerr << "\n   FAILED: Missing hits on one or both geometry types" << std::endl;
        if (mesh_pixels == 0) std::cerr << "     - No mesh hits detected" << std::endl;
        if (aabb_pixels == 0) std::cerr << "     - No AABB hits detected" << std::endl;
    }

    // Cleanup
    oroFree((oroDeviceptr)d_frame_buffer);
    oroFree((oroDeviceptr)d_aabb_data);
    oroStreamDestroy(stream);

    std::cout << "=== Mixed Geometry Test " << (passed ? "PASSED" : "FAILED") << " ===" << std::endl;
    context.cleanup();
    return passed ? 0 : 1;
}
