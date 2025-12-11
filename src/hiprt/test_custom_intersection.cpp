#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

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

// Test kernel source with custom intersection function for AABBs
// Uses only HIPRT types to avoid undefined symbol issues
const char* CUSTOM_INTERSECT_KERNEL_SOURCE = R"(
#include <hiprt/hiprt_device.h>

// Global counter for debugging
__device__ unsigned int g_intersect_calls = 0;

// Custom intersection function for AABB primitives
// Signature must match HIPRT expectations:
//   __device__ bool funcName(const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit)
// Note: hit.primID is pre-set by HIPRT before calling this function
__device__ bool intersectCustomAABB(
    const hiprtRay& ray,
    const void* data,
    void* payload,
    hiprtHit& hit
) {
    // DEBUG: Print once per warp to verify function is called
    int lane = threadIdx.x & 31;
    if (lane == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("intersectCustomAABB called! primID=%d data=%p ray.origin=(%f,%f,%f)\\n",
               hit.primID, data, ray.origin.x, ray.origin.y, ray.origin.z);
    }

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

    // Check if ray intersects the box and if hit is within valid range
    // Note: hit.t may start at a large value (ray.maxT), so we need to check against ray.maxT
    if (tmax < 0.0f || tmin > tmax || tmin < ray.minT || tmin > ray.maxT) {
        return false;
    }

    // Compute hit position and normal
    hit.t = tmin;

    // Compute hit point
    float px = ray.origin.x + tmin * ray.direction.x;
    float py = ray.origin.y + tmin * ray.direction.y;
    float pz = ray.origin.z + tmin * ray.direction.z;

    // Determine which face was hit based on which slab boundary was hit last
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

// Render kernel
extern "C" __global__ void customIntersectKernel(
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

    // Trace ray with custom function table
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

    // DEBUG: Print for center pixel
    if (x == 64 && y == 64) {
        printf("DEBUG: Pixel (64,64) ray origin=(%f,%f,%f) dir=(%f,%f,%f)\\n",
               ray.origin.x, ray.origin.y, ray.origin.z,
               ray.direction.x, ray.direction.y, ray.direction.z);
        printf("DEBUG: Pixel (64,64) hit=%d instanceID=%d primID=%d t=%f\\n",
               hit.hasHit() ? 1 : 0, hit.instanceID, hit.primID, hit.t);
    }

    // Shading
    unsigned char r, g, b, a;
    if (hit.hasHit()) {
        // Visualize normal
        float nx = hit.normal.x;
        float ny = hit.normal.y;
        float nz = hit.normal.z;
        float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
        if (nlen > 0.0f) {
            nx /= nlen; ny /= nlen; nz /= nlen;
        }
        r = (unsigned char)((nx * 0.5f + 0.5f) * 255.0f);
        g = (unsigned char)((ny * 0.5f + 0.5f) * 255.0f);
        b = (unsigned char)((nz * 0.5f + 0.5f) * 255.0f);
        a = 255;
    } else {
        r = 25; g = 25; b = 38; a = 255;
    }

    frame_buffer[pixel_idx * 4 + 0] = r;
    frame_buffer[pixel_idx * 4 + 1] = g;
    frame_buffer[pixel_idx * 4 + 2] = b;
    frame_buffer[pixel_idx * 4 + 3] = a;
}
)";

int main() {
    std::cout << "=== HIPRT Custom Intersection Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    // Build AABB geometry (a single box as custom primitive)
    hiprt::GeometryBuilder geom_builder(context);

    std::vector<hiprt::AABB> aabbs = {
        {-0.75f, -0.75f, -0.75f, 0.75f, 0.75f, 0.75f}  // Unit cube centered at origin
    };

    std::cout << "1. Building AABB geometry..." << std::endl;
    // geomType=1 for custom (neural) primitives
    // HIPRT transforms: stored_geomType = input << 1 = 2
    // During traversal: intersectFunc receives stored_geomType >> 1 = 1
    // Function table index = numGeomTypes * rayType + geomType = 2 * 0 + 1 = 1
    auto geom = geom_builder.build_aabb_geometry(aabbs, 1);
    if (!geom.valid()) {
        std::cerr << "FAILED: Could not build AABB geometry" << std::endl;
        return 1;
    }
    std::cout << "   AABB geometry built successfully\n" << std::endl;

    // Build scene
    hiprt::SceneBuilder scene_builder(context);
    scene_builder.add_instance(geom.get(), 0);
    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "FAILED: Could not build scene" << std::endl;
        return 1;
    }
    std::cout << "2. Scene built successfully\n" << std::endl;

    // Upload AABB data for the custom intersection function
    // Format: float[6] per primitive (min_x, min_y, min_z, max_x, max_y, max_z)
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

    // Compile kernel with custom intersection function
    std::cout << "3. Compiling render kernel with custom intersection..." << std::endl;
    hiprt::KernelCompiler compiler(context);

    auto compiled = compiler.compile(
        CUSTOM_INTERSECT_KERNEL_SOURCE,
        "customIntersectKernel",
        "intersectCustomAABB",  // Custom intersection function
        nullptr,                 // No filter function
        2,                       // 2 geometry types (0=triangles, 1=custom/neural)
        2                        // 2 ray types (like main renderer)
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

    // Render
    const uint32_t width = 128;
    const uint32_t height = 128;

    std::cout << "4. Rendering " << width << "x" << height << " test image..." << std::endl;

    void* d_frame_buffer = nullptr;
    size_t buffer_size = width * height * 4;
    ORO_CHECK(oroMalloc(&d_frame_buffer, buffer_size));

    // Camera parameters
    float camera_pos_x = 0.0f, camera_pos_y = 0.0f, camera_pos_z = 3.0f;
    float camera_dir_x = 0.0f, camera_dir_y = 0.0f, camera_dir_z = -1.0f;
    float camera_up_x = 0.0f, camera_up_y = 1.0f, camera_up_z = 0.0f;
    float camera_right_x = 1.0f, camera_right_y = 0.0f, camera_right_z = 0.0f;

    // Grid/block dimensions
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
    write_ppm("custom_intersect_test.ppm", frame_buffer.data(), width, height);

    // Count non-background pixels
    int hit_pixels = 0;
    for (uint32_t i = 0; i < width * height; ++i) {
        unsigned char r = frame_buffer[i * 4 + 0];
        unsigned char g = frame_buffer[i * 4 + 1];
        unsigned char b = frame_buffer[i * 4 + 2];
        if (r != 25 || g != 25 || b != 38) {
            hit_pixels++;
        }
    }

    std::cout << "   Hit pixels: " << hit_pixels << " / " << (width * height) << std::endl;

    if (hit_pixels > 0) {
        std::cout << "   PASSED: Custom intersection produced hits\n" << std::endl;
    } else {
        std::cerr << "   FAILED: No hits detected - custom intersection may not be working" << std::endl;
        oroFree((oroDeviceptr)d_frame_buffer);
        oroFree((oroDeviceptr)d_aabb_data);
        oroStreamDestroy(stream);
        return 1;
    }

    // Cleanup
    oroFree((oroDeviceptr)d_frame_buffer);
    oroFree((oroDeviceptr)d_aabb_data);
    oroStreamDestroy(stream);

    std::cout << "=== Custom Intersection Test PASSED ===" << std::endl;
    context.cleanup();
    return 0;
}
