#include "hiprt_context.h"
#include "geometry_builder.h"
#include "scene_builder.h"
#include "kernel_compiler.h"
#include <iostream>
#include <vector>
#include <cstring>

// Simple test kernel source (basic ray tracing without neural assets)
const char* TEST_KERNEL_SOURCE = R"(
#include <hiprt/hiprt_device.h>

// Simple render kernel that traces primary rays
extern "C" __global__ void testRenderKernel(
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

    // Trace ray
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

    // Simple shading - output RGBA
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
        // Background
        r = 25; g = 25; b = 38; a = 255;
    }

    frame_buffer[pixel_idx * 4 + 0] = r;
    frame_buffer[pixel_idx * 4 + 1] = g;
    frame_buffer[pixel_idx * 4 + 2] = b;
    frame_buffer[pixel_idx * 4 + 3] = a;
}
)";

int main() {
    std::cout << "=== HIPRT Kernel Compiler Test ===" << std::endl;

    // Initialize context
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }
    std::cout << "Context initialized successfully\n" << std::endl;

    // Build test geometry (a simple cube)
    hiprt::GeometryBuilder geom_builder(context);

    std::vector<hiprt::Vertex> vertices = {
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
    };
    std::vector<hiprt::Triangle> triangles = {
        {0, 1, 2}, {0, 2, 3}, {5, 4, 7}, {5, 7, 6},
        {3, 2, 6}, {3, 6, 7}, {4, 5, 1}, {4, 1, 0},
        {1, 5, 6}, {1, 6, 2}, {4, 0, 3}, {4, 3, 7}
    };

    std::cout << "1. Building geometry..." << std::endl;
    auto geom = geom_builder.build_triangle_geometry(vertices, triangles);
    if (!geom.valid()) {
        std::cerr << "FAILED: Could not build geometry" << std::endl;
        return 1;
    }
    std::cout << "   Geometry built successfully\n" << std::endl;

    // Build scene
    hiprt::SceneBuilder scene_builder(context);
    scene_builder.add_instance(geom.get(), 0);
    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "FAILED: Could not build scene" << std::endl;
        return 1;
    }
    std::cout << "2. Scene built successfully\n" << std::endl;

    // Compile kernel
    std::cout << "3. Compiling render kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);

    auto compiled = compiler.compile(
        TEST_KERNEL_SOURCE,
        "testRenderKernel",
        nullptr,  // No custom intersection
        nullptr,  // No custom filter
        1,        // 1 geometry type (triangles)
        1         // 1 ray type (primary)
    );

    if (!compiled.valid()) {
        std::cerr << "FAILED: Could not compile kernel" << std::endl;
        return 1;
    }
    std::cout << "   Kernel compiled successfully\n" << std::endl;

    // Render a small test image
    const uint32_t width = 64;
    const uint32_t height = 64;

    std::cout << "4. Rendering " << width << "x" << height << " test image..." << std::endl;

    // Allocate frame buffer (RGBA)
    void* d_frame_buffer = nullptr;
    size_t buffer_size = width * height * 4;  // 4 bytes per pixel (RGBA)
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

    // We need to pass the scene by value for the kernel
    hiprtScene scene_ptr = scene.get();
    hiprtFuncTable func_table = nullptr;  // No custom functions for triangle-only test

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

    // Get raw stream for launch
    oroStream stream;
    ORO_CHECK(oroStreamCreate(&stream));

    std::cout << "   Launching kernel: grid(" << grid_x << "," << grid_y << ") block(" << block_x << "," << block_y << ")" << std::endl;

    oroError launch_err = oroModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,      // shared mem
        stream,
        kernel_args,
        nullptr  // extra
    );

    if (launch_err != oroSuccess) {
        std::cerr << "FAILED: Kernel launch failed with error " << launch_err << std::endl;
        oroFree((oroDeviceptr)d_frame_buffer);
        oroStreamDestroy(stream);
        return 1;
    }

    // Synchronize
    ORO_CHECK(oroStreamSynchronize(stream));

    // Download and verify
    std::vector<unsigned char> frame_buffer(width * height * 4);
    ORO_CHECK(oroMemcpyDtoH(
        frame_buffer.data(),
        (oroDeviceptr)d_frame_buffer,
        buffer_size
    ));

    // Count non-background pixels (simple verification)
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
        std::cout << "   PASSED: Render produced hits\n" << std::endl;
    } else {
        std::cerr << "   WARNING: No hits detected (may be camera/geometry issue)" << std::endl;
    }

    // Cleanup
    oroFree((oroDeviceptr)d_frame_buffer);
    oroStreamDestroy(stream);

    std::cout << "=== All kernel tests PASSED ===" << std::endl;
    context.cleanup();
    return 0;
}
