#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// Simple 3D vector structure
struct float3_aligned {
    float x, y, z;
};

// Ray payload for primary rays
struct RayPayload {
    float3_aligned color;      // Output color (RGB)
    float3_aligned origin;     // Ray origin (for recursive rays)
    float3_aligned direction;  // Ray direction (for recursive rays)
    float t;                   // Hit distance
    int depth;                 // Recursion depth
    bool hit;                  // Whether ray hit anything
};

// Camera parameters
struct Camera {
    float3_aligned position;
    float3_aligned u, v, w;  // Camera basis vectors
    float fov;
};

// Neural asset bounds
struct NeuralAssetBounds {
    float3_aligned min;
    float3_aligned max;
};

// Launch parameters - shared between host and device
struct LaunchParams {
    // Output buffer
    uchar4* frame_buffer;
    uint32_t width;
    uint32_t height;

    // Camera
    Camera camera;

    // Scene traversal
    OptixTraversableHandle traversable;

    // Neural network device pointers (from tiny-cuda-nn)
    void* encoding_ptr;
    void* visibility_network_ptr;
    void* normal_network_ptr;
    void* depth_network_ptr;

    // Neural asset bounds
    NeuralAssetBounds neural_bounds;

    // Background color
    float3_aligned background_color;
};
