#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <optix.h>
#include "neural_inference.cuh"

// ============================================================================
// Warp Divergence Profiling Configuration
// ============================================================================

// Divergence metric indices for the divergence_buffer
#define DIVERGENCE_RAYGEN 0
#define DIVERGENCE_INTERSECTION 1
#define DIVERGENCE_CLOSESTHIT 2
#define DIVERGENCE_SHADOW 3
#define DIVERGENCE_HASH_ENCODING 4
#define DIVERGENCE_MLP_FORWARD 5
#define DIVERGENCE_EARLY_REJECT 6
#define DIVERGENCE_HIT_MISS 7
#define DIVERGENCE_INSTANCE_ENTROPY 8
#define NUM_DIVERGENCE_METRICS 9

// Simple 3D vector structure
struct float3_aligned {
    float x, y, z;
};

// Include Vertex structure (shared with host code)
#include "../include/vertex.h"

// Material data for Shader Binding Table (SBT)
// This struct is accessed by closest-hit programs via optixGetSbtDataPointer()
struct MaterialData {
    float3_aligned albedo;
    float roughness;

    // Neural network parameters (device pointer), nullptr for mesh instances
    NeuralNetworkParams* neural_params;

    // Triangle mesh data (device pointers), nullptr for neural assets
    Vertex* vertex_buffer;   // Device pointer to vertex array
    uint3* index_buffer;     // Device pointer to index array
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

// Point light structure
struct PointLight {
    float3_aligned position;
    float3_aligned color;
    float intensity;
};

// Launch parameters - shared between host and device
struct LaunchParams {
    // Output buffers
    uchar4* frame_buffer;
    uchar4* position_buffer;              // For hit_pos_normalized (quantized)
    uchar4* direction_buffer;             // For normalized_dir (quantized)
    float3_aligned* unnormalized_position_buffer; // For hit_pos (world space, full precision)
    uint32_t width;
    uint32_t height;

    // Camera
    Camera camera;

    // Scene traversal
    OptixTraversableHandle traversable;

    // Neural network parameters (custom OptiX-compatible implementation)
    // Support for multiple neural assets
    uint32_t num_neural_assets;
    NeuralNetworkParams* neural_networks;  // Device pointer to array
    NeuralAssetBounds* neural_bounds_array;  // Device pointer to array

    // Mapping from instance ID to neural asset index
    // instance_to_neural[instance_id] = neural_asset_index
    // -1 if instance is not a neural asset
    int* instance_to_neural_map;  // Device pointer to array

    // Legacy single neural network support (for backward compatibility)
    NeuralNetworkParams neural_network;
    NeuralAssetBounds neural_bounds;

    // Lighting
    PointLight light;

    // Background color
    float3_aligned background_color;

    // Warp divergence profiling output
    // Buffer size: width * height * NUM_DIVERGENCE_METRICS
    // Each pixel stores 7 divergence counters (one per metric)
    uint32_t* divergence_buffer;
};
