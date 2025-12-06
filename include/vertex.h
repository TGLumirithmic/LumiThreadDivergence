#pragma once

#include <cuda_runtime.h>

// Vertex structure for triangle meshes
// Shared between host and device code
struct Vertex {
    float3 position;
    float3 normal;
};
