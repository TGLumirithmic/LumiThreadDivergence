#include <optix.h>
#include "common.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>

extern "C" {
__constant__ LaunchParams params;
}

// Simple ray-AABB intersection test
static __forceinline__ __device__ bool intersect_aabb(
    const float3& ray_orig,
    const float3& ray_dir,
    const float3& aabb_min,
    const float3& aabb_max,
    float& t_near,
    float& t_far) {

    const float3 inv_dir = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);

    const float3 t0 = make_float3(
        (aabb_min.x - ray_orig.x) * inv_dir.x,
        (aabb_min.y - ray_orig.y) * inv_dir.y,
        (aabb_min.z - ray_orig.z) * inv_dir.z
    );

    const float3 t1 = make_float3(
        (aabb_max.x - ray_orig.x) * inv_dir.x,
        (aabb_max.y - ray_orig.y) * inv_dir.y,
        (aabb_max.z - ray_orig.z) * inv_dir.z
    );

    const float3 t_min = make_float3(
        fminf(t0.x, t1.x),
        fminf(t0.y, t1.y),
        fminf(t0.z, t1.z)
    );

    const float3 t_max = make_float3(
        fmaxf(t0.x, t1.x),
        fmaxf(t0.y, t1.y),
        fmaxf(t0.z, t1.z)
    );

    t_near = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    t_far = fminf(fminf(t_max.x, t_max.y), t_max.z);

    return t_far >= t_near && t_far > 0.0f;
}

// Intersection program for neural asset AABB
extern "C" __global__ void __intersection__neural() {
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    const float3 aabb_min = make_float3(
        params.neural_bounds.min.x,
        params.neural_bounds.min.y,
        params.neural_bounds.min.z
    );

    const float3 aabb_max = make_float3(
        params.neural_bounds.max.x,
        params.neural_bounds.max.y,
        params.neural_bounds.max.z
    );

    float t_near, t_far;
    if (intersect_aabb(ray_orig, ray_dir, aabb_min, aabb_max, t_near, t_far)) {
        // Report intersection at near point
        // Attributes: t and primitive_id (not used for now)
        optixReportIntersection(
            t_near,
            0,  // hit kind
            __float_as_uint(t_near),  // attribute 0
            0                          // attribute 1
        );
    }
}

// Closest-hit program for neural asset
// This is where we call the neural network
extern "C" __global__ void __closesthit__neural() {
    // Get ray information
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = __uint_as_float(optixGetAttribute_0());

    // Compute hit position
    const float3 hit_pos = ray_orig + t_hit * ray_dir;

    // For Phase 2, we'll do a simple visualization without full neural inference
    // In a complete implementation, we would:
    // 1. Call encoding on hit_pos
    // 2. Call the decoders (visibility, normal, depth)
    // 3. Use the outputs to compute color

    // Simple color based on position (for now)
    // This will be replaced with actual neural network inference
    const float3 normalized_pos = make_float3(
        (hit_pos.x - params.neural_bounds.min.x) /
            (params.neural_bounds.max.x - params.neural_bounds.min.x),
        (hit_pos.y - params.neural_bounds.min.y) /
            (params.neural_bounds.max.y - params.neural_bounds.min.y),
        (hit_pos.z - params.neural_bounds.min.z) /
            (params.neural_bounds.max.z - params.neural_bounds.min.z)
    );

    // Visualize position as color (temporary - will be replaced with neural inference)
    const float3 color = make_float3(
        normalized_pos.x,
        normalized_pos.y,
        normalized_pos.z
    );

    // Set payload
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}
