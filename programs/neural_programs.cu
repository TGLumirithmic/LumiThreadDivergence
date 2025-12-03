#include <optix.h>
#include <optix_device.h>
#include "common.h"

extern "C" {
__constant__ LaunchParams params;
}

// CUDA vector math helpers
static __forceinline__ __device__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
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
// Calls custom neural network inference
extern "C" __global__ void __closesthit__neural() {
    // Get ray information
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = __uint_as_float(optixGetAttribute_0());

    // Compute hit position
    const float3 hit_pos = ray_orig + t_hit * ray_dir;

    // Normalize position to [0, 1]^3 (neural network input space)
    const float3 normalized_pos = make_float3(
        (hit_pos.x - params.neural_bounds.min.x) /
            (params.neural_bounds.max.x - params.neural_bounds.min.x),
        (hit_pos.y - params.neural_bounds.min.y) /
            (params.neural_bounds.max.y - params.neural_bounds.min.y),
        (hit_pos.z - params.neural_bounds.min.z) /
            (params.neural_bounds.max.z - params.neural_bounds.min.z)
    );

    // Normalize ray direction (should already be normalized, but ensure)
    const float3 normalized_dir = normalize(ray_dir);

    // Run neural network inference
    float visibility;
    float3 predicted_normal;
    float depth;

    neural_inference(
        normalized_pos,
        normalized_dir,
        params.neural_network,
        visibility,
        predicted_normal,
        depth
    );

    // Compute final color using network outputs
    // For now: simple shading with predicted normal
    // Visibility modulates opacity, normal determines lighting

    // Normalize the predicted normal
    float3 normal = normalize(predicted_normal);

    // Simple diffuse shading with a directional light
    // Light direction (world space, pointing down and to the side)
    const float3 light_dir = normalize(make_float3(0.5f, -1.0f, 0.3f));

    // Diffuse term: max(0, -dot(normal, light_dir))
    // Negative because light_dir points toward light source
    float diffuse = fmaxf(0.0f, -dot(normal, light_dir));

    // Ambient term
    float ambient = 0.2f;

    // Combine lighting
    float lighting = ambient + (1.0f - ambient) * diffuse;

    // Apply visibility (opacity)
    lighting *= (visibility >= 0.5) ? 1.0: 0.0;

    // Base color (white for now)
    const float3 base_color = make_float3(1.0f, 1.0f, 1.0f);

    // Final color
    const float3 color = make_float3(
        base_color.x * lighting,
        base_color.y * lighting,
        base_color.z * lighting
    );

    // Set payload
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}
