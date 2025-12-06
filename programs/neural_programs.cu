#include <optix.h>
#include <optix_device.h>
#include "common.h"
#include "lighting.cuh"
#include "divergence_profiling.cuh"

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

#ifndef DOT_DEFINED
#define DOT_DEFINED
static __forceinline__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
#endif

#ifndef NORMALIZE_DEFINED
#define NORMALIZE_DEFINED
static __forceinline__ __device__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}
#endif

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
    // Use object-space rays (transformed by TLAS instance transform)
    const float3 ray_orig = optixGetObjectRayOrigin();
    const float3 ray_dir = optixGetObjectRayDirection();

    // Get instance ID to lookup correct neural asset
    const uint32_t instance_id = optixGetInstanceId();

    // Determine which neural asset to use
    int neural_idx = -1;
    NeuralAssetBounds bounds;
    NeuralNetworkParams neural_net;

    if (params.num_neural_assets > 0 && params.instance_to_neural_map != nullptr) {
        // Multi-neural-asset mode: lookup via mapping
        neural_idx = params.instance_to_neural_map[instance_id];
        if (neural_idx >= 0 && neural_idx < (int)params.num_neural_assets) {
            bounds = params.neural_bounds_array[neural_idx];
            neural_net = params.neural_networks[neural_idx];
        } else {
            // Invalid index, use legacy single neural asset
            bounds = params.neural_bounds;
            neural_net = params.neural_network;
        }
    } else {
        // Legacy single neural asset mode
        bounds = params.neural_bounds;
        neural_net = params.neural_network;
    }

    const float3 aabb_min = make_float3(bounds.min.x, bounds.min.y, bounds.min.z);
    const float3 aabb_max = make_float3(bounds.max.x, bounds.max.y, bounds.max.z);

    // Get divergence counters from payload
    unsigned int div_intersection = optixGetPayload_18();
    unsigned int div_hash = optixGetPayload_21();
    unsigned int div_mlp = optixGetPayload_22();
    unsigned int div_early_reject = optixGetPayload_23();

    float t_near, t_far;
    bool hit_aabb = intersect_aabb(ray_orig, ray_dir, aabb_min, aabb_max, t_near, t_far);

    // Measure AABB intersection divergence
    record_divergence(div_intersection, hit_aabb);

    if (hit_aabb) {
        // Compute hit position at near intersection
        const float3 hit_pos = ray_orig + t_near * ray_dir;

        // Normalize position to [0, 1]^3 (neural network input space)
        const float3 normalized_pos = make_float3(
            (hit_pos.x - bounds.min.x) / (bounds.max.x - bounds.min.x),
            (hit_pos.y - bounds.min.y) / (bounds.max.y - bounds.min.y),
            (hit_pos.z - bounds.min.z) / (bounds.max.z - bounds.min.z)
        );

        // Normalize ray direction
        const float3 normalized_dir = normalize(ray_dir);

        // Run neural network inference
        float visibility;
        float3 predicted_normal;
        float depth_normalised;

        neural_inference(
            normalized_pos,
            normalized_dir,
            neural_net,
            visibility,
            predicted_normal,
            depth_normalised,
            &div_hash,
            &div_mlp
        );

        // Measure early rejection divergence
        record_divergence(div_early_reject, visibility >= 0.5f);

        // Early rejection: only report intersection if visible
        if (visibility >= 0.5f) {
            float aabb_diagonal = sqrt((bounds.max.x-bounds.min.x)*(bounds.max.x-bounds.min.x) + 
            (bounds.max.y-bounds.min.y)*(bounds.max.y-bounds.min.y) +
            (bounds.max.z-bounds.min.z)*(bounds.max.z-bounds.min.z));
            
            float depth = depth_normalised;// * 1.3f;
            float3 intersect_point = hit_pos + depth * ray_dir;
            
            optixSetPayload_9(__float_as_uint(intersect_point.x));
            optixSetPayload_10(__float_as_uint(intersect_point.y));
            optixSetPayload_11(__float_as_uint(intersect_point.z));
            
            // Cache neural inference results in payload for closest-hit
            optixSetPayload_12(__float_as_uint(visibility));
            optixSetPayload_13(__float_as_uint(predicted_normal.x));
            optixSetPayload_14(__float_as_uint(predicted_normal.y));
            optixSetPayload_15(__float_as_uint(predicted_normal.z));
            optixSetPayload_16(__float_as_uint(depth));

            // Report intersection at near point
            optixReportIntersection(
                t_near,
                0,  // hit kind
                __float_as_uint(t_near),  // attribute 0
                0                          // attribute 1
            );
        }
        // If visibility < 0.5, don't report intersection (early rejection)
    }

    // Update divergence counters in payload
    optixSetPayload_18(div_intersection);
    optixSetPayload_21(div_hash);
    optixSetPayload_22(div_mlp);
    optixSetPayload_23(div_early_reject);
}

// Closest-hit program for neural asset
// Reads cached neural network inference results from payload
extern "C" __global__ void __closesthit__neural() {
    // Mark this ray as hitting neural geometry (for program-type divergence measurement in raygen)
    optixSetPayload_24(2);  // 2 = neural geometry

    // Mark instance ID for BLAS traversal divergence measurement
    optixSetPayload_25(__float_as_uint((float)optixGetInstanceId()));

    // Get ray information
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = __uint_as_float(optixGetAttribute_0());
    const float depth = __uint_as_float(optixGetPayload_16());

    // Compute hit position
    const float3 hit_pos = ray_orig + (t_hit + depth) * ray_dir;

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

    // Read cached neural inference results from payload (computed in intersection program)
    // const float visibility = __uint_as_float(optixGetPayload_12());
    float3 predicted_normal;
    predicted_normal.x = __uint_as_float(optixGetPayload_13());
    predicted_normal.y = __uint_as_float(optixGetPayload_14());
    predicted_normal.z = __uint_as_float(optixGetPayload_15());

    

    // Compute final color using network outputs
    // For now: simple shading with predicted normal
    // Visibility modulates opacity, normal determines lighting

    // Normalize the predicted normal
    float3 normal = normalize(predicted_normal);

    // // Simple diffuse shading with a directional light
    // // Light direction (world space, pointing down and to the side)
    // const float3 light_dir = normalize(make_float3(0.5f, -1.0f, 0.3f));

    // // Diffuse term: max(0, -dot(normal, light_dir))
    // // Negative because light_dir points toward light source
    // float diffuse = fmaxf(0.0f, -dot(normal, light_dir));

    // // Ambient term
    // float ambient = 0.2f;

    // // Combine lighting
    // float lighting = ambient + (1.0f - ambient) * diffuse;

    // // Apply visibility (opacity)
    // lighting *= (visibility >= 0.5) ? 1.0: 0.0;

    // // Base color (white for now)
    // const float3 base_color = make_float3(1.0f, 1.0f, 1.0f);

    // Final color
    // const float3 color = make_float3(
    //     base_color.x * lighting,
    //     base_color.y * lighting,
    //     base_color.z * lighting
    // );
    
    const float3 albedo = make_float3(0.8f, 0.8f, 0.8f);

    // Get light parameters
    const float3 light_pos = make_float3(
        params.light.position.x,
        params.light.position.y,
        params.light.position.z
    );
    const float3 light_color = make_float3(
        params.light.color.x * params.light.intensity,
        params.light.color.y * params.light.intensity,
        params.light.color.z * params.light.intensity
    );

    // Get shadow divergence counter for passing to lighting function
    unsigned int div_shadow = optixGetPayload_20();

    float3 color = compute_direct_lighting(
        hit_pos,
        optixTransformNormalFromObjectToWorldSpace(predicted_normal),
        albedo,
        light_pos,
        light_color,
        params.traversable,
        &div_shadow
    );

    // Update shadow divergence counter
    optixSetPayload_20(div_shadow);
    // const float3 to_light = make_float3(
    //     light_pos.x - hit_pos.x,
    //     light_pos.y - hit_pos.y,
    //     light_pos.z - hit_pos.z
    // );
    // const float distance_to_light = length(to_light);
    // const float3 light_dir = make_float3(
    //     to_light.x / distance_to_light,
    //     to_light.y / distance_to_light,
    //     to_light.z / distance_to_light
    // );

    // const float n_dot_l = fmaxf(0.0f, dot(normal, light_dir));

    // float visibility = 1.0f;
    // const float attenuation = 1.0f / (distance_to_light * distance_to_light);

    // // Diffuse contribution
    // const float3 diffuse = make_float3(
    //     albedo.x * light_color.x * n_dot_l * attenuation * visibility,
    //     albedo.y * light_color.y * n_dot_l * attenuation * visibility,
    //     albedo.z * light_color.z * n_dot_l * attenuation * visibility
    // );

    // // Ambient term (simple constant)
    // const float3 ambient = make_float3(
    //     albedo.x * 0.1f,
    //     albedo.y * 0.1f,
    //     albedo.z * 0.1f
    // );

    // float3 color = make_float3(
    //     ambient.x + diffuse.x,
    //     ambient.y + diffuse.y,
    //     ambient.z + diffuse.z
    // );

    // Clamp color to [0, 1]
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));

    // Set payload
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));

    // Set hit position (normalized to [0,1]^3)
    optixSetPayload_3(__float_as_uint(normalized_pos.x));
    optixSetPayload_4(__float_as_uint(normalized_pos.y));
    optixSetPayload_5(__float_as_uint(normalized_pos.z));

    // Set direction (normalized to [-1,1]^3)
    optixSetPayload_6(__float_as_uint(normalized_dir.x));
    optixSetPayload_7(__float_as_uint(normalized_dir.y));
    optixSetPayload_8(__float_as_uint(normalized_dir.z));

    // // Set unnormalized hit position (world space)
    // optixSetPayload_9(__float_as_uint(normalized_pos.x));
    // optixSetPayload_10(__float_as_uint(normalized_pos.y));
    // optixSetPayload_11(__float_as_uint(normalized_pos.z));
}

// Any-hit program for neural asset (for shadow rays)
extern "C" __global__ void __anyhit__neural_shadow() {
    // Shadow ray hit neural asset - mark as occluded
    optixSetPayload_0(__float_as_uint(0.0f));  // Not visible
    optixTerminateRay();
}
