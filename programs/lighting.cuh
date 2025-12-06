#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "common.h"
#include "divergence_profiling.cuh"

// Helper to compute squared length
__device__ __forceinline__ float length_squared(const float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Helper to compute length
__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(length_squared(v));
}

// Helper to normalize vector (only if not already defined)
#ifndef NORMALIZE_DEFINED
#define NORMALIZE_DEFINED
__device__ __forceinline__ float3 normalize(const float3& v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}
#endif

// Helper for dot product (only if not already defined)
#ifndef DOT_DEFINED
#define DOT_DEFINED
__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
#endif

// Compute direct lighting with shadow rays
// div_shadow: optional pointer to accumulate shadow ray divergence
__device__ float3 compute_direct_lighting(
    const float3& hit_pos,
    const float3& normal,
    const float3& albedo,
    const float3& light_pos,
    const float3& light_color,
    OptixTraversableHandle traversable,
    unsigned int* div_shadow = nullptr
) {
    // Vector to light
    const float3 to_light = make_float3(
        light_pos.x - hit_pos.x,
        light_pos.y - hit_pos.y,
        light_pos.z - hit_pos.z
    );
    const float distance_to_light = length(to_light);
    const float3 light_dir = make_float3(
        to_light.x / distance_to_light,
        to_light.y / distance_to_light,
        to_light.z / distance_to_light
    );

    // Diffuse term (Lambertian)
    const float n_dot_l = fmaxf(0.0f, dot(normal, light_dir));

    // Measure shadow ray divergence
    bool should_trace_shadow = (n_dot_l > 0.0f);
    if (div_shadow != nullptr) {
        *div_shadow += measure_divergence(should_trace_shadow);
    }

    // Trace shadow ray if surface faces light
    float visibility = 0.0f;
    if (should_trace_shadow) {
        // Shadow ray payload: just visibility
        unsigned int p0 = __float_as_uint(1.0f);  // Default: visible

        optixTrace(
            traversable,
            hit_pos,
            light_dir,
            0.001f,                       // tmin (avoid self-intersection)
            distance_to_light - 0.001f,   // tmax (stop at light)
            0.0f,                         // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            1,   // SBT offset (shadow ray type)
            2,   // SBT stride (primary and shadow)
            1,   // miss SBT index (shadow miss)
            p0   // payload: visibility
        );

        visibility = __uint_as_float(p0);
    }

    // Attenuation (inverse square law)
    const float attenuation = 1.0f / (distance_to_light * distance_to_light);

    // Diffuse contribution
    const float3 diffuse = make_float3(
        albedo.x * light_color.x * n_dot_l * attenuation * visibility,
        albedo.y * light_color.y * n_dot_l * attenuation * visibility,
        albedo.z * light_color.z * n_dot_l * attenuation * visibility
    );

    // Ambient term (simple constant)
    const float3 ambient = make_float3(
        albedo.x * 0.1f,
        albedo.y * 0.1f,
        albedo.z * 0.1f
    );

    return make_float3(
        ambient.x + diffuse.x,
        ambient.y + diffuse.y,
        ambient.z + diffuse.z
    );
}
