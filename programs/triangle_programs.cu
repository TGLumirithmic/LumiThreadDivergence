#include <optix.h>
#include <optix_device.h>
#include "common.h"
#include "lighting.cuh"
#include "divergence_profiling.cuh"

extern "C" {
__constant__ LaunchParams params;
}

// Triangle closest-hit program (for primary rays)
extern "C" __global__ void __closesthit__triangle() {
    // Get material data from SBT
    const MaterialData& mat = *(const MaterialData*)optixGetSbtDataPointer();

    // Mark this ray as hitting triangle geometry (for program-type divergence measurement in raygen)
    optixSetPayload_24(1);  // 1 = triangle geometry

    // Mark instance ID for BLAS traversal divergence measurement
    optixSetPayload_25(__float_as_uint((float)optixGetInstanceId()));

    // Get ray information
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = optixGetRayTmax();

    // Compute hit position
    const float3 hit_pos = make_float3(
        ray_orig.x + t_hit * ray_dir.x,
        ray_orig.y + t_hit * ray_dir.y,
        ray_orig.z + t_hit * ray_dir.z
    );

    // Get barycentric coordinates (u, v) where w = 1 - u - v
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;
    const float w = 1.0f - u - v;

    // Interpolate vertex normals from mesh data
    float3 world_normal;

    if (mat.vertex_buffer != nullptr && mat.index_buffer != nullptr) {
        // Get triangle index (which triangle in the mesh was hit)
        const unsigned int prim_idx = optixGetPrimitiveIndex();

        // Fetch triangle indices
        const uint3 tri_indices = mat.index_buffer[prim_idx];

        // Fetch vertex normals
        const float3 n0 = mat.vertex_buffer[tri_indices.x].normal;
        const float3 n1 = mat.vertex_buffer[tri_indices.y].normal;
        const float3 n2 = mat.vertex_buffer[tri_indices.z].normal;

        // Interpolate using barycentric coordinates
        world_normal = make_float3(
            w * n0.x + u * n1.x + v * n2.x,
            w * n0.y + u * n1.y + v * n2.y,
            w * n0.z + u * n1.z + v * n2.z
        );

        // Normalize (vertex normals may not be unit length after interpolation)
        const float len = sqrtf(world_normal.x * world_normal.x +
                                world_normal.y * world_normal.y +
                                world_normal.z * world_normal.z);
        if (len > 0.0f) {
            world_normal.x /= len;
            world_normal.y /= len;
            world_normal.z /= len;
        }
    } else {
        // Fallback to hardcoded normals (for cases where buffers aren't set)
        if (fabsf(hit_pos.y - (-1.0f)) < 0.1f) {
            world_normal = make_float3(0.0f, 1.0f, 0.0f);  // Floor
        } else {
            world_normal = make_float3(0.0f, 0.0f, 1.0f);  // Wall
        }
    }

    // Face-forward: ensure normal faces the ray
    if (dot(world_normal, ray_dir) > 0.0f) {
        world_normal = make_float3(-world_normal.x, -world_normal.y, -world_normal.z);
    }

    // Material properties (hardcoded diffuse material)
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

    // Compute direct lighting with shadow rays
    float3 color = compute_direct_lighting(
        hit_pos,
        world_normal,
        albedo,
        light_pos,
        light_color,
        params.traversable,
        &div_shadow
    );

    // Update shadow divergence counter
    optixSetPayload_20(div_shadow);

    // Clamp color to [0, 1]
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));

    // Set payload (color)
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));

    // Set hit position (normalized to [0, 1]³ for visualization)
    // For triangles, just use world position directly
    optixSetPayload_3(__float_as_uint(hit_pos.x));
    optixSetPayload_4(__float_as_uint(hit_pos.y));
    optixSetPayload_5(__float_as_uint(hit_pos.z));

    // Set direction
    optixSetPayload_6(__float_as_uint(ray_dir.x));
    optixSetPayload_7(__float_as_uint(ray_dir.y));
    optixSetPayload_8(__float_as_uint(ray_dir.z));

    // Set unnormalized position (same as hit_pos for triangles)
    optixSetPayload_14(__float_as_uint(hit_pos.x));
    optixSetPayload_15(__float_as_uint(hit_pos.y));
    optixSetPayload_16(__float_as_uint(hit_pos.z));
}

// Triangle intersection program (for custom primitives - not needed, using built-in)
// Note: Built-in triangle intersection is used, so no custom intersection program needed

// Triangle any-hit program (for shadow rays)
extern "C" __global__ void __anyhit__triangle_shadow() {
    // Shadow ray hit something - mark as occluded
    optixSetPayload_0(__float_as_uint(0.0f));  // Not visible
    optixTerminateRay();
}
