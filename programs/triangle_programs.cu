#include <optix.h>
#include <optix_device.h>
#include "common.h"
#include "lighting.cuh"

extern "C" {
__constant__ LaunchParams params;
}

// Triangle closest-hit program (for primary rays)
extern "C" __global__ void __closesthit__triangle() {
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

    // For built-in triangle intersection, we need to compute the normal from the vertices
    // Since we have hardcoded geometry, we can use the vertex normals directly
    // For now, use a simple face-forward approach: if ray hits back face, flip normal
    // The actual vertex normals are in the geometry, but we'll approximate with the face normal

    // Get barycentric coordinates
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // For hardcoded geometry (floor and walls), we know the normals
    // This is a simplification - in a real renderer we'd fetch vertex data
    // Floor normal: (0, 1, 0), Wall normal: (0, 0, 1)
    // For now, use the geometric normal from cross product of edges (approximation)
    // Face-forward: flip if ray is coming from the back

    // Simple approximation: use the ray direction to determine which surface we hit
    float3 world_normal;
    if (fabsf(hit_pos.y - (-1.0f)) < 0.1f) {
        // Hit floor
        world_normal = make_float3(0.0f, 1.0f, 0.0f);
    } else {
        // Hit wall
        world_normal = make_float3(0.0f, 0.0f, 1.0f);
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

    // Compute direct lighting with shadow rays
    float3 color = compute_direct_lighting(
        hit_pos,
        world_normal,
        albedo,
        light_pos,
        light_color,
        params.traversable
    );

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
