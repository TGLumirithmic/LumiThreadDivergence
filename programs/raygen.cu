#include <optix.h>
#include "common.h"

extern "C" {
__constant__ LaunchParams params;
}

// Helper to convert float3_aligned to float3
static __forceinline__ __device__ float3 make_float3_from_aligned(const float3_aligned& a) {
    return make_float3(a.x, a.y, a.z);
}

// Helper to convert float3 to float3_aligned
static __forceinline__ __device__ float3_aligned make_float3_aligned(const float3& f) {
    float3_aligned result;
    result.x = f.x;
    result.y = f.y;
    result.z = f.z;
    return result;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const uint32_t x = idx.x;
    const uint32_t y = idx.y;

    // Compute normalized device coordinates
    const float2 ndc = make_float2(
        (float)x / (float)params.width,
        (float)y / (float)params.height
    );

    // Convert to screen space [-1, 1]
    const float2 screen = make_float2(
        2.0f * ndc.x - 1.0f,
        1.0f - 2.0f * ndc.y  // Flip Y
    );

    // Compute ray direction from camera
    const float3 camera_pos = make_float3_from_aligned(params.camera.position);
    const float3 camera_u = make_float3_from_aligned(params.camera.u);
    const float3 camera_v = make_float3_from_aligned(params.camera.v);
    const float3 camera_w = make_float3_from_aligned(params.camera.w);

    const float3 ray_direction = normalize(
        screen.x * camera_u +
        screen.y * camera_v +
        camera_w
    );

    // Trace ray
    float3 payload_color = make_float3(0.0f, 0.0f, 0.0f);

    unsigned int p0, p1, p2;
    p0 = __float_as_uint(payload_color.x);
    p1 = __float_as_uint(payload_color.y);
    p2 = __float_as_uint(payload_color.z);

    optixTrace(
        params.traversable,
        camera_pos,
        ray_direction,
        0.0f,                // tmin
        1e16f,               // tmax
        0.0f,                // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        1,                   // SBT stride
        0,                   // missSBTIndex
        p0, p1, p2);

    // Retrieve color from payload
    payload_color.x = __uint_as_float(p0);
    payload_color.y = __uint_as_float(p1);
    payload_color.z = __uint_as_float(p2);

    // Convert to 8-bit color and write to frame buffer
    const uint32_t pixel_idx = y * params.width + x;
    params.frame_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(fminf(payload_color.x, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.y, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.z, 1.0f) * 255.0f),
        255
    );
}
