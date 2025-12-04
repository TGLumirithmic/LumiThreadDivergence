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
    float3 payload_hit_pos_normalized = make_float3(0.0f, 0.0f, 0.0f);
    float3 payload_dir = make_float3(0.0f, 0.0f, 0.0f);
    float3 payload_hit_pos_unnormalized = make_float3(0.0f, 0.0f, 0.0f);

    unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    p0 = __float_as_uint(payload_color.x);
    p1 = __float_as_uint(payload_color.y);
    p2 = __float_as_uint(payload_color.z);
    p3 = __float_as_uint(payload_hit_pos_normalized.x);
    p4 = __float_as_uint(payload_hit_pos_normalized.y);
    p5 = __float_as_uint(payload_hit_pos_normalized.z);
    p6 = __float_as_uint(payload_dir.x);
    p7 = __float_as_uint(payload_dir.y);
    p8 = __float_as_uint(payload_dir.z);
    p9 = __float_as_uint(payload_hit_pos_unnormalized.x);
    p10 = __float_as_uint(payload_hit_pos_unnormalized.y);
    p11 = __float_as_uint(payload_hit_pos_unnormalized.z);

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
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);

    // Retrieve color from payload
    payload_color.x = __uint_as_float(p0);
    payload_color.y = __uint_as_float(p1);
    payload_color.z = __uint_as_float(p2);
    payload_hit_pos_normalized.x = __uint_as_float(p3);
    payload_hit_pos_normalized.y = __uint_as_float(p4);
    payload_hit_pos_normalized.z = __uint_as_float(p5);
    payload_dir.x = __uint_as_float(p6);
    payload_dir.y = __uint_as_float(p7);
    payload_dir.z = __uint_as_float(p8);
    payload_hit_pos_unnormalized.x = __uint_as_float(p9);
    payload_hit_pos_unnormalized.y = __uint_as_float(p10);
    payload_hit_pos_unnormalized.z = __uint_as_float(p11);

    // Convert to 8-bit color and write to frame buffer
    const uint32_t pixel_idx = y * params.width + x;
    params.frame_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(fminf(payload_color.x, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.y, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.z, 1.0f) * 255.0f),
        255
    );

    // Write hit position normalized (already normalized to [0,1])
    params.position_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(fminf(fmaxf(payload_hit_pos_normalized.x, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(payload_hit_pos_normalized.y, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(payload_hit_pos_normalized.z, 0.0f), 1.0f) * 255.0f),
        255
    );

    // Write direction (scale from [-1,1] to [0,1] for visualization)
    params.direction_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(fminf(fmaxf(payload_dir.x * 0.5f + 0.5f, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(payload_dir.y * 0.5f + 0.5f, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(payload_dir.z * 0.5f + 0.5f, 0.0f), 1.0f) * 255.0f),
        255
    );

    // Write unnormalized hit position (full precision, world space)
    params.unnormalized_position_buffer[pixel_idx] = make_float3_aligned(payload_hit_pos_unnormalized);
}
