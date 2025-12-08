#include <optix.h>
#include <optix_device.h>
#include "common.h"
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
    // Payload layout (20 slots):
    //   p0-p2:   Color (RGB)
    //   p3-p5:   World-space hit position
    //   p6-p12:  Divergence counters (7 slots)
    //   p13:     Geometry type marker (0=miss, 1=triangle, 2=neural)
    //   p14:     Instance ID (-1 for miss, 0+ for hit)
    //   p15-p19: Neural inference cache (visibility, normal.xyz, depth)

    float3 payload_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 payload_hit_pos = make_float3(0.0f, 0.0f, 0.0f);

    unsigned int p0, p1, p2, p3, p4, p5;
    p0 = __float_as_uint(payload_color.x);
    p1 = __float_as_uint(payload_color.y);
    p2 = __float_as_uint(payload_color.z);
    p3 = __float_as_uint(payload_hit_pos.x);
    p4 = __float_as_uint(payload_hit_pos.y);
    p5 = __float_as_uint(payload_hit_pos.z);

    // Divergence profiling payload registers (p6-p12)
    unsigned int p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0;

    // Geometry type marker (p13): 0=miss, 1=triangle, 2=neural
    unsigned int p13 = 0;

    // BLAS traversal: instance_id (-1 for miss, 0+ for hit)
    unsigned int p14 = __float_as_uint(-1.0f);

    // Neural inference cache slots (p15-p19)
    unsigned int p15 = 0, p16 = 0, p17 = 0, p18 = 0, p19 = 0;

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
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);

    // Measure program-type divergence: neural vs triangle/miss
    // When rays in the same warp hit different geometry types, this measures the divergence
    p8 += measure_divergence(p13 == 2);  // true=neural, false=triangle/miss (p8 = DIVERGENCE_CLOSESTHIT)

    // Measure BLAS traversal divergence
    int instance_id = __float_as_int(__uint_as_float(p14));  // -1 for miss, 0+ for hit

    // 1. Hit/Miss divergence - proxy for early BVH exit behavior
    unsigned maskHit = __ballot_sync(__activemask(), instance_id >= 0);
    unsigned numHit = __popc(maskHit);
    unsigned numMiss = 32 - numHit;
    unsigned div_hit_miss = min(numHit, numMiss);

    // 2. Instance entropy - measures spatial coherence across instances
    float entropy = warpInstanceEntropy(instance_id);
    unsigned div_instance_entropy = (unsigned)(entropy * 1000.0f);  // Fixed-point encoding

    // Retrieve color and hit position from payload
    payload_color.x = __uint_as_float(p0);
    payload_color.y = __uint_as_float(p1);
    payload_color.z = __uint_as_float(p2);
    payload_hit_pos.x = __uint_as_float(p3);
    payload_hit_pos.y = __uint_as_float(p4);
    payload_hit_pos.z = __uint_as_float(p5);

    // Convert to 8-bit color and write to frame buffer
    const uint32_t pixel_idx = y * params.width + x;
    params.frame_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(fminf(payload_color.x, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.y, 1.0f) * 255.0f),
        (unsigned char)(fminf(payload_color.z, 1.0f) * 255.0f),
        255
    );

    // Write world-space hit position (full precision)
    params.hit_position_buffer[pixel_idx] = make_float3_aligned(payload_hit_pos);

    // Write instance ID (-1 for miss, 0+ for hit)
    params.instance_id_buffer[pixel_idx] = instance_id;

    // Write divergence profiling metrics
    // New payload mapping: p6-p12 are divergence counters
    if (params.divergence_buffer != nullptr) {
        const uint32_t base_idx = pixel_idx * NUM_DIVERGENCE_METRICS;

        params.divergence_buffer[base_idx + DIVERGENCE_RAYGEN] = p6;
        params.divergence_buffer[base_idx + DIVERGENCE_INTERSECTION] = p7;
        params.divergence_buffer[base_idx + DIVERGENCE_CLOSESTHIT] = p8;
        params.divergence_buffer[base_idx + DIVERGENCE_SHADOW] = p9;
        params.divergence_buffer[base_idx + DIVERGENCE_HASH_ENCODING] = p10;
        params.divergence_buffer[base_idx + DIVERGENCE_MLP_FORWARD] = p11;
        params.divergence_buffer[base_idx + DIVERGENCE_EARLY_REJECT] = p12;
        params.divergence_buffer[base_idx + DIVERGENCE_HIT_MISS] = div_hit_miss;
        params.divergence_buffer[base_idx + DIVERGENCE_INSTANCE_ENTROPY] = div_instance_entropy;
    }
}
