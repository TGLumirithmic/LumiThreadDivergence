#include <optix.h>
#include "common.h"

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __miss__ms() {
    // Return background color
    const float3 bg_color = make_float3(
        params.background_color.x,
        params.background_color.y,
        params.background_color.z
    );

    // Set payload
    optixSetPayload_0(__float_as_uint(bg_color.x));
    optixSetPayload_1(__float_as_uint(bg_color.y));
    optixSetPayload_2(__float_as_uint(bg_color.z));
}
