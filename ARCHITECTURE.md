# LumiThreadDivergence Renderer Architecture

This document describes the architecture of the OptiX Neural Renderer, a hybrid rendering system that combines traditional triangle mesh rendering with neural asset rendering using NVIDIA OptiX 7.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Build System](#build-system)
4. [Core Components](#core-components)
5. [OptiX Pipeline Architecture](#optix-pipeline-architecture)
6. [GPU Programs](#gpu-programs)
7. [Neural Network Integration](#neural-network-integration)
8. [Data Flow](#data-flow)
9. [Shader Binding Table (SBT)](#shader-binding-table-sbt)
10. [Divergence Profiling](#divergence-profiling)

---

## Overview

The renderer is designed to study GPU thread divergence in hybrid neural-mesh rendering scenarios. It supports:

- **Traditional triangle mesh rendering** using OptiX built-in triangle intersection
- **Neural asset rendering** using custom AABB intersection with embedded neural network inference
- **Mixed scenes** with both geometry types in a single TLAS (Top-Level Acceleration Structure)
- **Warp divergence profiling** to measure GPU efficiency

The renderer uses Instant-NGP style hash encoding for position encoding and MLP decoders for visibility, normal, and depth prediction.

---

## Directory Structure

```
LumiThreadDivergence/
├── src/
│   ├── main.cpp                    # Application entry point
│   ├── neural/                     # Neural network host-side code
│   │   ├── config.h/.cpp          # Network configuration
│   │   ├── weight_loader.h/.cpp   # Weight file loading
│   │   └── network.h/.cpp         # Network management
│   ├── optix/                      # OptiX host-side infrastructure
│   │   ├── context.h/.cpp         # CUDA/OptiX context management
│   │   ├── pipeline.h/.cpp        # OptiX pipeline construction
│   │   ├── sbt.h/.cpp             # Shader Binding Table management
│   │   ├── geometry.h/.cpp        # Neural asset BLAS builder
│   │   ├── triangle_geometry.h/.cpp # Triangle mesh BLAS builder
│   │   ├── tlas_builder.h/.cpp    # TLAS construction
│   │   └── neural_params.h/.cpp   # Neural params device upload
│   └── utils/                      # Utility functions
│       ├── error.h                # Error handling macros
│       ├── cuda_utils.h           # CUDA helper functions
│       └── debug_utils.h/.cu      # Debugging utilities
├── programs/                       # OptiX device programs (GPU)
│   ├── common.h                   # Shared structures (LaunchParams)
│   ├── neural_types.h             # Neural network data structures
│   ├── raygen.cu                  # Ray generation program
│   ├── miss.cu                    # Miss program
│   ├── triangle_programs.cu       # Triangle hit/shadow programs
│   ├── neural_programs.cu         # Neural intersection/hit programs
│   ├── neural_inference.cuh       # Neural network inference
│   ├── lighting.cuh               # Direct lighting computation
│   ├── divergence_profiling.cuh   # Warp divergence utilities
│   └── CMakeLists.txt             # PTX compilation rules
├── include/
│   └── vertex.h                   # Vertex structure definition
├── scenes/                        # YAML scene definitions
├── data/                          # Model weights and mesh files
└── CMakeLists.txt                 # Main build configuration
```

---

## Build System

The project uses CMake with the following key configurations:

### Dependencies

| Dependency | Purpose |
|------------|---------|
| CUDA Toolkit | GPU compute and OptiX runtime |
| OptiX 7 SDK | Ray tracing framework |
| tiny-cuda-nn | Reference for hash encoding (header-only) |
| yaml-cpp | Scene file parsing |
| tinyobjloader | OBJ mesh loading |

### Build Process

1. **Host code compilation**: Standard C++17/CUDA compilation for `src/` files
2. **PTX generation**: Device programs in `programs/` are compiled to PTX using nvcc
3. **Linking**: Host executable links against OptiX stubs and CUDA runtime

```cmake
# PTX compilation for each device program
add_ptx_target(raygen raygen.cu)
add_ptx_target(miss miss.cu)
add_ptx_target(neural neural_programs.cu)
add_ptx_target(triangle triangle_programs.cu)
```

The PTX files are output to `build/lib/` and loaded at runtime by the pipeline builder.

---

## Core Components

### Context (`src/optix/context.h`)

Manages CUDA and OptiX initialization:

```cpp
class Context {
    OptixDeviceContext context_;    // OptiX device context
    CUcontext cuda_context_;        // CUDA context
    cudaStream_t stream_;           // CUDA stream for async operations
};
```

### Pipeline (`src/optix/pipeline.h`)

Constructs the OptiX pipeline from PTX modules:

```cpp
class Pipeline {
    // Modules (compiled PTX)
    OptixModule raygen_module_;
    OptixModule miss_module_;
    OptixModule neural_module_;
    OptixModule triangle_module_;

    // Program groups
    OptixProgramGroup raygen_group_;
    OptixProgramGroup miss_group_;
    OptixProgramGroup neural_hit_group_;      // intersection + closesthit
    OptixProgramGroup triangle_hit_group_;    // closesthit only
    OptixProgramGroup triangle_shadow_group_; // anyhit for shadows
    OptixProgramGroup neural_shadow_group_;   // intersection + anyhit
};
```

### Geometry Builders

Two separate builders handle different geometry types:

- **`GeometryBuilder`** (`geometry.h`): Builds AABB-based BLAS for neural assets
- **`TriangleGeometry`** (`triangle_geometry.h`): Builds triangle BLAS for meshes

### TLASBuilder (`src/optix/tlas_builder.h`)

Combines all BLASes into a single traversable structure:

```cpp
class TLASBuilder {
    std::vector<OptixInstance> instances_;
    std::vector<InstanceMetadata> instance_metadata_;

    void add_instance(blas, instance_id, geometry_type, transform);
    OptixTraversableHandle build();
};
```

---

## OptiX Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         OptiX Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                                │
│  │   Raygen    │ ─── Primary rays from camera                   │
│  │ __raygen__rg│                                                │
│  └──────┬──────┘                                                │
│         │ optixTrace()                                          │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                         TLAS                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │  Mesh BLAS  │  │  Mesh BLAS  │  │ Neural BLAS │ ...  │   │
│  │  │ (triangles) │  │  (floor)    │  │   (AABB)    │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ├── Hit triangle ──▶ __closesthit__triangle            │
│         │                                                       │
│         ├── Hit neural AABB ──▶ __intersection__neural         │
│         │                       ──▶ __closesthit__neural       │
│         │                                                       │
│         └── Miss ──▶ __miss__ms                                │
│                                                                 │
│  Shadow rays (from hit programs):                              │
│         ├── Hit triangle ──▶ __anyhit__triangle_shadow         │
│         └── Hit neural ──▶ __intersection__neural              │
│                         ──▶ __anyhit__neural_shadow            │
└─────────────────────────────────────────────────────────────────┘
```

### Ray Types

| Ray Type | SBT Offset | Purpose |
|----------|------------|---------|
| Primary | 0 | Camera rays, find closest hit |
| Shadow | 1 | Light occlusion test |

### Pipeline Configuration

```cpp
pipeline_compile_options_.numPayloadValues = 26;
// Payload layout:
//   p0-p2:   Color (RGB)
//   p3-p5:   World-space hit position
//   p6-p12:  Divergence counters (7 metrics)
//   p13:     Geometry type (0=miss, 1=triangle, 2=neural)
//   p14:     Instance ID
//   p15-p19: Neural inference cache
```

---

## GPU Programs

### Ray Generation (`programs/raygen.cu`)

Entry point for rendering. For each pixel:

1. Compute ray direction from camera parameters
2. Trace primary ray through scene
3. Collect divergence metrics from payload
4. Write color, hit position, and instance ID to buffers

```cuda
extern "C" __global__ void __raygen__rg() {
    // Compute ray from NDC coordinates
    const float3 ray_direction = normalize(
        screen.x * camera_u + screen.y * camera_v + camera_w
    );

    // Trace with 20-slot payload
    optixTrace(traversable, camera_pos, ray_direction, ...);

    // Write outputs
    params.frame_buffer[pixel_idx] = color;
    params.hit_position_buffer[pixel_idx] = hit_pos;
    params.divergence_buffer[...] = divergence_metrics;
}
```

### Miss Program (`programs/miss.cu`)

Returns background color when ray hits nothing:

```cuda
extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(__float_as_uint(params.background_color.x));
    optixSetPayload_1(__float_as_uint(params.background_color.y));
    optixSetPayload_2(__float_as_uint(params.background_color.z));
}
```

### Triangle Programs (`programs/triangle_programs.cu`)

**Closest-Hit** (`__closesthit__triangle`):
1. Interpolate vertex normals using barycentric coordinates
2. Compute direct lighting with shadow rays
3. Return shaded color

**Any-Hit** (`__anyhit__triangle_shadow`):
- Terminate ray immediately (surface is opaque)

### Neural Programs (`programs/neural_programs.cu`)

**Intersection** (`__intersection__neural`):
1. Test ray against AABB bounds
2. If hit, run neural network inference:
   - Hash encode position
   - Encode direction
   - Decode visibility, normal, depth
3. Early rejection if visibility < 0.5
4. Cache neural outputs in payload
5. Report intersection

**Closest-Hit** (`__closesthit__neural`):
1. Read cached neural outputs from payload
2. Compute direct lighting using predicted normal
3. Return shaded color

**Any-Hit** (`__anyhit__neural_shadow`):
- Mark ray as occluded and terminate

---

## Neural Network Integration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Network Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Position (3D)     Direction (3D)                           │
│       │                  │                                  │
│       ▼                  ▼                                  │
│  ┌─────────────┐   ┌─────────────────┐                     │
│  │ Hash Grid   │   │ Direction       │                     │
│  │ Encoding    │   │ Encoder MLP     │                     │
│  │ (16 levels) │   │ (16 → 16)       │                     │
│  └──────┬──────┘   └────────┬────────┘                     │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   │ Concatenate (32 + 16 = 48D)            │
│                   ▼                                         │
│         ┌─────────────────────┐                            │
│         │ Shared Encoding     │                            │
│         │ (48 dimensions)     │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│    ┌───────────────┼───────────────┐                       │
│    ▼               ▼               ▼                        │
│ ┌────────┐   ┌──────────┐   ┌──────────┐                   │
│ │Visibility│ │  Normal  │   │  Depth   │                   │
│ │ Decoder │ │  Decoder │   │  Decoder │                   │
│ │(48→32→1)│ │(48→32→3) │   │(48→32→1) │                   │
│ │ Sigmoid │ │   None   │   │   None   │                   │
│ └────┬────┘ └────┬─────┘   └────┬─────┘                   │
│      │           │              │                          │
│      ▼           ▼              ▼                          │
│ visibility   normal(3D)      depth                         │
│   [0,1]       vector        offset                         │
└─────────────────────────────────────────────────────────────┘
```

### Hash Grid Encoding (`programs/neural_inference.cuh`)

Multi-resolution hash encoding with trilinear interpolation:

```cuda
void hash_encode(const float3& position, const HashGridParams& params, float* output) {
    for (uint32_t level = 0; level < params.n_levels; ++level) {
        float scale = base_resolution * pow(per_level_scale, level);

        // Get voxel corners
        uint32_t x0 = floor(position.x * scale);
        // ... y0, z0

        // Trilinear interpolation of 8 corners
        for (int corner = 0; corner < 8; ++corner) {
            uint32_t hash_idx = hash_grid_index(x + dx, y + dy, z + dz);
            values[corner] = params.hash_table[hash_idx * features + f];
        }

        output[level * 2 + f] = trilinear_interpolate(values, fx, fy, fz);
    }
}
```

### Weight Loading (`src/neural/weight_loader.h`)

Loads pre-trained weights from binary format:

```cpp
class WeightLoader {
    std::map<std::string, Tensor> tensors_;

    bool load_from_file(const std::string& path);
    const Tensor* get_tensor(const std::string& name) const;
};
```

### Device Parameters (`src/optix/neural_params.h`)

Manages GPU memory for neural network weights:

```cpp
class NeuralNetworkParamsHost {
    // Device memory for hash grid
    float* d_hash_table_;
    uint32_t* d_hash_offsets_;

    // Device memory for each MLP
    MLPLayer* d_dir_encoder_layers_;
    MLPLayer* d_vis_decoder_layers_;
    MLPLayer* d_norm_decoder_layers_;
    MLPLayer* d_depth_decoder_layers_;

    bool load_from_weights(const WeightLoader& loader);
    NeuralNetworkParams get_device_params() const;
};
```

---

## Data Flow

### Initialization Flow

```
main.cpp
    │
    ├── Load YAML scene file
    │
    ├── Initialize OptiX Context
    │
    ├── Build Pipeline (load PTX, create program groups)
    │
    ├── For each scene object:
    │   ├── MESH: Build triangle BLAS, add to TLAS
    │   └── NEURAL: Load weights, build AABB BLAS, add to TLAS
    │
    ├── Build TLAS from all instances
    │
    ├── Build SBT with instance metadata
    │
    ├── Upload LaunchParams to device
    │
    └── optixLaunch() → Rendering
```

### Per-Frame Rendering Flow

```
optixLaunch(width, height)
    │
    ├── __raygen__rg (per pixel)
    │   │
    │   └── optixTrace() ──────────────────────────┐
    │                                              │
    │   ┌──────────────────────────────────────────┘
    │   │
    │   ├── TLAS Traversal
    │   │
    │   ├── [Triangle Hit]
    │   │   └── __closesthit__triangle
    │   │       ├── Interpolate normals
    │   │       ├── optixTrace() (shadow ray)
    │   │       │   └── __anyhit__triangle_shadow
    │   │       └── Compute lighting → payload
    │   │
    │   ├── [Neural Hit]
    │   │   ├── __intersection__neural
    │   │   │   ├── AABB intersection test
    │   │   │   ├── hash_encode(position)
    │   │   │   ├── mlp_forward(direction_encoder)
    │   │   │   ├── mlp_forward(visibility_decoder)
    │   │   │   ├── mlp_forward(normal_decoder)
    │   │   │   ├── mlp_forward(depth_decoder)
    │   │   │   └── optixReportIntersection() if visible
    │   │   └── __closesthit__neural
    │   │       ├── Read cached neural outputs
    │   │       ├── optixTrace() (shadow ray)
    │   │       └── Compute lighting → payload
    │   │
    │   └── [Miss]
    │       └── __miss__ms → background color
    │
    └── Write outputs (color, hit_pos, divergence)
```

---

## Shader Binding Table (SBT)

The SBT maps ray types and geometry to program groups:

### SBT Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Raygen Record                                               │
│ [header][empty data]                                        │
├─────────────────────────────────────────────────────────────┤
│ Miss Records (stride = sizeof(MissRecord))                  │
│ [0] Primary miss (background)                               │
│ [1] Shadow miss (no occlusion)                              │
├─────────────────────────────────────────────────────────────┤
│ Hitgroup Records (stride = sizeof(HitgroupRecord))          │
│ For each instance i:                                        │
│   [i*2 + 0] Primary hit program                             │
│   [i*2 + 1] Shadow hit program                              │
│                                                             │
│ Triangle instances:                                         │
│   - MaterialData: albedo, roughness, vertex/index buffers   │
│   - Primary: __closesthit__triangle                         │
│   - Shadow: __anyhit__triangle_shadow                       │
│                                                             │
│ Neural instances:                                           │
│   - MaterialData: neural_params pointer                     │
│   - Primary: __intersection__neural + __closesthit__neural  │
│   - Shadow: __intersection__neural + __anyhit__neural_shadow│
└─────────────────────────────────────────────────────────────┘
```

### SBT Record Structure

```cpp
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct MaterialData {
    float3_aligned albedo;
    float roughness;
    NeuralNetworkParams* neural_params;  // For neural assets
    Vertex* vertex_buffer;               // For triangles
    uint3* index_buffer;                 // For triangles
};
```

---

## Divergence Profiling

### Metrics Tracked

| Index | Metric | Description |
|-------|--------|-------------|
| 0 | RAYGEN | Ray generation divergence |
| 1 | INTERSECTION | AABB hit/miss within warp |
| 2 | CLOSESTHIT | Neural vs triangle geometry type |
| 3 | SHADOW | Shadow ray tracing divergence |
| 4 | HASH_ENCODING | Direct index vs hash lookup |
| 5 | MLP_FORWARD | Hidden vs output layer activation |
| 6 | EARLY_REJECT | Visibility threshold divergence |
| 7 | HIT_MISS | Ray hit vs miss divergence |
| 8 | INSTANCE_ENTROPY | Instance ID variety in warp |

### Measurement Technique

Using CUDA warp intrinsics to measure thread divergence:

```cuda
__device__ uint32_t measure_divergence(bool condition) {
    unsigned int active_mask = __activemask();
    unsigned int ballot = __ballot_sync(active_mask, condition);

    unsigned int true_count = __popc(ballot);
    unsigned int false_count = __popc(active_mask & ~ballot);

    // Divergence = threads that must idle
    return min(true_count, false_count);
}
```

### Instance Entropy Calculation

Measures spatial coherence across instances in a warp:

```cuda
float warpInstanceEntropy(int instanceID) {
    // For each unique instance ID in the warp
    // Calculate: entropy = -sum(p * log2(p))
    // where p = fraction of threads hitting that instance

    // Higher entropy = more scattered rays = worse coherence
}
```

### Output Format

Divergence data is written to a binary file with header:

```
[width: u32][height: u32][num_metrics: u32]
[pixel_0_metric_0: u32][pixel_0_metric_1: u32]...
[pixel_1_metric_0: u32][pixel_1_metric_1: u32]...
...
```

---

## Scene File Format

Scenes are defined in YAML format:

```yaml
scene:
  camera:
    position: [4.33, 7.0, 7.5]
    look_at: [0.0, 2.0, 0.0]
    fov: 90

  light:
    type: point
    position: [0.0, 5.0, 3.0]
    color: [1.0, 1.0, 1.0]
    intensity: 100.0

  objects:
    - type: mesh
      file: data/obj/floor.obj

    - type: neural_asset
      bounds:
        min: [-1.0, -1.0, -1.0]
        max: [1.0, 1.0, 1.0]
      transform:
        position: [0.0, 1.0, 0.0]
        scale: [0.5, 0.5, 0.5]
      weights: data/models/weights.bin
```

---

## Key Implementation Details

### Payload Management

The renderer uses 20 payload registers to pass data between programs:

- **p0-p2**: Final color (RGB floats)
- **p3-p5**: World-space hit position
- **p6-p12**: Divergence counters
- **p13**: Geometry type marker
- **p14**: Instance ID
- **p15-p19**: Neural inference cache (visibility, normal.xyz, depth)

### Neural Inference Caching

To avoid redundant computation, neural inference results are cached in the payload during the intersection program and reused in the closest-hit program:

```cuda
// In __intersection__neural:
optixSetPayload_15(__float_as_uint(visibility));
optixSetPayload_16(__float_as_uint(normal.x));
// ...

// In __closesthit__neural:
float visibility = __uint_as_float(optixGetPayload_15());
float3 normal;
normal.x = __uint_as_float(optixGetPayload_16());
// ...
```

### Transform Handling

Neural assets use object-space AABB with instance transforms:

```cuda
// Intersection program uses object-space ray
const float3 ray_orig = optixGetObjectRayOrigin();
const float3 ray_dir = optixGetObjectRayDirection();

// Convert hit point to world space for shading
float3 world_pos = optixTransformPointFromObjectToWorldSpace(hit_pos);
float3 world_normal = optixTransformNormalFromObjectToWorldSpace(normal);
```

---

## HIPRT Implementation

In addition to the OptiX renderer, there is an alternative HIPRT-based renderer that provides cross-platform ray tracing with full traversal visibility for divergence profiling.

### Why HIPRT?

- **Cross-platform**: Works on both NVIDIA (via Orochi/CUDA) and AMD GPUs
- **Traversal visibility**: Software-based traversal exposes all BVH traversal steps
- **Runtime kernel compilation**: Uses `hiprtBuildTraceKernels()` for flexible kernel customization

### HIPRT Directory Structure

```
src/hiprt/
├── hiprt_context.h/.cpp      # Orochi/HIPRT context initialization
├── geometry_builder.h/.cpp   # Triangle and AABB BLAS construction
├── scene_builder.h/.cpp      # TLAS (scene) construction
├── kernel_compiler.h/.cpp    # Runtime kernel compilation
└── kernel_source.h           # Render kernel source (as string constant)
```

### Key Differences from OptiX

| Aspect | OptiX | HIPRT |
|--------|-------|-------|
| Kernel compilation | PTX at build time | Runtime via `hiprtBuildTraceKernels()` |
| Traversal | Hardware accelerated | Software (with hardware BVH) |
| Custom intersection | `__intersection__` program | Function pointer in function table |
| Hit data | `optixGetAttribute_*()` | `hiprtHit` structure with `.uv`, `.normal`, `.primID` |
| Scene building | `optixAccelBuild()` | `hiprtBuildScene()` with `frameCount >= instanceCount` |

### HIPRT Kernel Architecture

The render kernel is compiled at runtime from `kernel_source.h`:

```cpp
extern "C" __global__ void renderKernel(
    hiprtScene scene,
    hiprtFuncTable funcTable,      // Custom intersection functions
    NeuralAssetData* neural_data,
    MeshData* mesh_data,           // Vertex normals for smooth shading
    uint32_t num_mesh_instances,
    CameraParams camera,
    LightParams light,
    uchar4* frame_buffer,
    TraversalMetrics* metrics_buffer,
    uint32_t width,
    uint32_t height
);
```

### Smooth Shading with Barycentric Interpolation

HIPRT provides barycentric coordinates in `hit.uv` for triangle hits:

```cpp
// Barycentric weights: w = 1 - u - v
float u = hit.uv.x;  // Weight for v1
float v = hit.uv.y;  // Weight for v2
float w = 1.0f - u - v;  // Weight for v0

// Interpolate vertex normals
normal = w * n0 + u * n1 + v * n2;
```

The host code uploads per-mesh vertex normals and triangle indices to GPU memory, allowing the kernel to perform smooth shading by interpolating normals.

### Important HIPRT Learnings

1. **frameCount must equal instanceCount**: When building a scene with multiple instances, `sceneInput.frameCount` must be set to the number of instances, not 1. Each instance needs its own transform frame.

2. **Transform matrix format**: HIPRT uses `hiprtFrameMatrix` with a `float matrix[3][4]` array in row-major format (3 rows × 4 columns).

3. **OBJ normals**: Use normals from OBJ files directly when available. Computing normals from cross-products may give inverted normals depending on winding order.

4. **Custom intersection registration**: Neural/AABB geometry uses `geomType=1` which maps to the custom intersection function via the function table.

### HIPRT Usage

```bash
# Build
mkdir build && cd build
cmake ..
make renderer_hiprt -j

# Run with scene file
./bin/renderer_hiprt scenes/scene_mesh_1.yaml output/render.ppm 512 512
```

---

## Usage

### OptiX Renderer

```bash
# Build
mkdir build && cd build
cmake ..
make -j

# Run with scene file
./bin/renderer scenes/scene_neural_4.yaml output/render.ppm 512 512
```

### HIPRT Renderer

```bash
# Build
make renderer_hiprt -j

# Run with scene file
./bin/renderer_hiprt scenes/scene_mesh_1.yaml output/render.ppm 512 512
```

### Output Files

| File | Content |
|------|---------|
| `render.ppm` | Rendered image (RGB) |
| `render_hit_position.bin` | World-space hit positions (float3 per pixel) |
| `render_instance_id.bin` | Instance IDs (int32 per pixel, -1 for miss) |
| `render_divergence.bin` | Divergence metrics (9 uint32 per pixel) |
