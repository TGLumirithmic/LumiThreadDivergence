# HIPRT Neural Renderer Architecture

This document describes the architecture of the HIPRT-based neural renderer with divergence profiling capabilities.

## Overview

The HIPRT renderer is a software-based ray tracer that replaces an OptiX implementation, providing full visibility into the traversal process for GPU divergence profiling. It supports both traditional triangle meshes and neural network-based geometry representations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main_hiprt.cpp                                  │
│                           (Application Entry Point)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scene Loading (YAML)  │  Geometry Building  │  Kernel Execution  │ Output │
└────────────┬───────────┴─────────┬───────────┴─────────┬──────────┴────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│   yaml-cpp         │  │   src/hiprt/       │  │   GPU Execution    │
│   tinyobjloader    │  │   Core Components  │  │   (Orochi/HIP)     │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

## Directory Structure

```
src/
├── main_hiprt.cpp              # Application entry point
├── hiprt/                      # HIPRT abstraction layer
│   ├── hiprt_context.h/cpp     # GPU context management
│   ├── geometry_builder.h/cpp  # BLAS construction
│   ├── scene_builder.h/cpp     # TLAS construction
│   ├── kernel_compiler.h/cpp   # Runtime kernel compilation
│   ├── kernel_source.h         # GPU kernel source code
│   └── traversal_metrics.h     # Divergence measurement types
└── neural/                     # Neural network support
    ├── weight_loader.h/cpp     # Binary weight file loading
    ├── config.h/cpp            # Network architecture config
    └── network_kernels.h       # Preprocessing kernels
```

## Core Components

### 1. HIPRTContext (`src/hiprt/hiprt_context.h`)

Manages GPU device initialization and provides memory operations.

**Responsibilities:**
- Initialize Orochi (CUDA/HIP abstraction layer)
- Create HIPRT context for ray tracing
- Provide device memory allocation/deallocation
- Handle host-device memory transfers

```cpp
class HIPRTContext {
    hiprtContext m_context;
    oroCtx       m_oro_ctx;
    oroDevice    m_oro_device;
    oroStream    m_stream;
};
```

### 2. GeometryBuilder (`src/hiprt/geometry_builder.h`)

Constructs bottom-level acceleration structures (BLAS) for geometry primitives.

**Supported Geometry Types:**
| geomType | Primitive | Intersection |
|----------|-----------|--------------|
| 0 | Triangles | Built-in |
| 1 | AABBs | Custom (neural) |

```cpp
GeometryHandle build_triangle_geometry(vertices, triangles);
GeometryHandle build_aabb_geometry(aabbs, geomType=1);
```

### 3. SceneBuilder (`src/hiprt/scene_builder.h`)

Constructs the top-level acceleration structure (TLAS) from geometry instances.

**Features:**
- Instance transforms (3x4 affine matrices)
- Per-instance IDs for identification
- Visibility masks for ray filtering

```cpp
void add_instance(geometry, transform, instance_id);
SceneHandle build();
```

### 4. KernelCompiler (`src/hiprt/kernel_compiler.h`)

Handles runtime compilation of GPU kernels with custom intersection support.

**Compilation Pipeline:**
1. Parse kernel source from `kernel_source.h`
2. Register custom intersection functions per (geomType, rayType)
3. Compile via HIPRT's `hiprtBuildTraceKernels()`
4. Create function tables for custom primitives
5. Return compiled kernel with function pointers

### 5. Kernel Source (`src/hiprt/kernel_source.h`)

Contains the complete GPU kernel as an inline string (~1000 lines).

**Key Functions:**
- `renderKernel()` - Main ray tracing entry point
- `intersectNeuralAABB()` - Custom intersection for neural geometry
- `hash_encode()` - Instant-NGP style position encoding
- `mlp_forward()` - Neural network forward pass
- `measure_divergence()` - Warp-level divergence tracking

## Data Flow

### Phase 1: Scene Loading

```
scene.yaml ──► YAML::LoadFile() ──► load_scene_objects()
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                              ▼
            SceneObject::MESH                              SceneObject::NEURAL
                    │                                              │
                    ▼                                              ▼
            tinyobjloader                                  WeightLoader
            (OBJ parsing)                                  (Binary weights)
```

### Phase 2: Acceleration Structure Building

```
┌─────────────────────────────────────────────────────────────────┐
│                       GeometryBuilder                            │
├─────────────────────────────────────────────────────────────────┤
│  Mesh Objects:                    Neural Objects:                │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Triangle Mesh   │              │ AABB Primitive  │           │
│  │ geomType = 0    │              │ geomType = 1    │           │
│  │ Built-in intersect│            │ Custom intersect │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │                                │                     │
│           └────────────┬───────────────────┘                     │
│                        ▼                                         │
│              GeometryHandle (BLAS)                               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SceneBuilder                              │
├─────────────────────────────────────────────────────────────────┤
│  add_instance(geometry, transform, instance_id)                  │
│                        │                                         │
│                        ▼                                         │
│               SceneHandle (TLAS)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Kernel Execution

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Kernel Execution                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Ray Generation                                              │
│      Camera params → Primary ray per pixel                       │
│                                                                  │
│   2. Traversal (HIPRT)                                          │
│      ┌─────────────────────────────────────────────┐            │
│      │  For each BVH node:                         │            │
│      │    - Test ray-box intersection              │            │
│      │    - Measure node divergence                │            │
│      │                                             │            │
│      │  For leaf primitives:                       │            │
│      │    geomType=0 → Built-in triangle test      │            │
│      │    geomType=1 → intersectNeuralAABB()       │            │
│      └─────────────────────────────────────────────┘            │
│                                                                  │
│   3. Neural Intersection (geomType=1)                           │
│      ┌─────────────────────────────────────────────┐            │
│      │  a) Ray-AABB slab test                      │            │
│      │  b) Normalize position to [0,1]³            │            │
│      │  c) Hash grid encoding                      │            │
│      │  d) Direction encoding (MLP)                │            │
│      │  e) Visibility decoder → accept/reject      │            │
│      │  f) Normal decoder → surface normal         │            │
│      │  g) Depth decoder → refined hit distance    │            │
│      └─────────────────────────────────────────────┘            │
│                                                                  │
│   4. Shading                                                     │
│      - Interpolate normals (mesh) or use decoded (neural)       │
│      - Diffuse lighting calculation                             │
│      - Shadow ray for visibility test                           │
│                                                                  │
│   5. Output                                                      │
│      - Write RGBA to frame buffer                               │
│      - Accumulate divergence metrics                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Neural Network Architecture

The renderer uses an Instant-NGP style neural representation:

```
                    Input Position (x, y, z)
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Hash Grid Encoder   │
                   │   16 levels × 2 features│
                   │   = 32D output        │
                   └──────────┬───────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                         ┌─────────────────┐
│ Direction (3D) │                         │  Position Enc   │
│       │        │                         │     (32D)       │
│       ▼        │                         └────────┬────────┘
│ Direction MLP  │                                  │
│  3→16→16       │                                  │
└───────┬───────┘                                   │
        │                                           │
        └─────────────────┬─────────────────────────┘
                          │
                          ▼ Concatenate (48D)
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│ Visibility Decoder│               │  Normal Decoder   │
│  48→32→32→32→32→1 │               │ 48→32→32→32→32→3  │
│  Sigmoid output   │               │  Linear output    │
└─────────┬─────────┘               └─────────┬─────────┘
          │                                   │
          ▼                                   ▼
   Hit probability                    Surface normal
      (0 to 1)                          (nx,ny,nz)

                    ┌───────────────────┐
                    │   Depth Decoder   │
                    │ 48→32→32→32→32→1  │
                    │   Linear output   │
                    └─────────┬─────────┘
                              │
                              ▼
                    Depth within AABB
                         (0 to 1)
```

## Divergence Profiling

The renderer tracks warp-level divergence at multiple execution points:

### TraversalMetrics Structure

```cpp
struct TraversalMetrics {
    uint32_t traversal_steps;        // BVH nodes visited
    uint32_t node_divergence;        // Divergent node visits
    uint32_t triangle_tests;         // Triangle intersections
    uint32_t triangle_divergence;    // Divergent triangle tests
    uint32_t neural_tests;           // Neural intersections
    uint32_t neural_divergence;      // Divergent neural tests
    uint32_t early_reject_divergence;// Divergent early rejections
    uint32_t hash_divergence;        // Hash encoding divergence
    uint32_t mlp_divergence;         // MLP layer divergence
    uint32_t shadow_tests;           // Shadow ray casts
    uint32_t shadow_divergence;      // Divergent shadow rays
    float    instance_entropy;       // Instance ID entropy in warp
};
```

### Measurement Method

Divergence is measured using CUDA warp intrinsics:

```cpp
__device__ uint32_t measure_divergence() {
    uint32_t active = __activemask();
    uint32_t active_count = __popc(active);
    return (active_count < 32) ? 1 : 0;  // Divergent if not all active
}
```

## Key Data Structures

### Host-Side

| Structure | Purpose | Location |
|-----------|---------|----------|
| `HIPRTContext` | GPU context management | hiprt_context.h |
| `GeometryHandle` | RAII wrapper for BLAS | geometry_builder.h |
| `SceneHandle` | RAII wrapper for TLAS | scene_builder.h |
| `NeuralNetworkParamsOrochi` | GPU neural weights | main_hiprt.cpp |
| `MeshNormalData` | Vertex normals for smooth shading | main_hiprt.cpp |

### Device-Side

| Structure | Purpose | Location |
|-----------|---------|----------|
| `TraversalPayload` | Pass data through traversal | kernel_source.h |
| `NeuralAssetData` | Neural asset parameters | kernel_source.h |
| `HashGridParams` | Hash encoding config | kernel_source.h |
| `MLPParams` | MLP layer configuration | kernel_source.h |
| `CameraParams` | Ray generation parameters | kernel_source.h |

## Custom Intersection Protocol

The custom intersection function for neural assets follows HIPRT's callback protocol:

```cpp
__device__ bool intersectNeuralAABB(
    hiprtRay          ray,           // Input ray
    const void*       data,          // AABB bounds data
    void*             payload,       // TraversalPayload*
    hiprtHit&         hit            // Output hit info
) {
    // 1. Ray-AABB intersection test
    // 2. If hit, normalize position to [0,1]³
    // 3. Run neural inference
    // 4. Check visibility threshold (>0.5 = hit)
    // 5. Populate hit.normal, hit.t from network outputs
    // 6. Update divergence metrics
    return hit_accepted;
}
```

## Build Configuration

### Dependencies

- **HIPRT**: AMD's ray tracing library (works on NVIDIA via Orochi)
- **Orochi**: CUDA/HIP abstraction layer
- **yaml-cpp**: YAML scene file parsing
- **tinyobjloader**: OBJ mesh loading

### Compilation

```bash
mkdir build && cd build
cmake ..
make hiprt_renderer
```

## Usage

```bash
./hiprt_renderer <scene.yaml> [output.ppm] [width] [height] [--no-divergence]
```

### Scene File Format

```yaml
scene:
  camera:
    position: [0, 2, 5]
    look_at: [0, 0, 0]
    fov: 45

  light:
    position: [5, 10, 5]
    color: [1, 1, 1]
    intensity: 1.0

  objects:
    - type: mesh
      file: models/bunny.obj
      transform:
        position: [0, 0, 0]
        scale: [1, 1, 1]

    - type: neural_asset
      weights: weights/neural_object.bin
      bounds:
        min: [-1, -1, -1]
        max: [1, 1, 1]
      transform:
        position: [2, 0, 0]
```

## Output

### Image Output
- Format: PPM (Portable Pixmap)
- Color: RGB, 8-bit per channel

### Divergence Metrics Output
- Format: Binary file (`*_divergence.bin`)
- Header: `[width, height, num_metrics]` (3 × uint32)
- Data: `[metrics × pixels]` (12 × width × height × uint32)

## Performance Considerations

1. **Warp Divergence**: Neural inference per-thread without shared memory optimization
2. **Hash Table Access**: May diverge between direct indexing and hash lookup
3. **Function Tables**: Custom intersection adds indirection overhead
4. **Memory Coalescing**: Per-pixel metrics buffer may not be optimal

## Future Enhancements

- [ ] Shared memory optimization for MLP inference
- [ ] Batched neural queries within warps
- [ ] Adaptive divergence measurement granularity
- [ ] Support for additional neural architectures
