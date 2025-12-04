# OptiX Neural Asset Rendering Demo - Specification

## Overview
Create an OptiX-based ray tracer that can render scenes containing both traditional triangle meshes and neural assets (Instant-NGP style NeRFs using tiny-cuda-nn). The goal is to demonstrate SIMD/thread divergence when mixing geometry types, measurable via Nsight Compute profiling.

## Phase 1: Neural Network Integration (Proof of Concept)
**Goal:** Verify we can load PyTorch weights into tiny-cuda-nn and get correct inference results.

**Tasks:**
1. Set up tiny-cuda-nn library integration (link against library, verify CUDA compatibility)
2. Create a simple test harness that:
   - Loads Instant-NGP architecture weights from PyTorch (`.pth` or similar format)
   - Initializes tiny-cuda-nn network with matching architecture (hash encoding + MLP)
   - Runs test queries at known 3D positions
   - Visualizes density/color output (simple image write, doesn't need OptiX yet)
3. Document the weight loading process and network configuration

**Deliverables:**
- Standalone CUDA program that loads PyTorch weights into tiny-cuda-nn
- Test output showing network produces reasonable density/color fields
- Documentation of weight format and loading procedure

## Phase 2: OptiX Integration with Custom Neural Network Implementation
**Goal:** Render a single neural asset using custom CUDA kernels (not calling tiny-cuda-nn from OptiX).

**Approach Change:**
Since tiny-cuda-nn cannot be called directly from OptiX device code (OptiX programs run in a restricted context), we will:
1. Use tiny-cuda-nn for **training only** (Python/PyTorch side)
2. Export trained weights from tiny-cuda-nn
3. **Reimplement the network inference in custom CUDA kernels** that can run inside OptiX programs
4. Load the pre-trained weights into our custom implementation

**Tasks:**
1. Create minimal OptiX pipeline with:
   - Ray generation program ✓
   - Miss program (background color) ✓
   - Custom intersection program for neural asset AABB ✓
   - Closest-hit program that calls custom neural inference kernels
2. Set up neural asset as custom primitive:
   - Define bounding box (AABB) ✓
   - Build BLAS with `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`
   - Implement intersection program (ray-box test, then neural field sampling) ✓
3. **Implement custom CUDA kernels for neural inference** (callable from OptiX):
   - Hash grid encoding (matching tiny-cuda-nn's HashGrid implementation)
   - Fully-fused MLP layers (or simplified MLP with similar architecture)
   - Direction encoder MLP
   - Three decoder MLPs (visibility, normal, depth)
   - All kernels must be `__device__` functions callable from OptiX programs
4. Implement per-ray neural inference in closest-hit:
   - Pass network weight pointers via launch params
   - Query network at hit point on bounding box surface
   - Use normal to compute reflection, specular, use depth to determine hit position
   - Diffuse colour just [1, 1, 1] for now
5. Render a simple test scene: just the neural asset against a background

**Implementation Notes:**
- Network weights will be stored in device memory, passed via `LaunchParams`
- Hash grid parameters: hash table, grid resolution, interpolation coefficients
- MLP weights: organized as layer-by-layer matrices
- Activation functions: ReLU, Sigmoid (as needed)
- All computations must be single-threaded (per-ray) inside OptiX programs

**Deliverables:**
- Custom CUDA neural network kernels compatible with OptiX device code
- Weight loading system that reads tiny-cuda-nn exported weights
- OptiX program that renders a single neural asset using custom inference
- Image output showing recognizable neural field (should match PyTorch training results)
- Validation that outputs match tiny-cuda-nn inference (within tolerance)

## Phase 3: Mixed Geometry Types
**Goal:** Support both triangle meshes and neural assets in the same scene.

**Tasks:**
1. Add triangle mesh support:
   - Load simple mesh data (start with hardcoded cube/plane geometry)
   - Build triangle BLAS with `OPTIX_BUILD_INPUT_TYPE_TRIANGLES`
   - Implement any-hit/closest-hit for triangles (basic diffuse shading)
2. Build TLAS that contains multiple instances:
   - Instance 1: Triangle mesh (floor/walls)
   - Instance 2: Neural asset (AABB-based custom primitive)
   - Implement instance lookup in hit programs to determine geometry type
3. Implement basic path tracing or direct lighting:
   - Single point light
   - Simple material model (diffuse for meshes and neural assets)
4. Test scene: room with walls/floor (triangles), top-down light source + single neural asset

**Deliverables:**
- Scene with mixed geometry types rendering correctly
- TLAS construction code showing how instances are created
- Image showing neural asset integrated with traditional geometry

**TLAS Construction Notes:**
You'll need to explicitly build the TLAS:
```cpp
// Build individual BLASes first
OptixTraversableHandle mesh_blas = build_triangle_blas(...);
OptixTraversableHandle neural_blas = build_custom_blas(...);

// Create instances
OptixInstance instances[2];
instances[0].traversableHandle = mesh_blas;
instances[0].transform = ...; // 3x4 transform matrix
instances[0].instanceId = 0;  // Used to identify geometry type

instances[1].traversableHandle = neural_blas;
instances[1].transform = ...;
instances[1].instanceId = 1;

// Build TLAS
OptixTraversableHandle tlas = build_tlas(instances, 2);
```

## Phase 4: Scene Loading from YAML
**Goal:** Support flexible scene definition without recompiling.

**Tasks:**
1. Define YAML schema for scenes:
```yaml
scene:
  camera:
    position: [x, y, z]
    look_at: [x, y, z]
    fov: 60
  
  lights:
    - type: point
      position: [x, y, z]
      intensity: [r, g, b]
  
  objects:
    - type: mesh
      file: "path/to/mesh.obj"  # or embedded geometry
      transform:
        position: [x, y, z]
        rotation: [x, y, z]
        scale: [x, y, z]
      material:
        type: diffuse
        color: [r, g, b]
    
    - type: neural_asset
      weights: "path/to/weights.pth"
      bounds: 
        min: [x, y, z]
        max: [x, y, z]
      transform:
        position: [x, y, z]
        scale: [x, y, z]
```

2. Implement YAML parser (use yaml-cpp or similar)
3. Build scene from YAML:
   - Load meshes (OBJ or similar format)
   - Load neural asset weights
   - Construct BLASes and TLAS dynamically
   - Set up camera and lighting

**Deliverables:**
- YAML parser that builds OptiX scene
- Example YAML files showing different scene configurations
- Ability to switch scenes without recompiling

## Technical Requirements
- **CUDA Version:** 11.x, 12.x, or 13.x (compatible with tiny-cuda-nn)
- **OptiX Version:** 7.x or 8.x
- **Dependencies:**
  - tiny-cuda-nn
  - OptiX SDK
  - yaml-cpp (for scene loading)
  - Optional: tinyobjloader (for mesh loading)
- **Build System:** CMake

## Testing & Validation
- After Phase 2: Verify neural asset renders match PyTorch ground truth
- After Phase 3: Profile with Nsight Compute to measure baseline divergence
- After Phase 4: Test multiple scene configurations

## Future Extensions (Not in Initial Scope)
- Real-time divergence overlay visualization
- Dynamic scene - camera trajectory
- Multiple neural assets with different weights
- More complex lighting (global illumination)
- Batch inference optimization
- Interactivity

## Notes on Divergence Measurement
For now, we're focusing on getting the infrastructure working. Divergence analysis will be done via Nsight Compute post-processing:
```bash
ncu --set full --target-processes all ./renderer scene.yaml
```

Look for metrics: `smsp__sass_average_branch_targets_threads_uniform.pct` and warp execution efficiency during neural asset intersection/closest-hit programs.
