# Warp Efficiency Analysis: Mesh vs Neural Rendering

## TODO: Next Session

### 1. Run Release Build Profiling (With and Without Shadows)

```bash
# Build Release version
cd /workspaces/LumiThreadDivergence/build
cmake -DCMAKE_BUILD_TYPE=Release .. && make renderer_hiprt -j8

# Run profiling WITH shadows (default)
cd /workspaces/LumiThreadDivergence
./profile_branch_divergence.sh

# Run profiling WITHOUT shadows
./profile_branch_divergence.sh --no-shadows
```

This will generate:
- `branch_divergence_results.csv` (with shadows)
- `branch_divergence_results_no_shadows.csv` (without shadows)
- `output/*.ncu-rep` files for both configurations

### 2. Compare Shadow Ray Impact

After profiling completes, compare the two CSV files to see how much shadow rays affect:
- Warp efficiency
- Branch efficiency
- Kernel duration
- Memory stalls

### 3. Debug Build Source-Level Analysis

```bash
# Build Debug version for source correlation
cd /workspaces/LumiThreadDivergence/build
cmake -DCMAKE_BUILD_TYPE=Debug .. && make renderer_hiprt -j8

# Profile a single mesh scene (4 spheres) with source correlation
sudo ncu --set full \
    --source-folders /opt/hiprt,/workspaces/LumiThreadDivergence \
    -o output/mesh_4_debug \
    build/bin/Debug/renderer_hiprt scenes/scene_mesh_4.yaml output/mesh_4_debug.ppm

# Profile a single neural scene (4 spheres) with source correlation
sudo ncu --set full \
    --source-folders /opt/hiprt,/workspaces/LumiThreadDivergence \
    -o output/neural_4_debug \
    build/bin/Debug/renderer_hiprt scenes/scene_neural_4.yaml output/neural_4_debug.ppm
```

### 4. Extract Source-Level Divergence

```bash
# Export source-level metrics to CSV for analysis
ncu --import output/mesh_4_debug.ncu-rep --page source --csv > output/mesh_4_debug_source.csv
ncu --import output/neural_4_debug.ncu-rep --page source --csv > output/neural_4_debug_source.csv
```

Key columns to look at:
- `Function Name` - which function has divergence
- `Warp Stall Sampling (All)` - where threads are waiting
- `Instructions Executed` - how many instructions per function
- `Avg. Threads Executed` - warp efficiency per function

### 5. Identify Top Divergence Sources

Parse the source CSVs to find functions with:
- Low `Avg. Threads Executed` (< 25 out of 32)
- High `Instructions Executed` (significant time spent)

Focus areas based on previous analysis:
- **Mesh**: `pop`, `push`, `intersect` (BVH traversal)
- **Neural**: `hiprtSceneTraversalAnyHit` (shadow rays)

---

## Executive Summary

This analysis compares GPU warp efficiency between traditional mesh-based BVH traversal and neural asset rendering using HIPRT. The goal is to understand how replacing mesh BLAS with neural assets affects thread divergence and overall GPU efficiency.

## Important Clarification: Debug vs Release Builds

The source-level analysis below was performed with **debug builds** (`-G` flag) which have source correlation but different optimization characteristics than release builds. The release build metrics from `branch_divergence_results.csv` show:

| Metric | Mesh (Release) | Neural (Release) |
|--------|----------------|------------------|
| **Warp Efficiency** | 74.25% | 80.81% |
| **Avg Active Threads** | 23.76 | 25.86 |
| **Branch Efficiency** | 99.03% | 99.77% |

The gap in release builds (~6.5%) is smaller than in debug builds (~24%), but neural still outperforms mesh.

## Debug Build Analysis (Source-Level)

| Metric | Mesh | Neural | Difference |
|--------|------|--------|------------|
| **Total Instructions** | 282M | 135M | -52% (neural executes fewer instructions) |
| **Overall Warp Efficiency** | 74.3% | 98.0% | **+23.7%** (neural is more efficient) |
| **Branch Efficiency** | ~99% | ~99.8% | Similar (most branches are uniform) |
| **Avg Active Threads** | ~24/32 | ~26/32 | Neural maintains more active threads |

## Key Finding: Neural is MORE Efficient Than Mesh

Contrary to initial expectations, the **neural renderer achieves significantly higher warp efficiency** than the mesh renderer. This is because:

1. **BVH Traversal Divergence**: Mesh scenes require traversing complex BVH structures where rays diverge through different tree paths. This causes significant warp efficiency loss in stack operations and AABB intersection tests.

2. **Simpler Geometry**: Neural assets use single AABBs with custom intersection, avoiding the deep BVH traversal that causes divergence.

## Function-Level Breakdown

### Functions Where Neural is More Efficient

| Function | Mesh Eff | Neural Eff | Impact | Cause |
|----------|----------|------------|--------|-------|
| `pop` (stack) | 55.2% | 96.5% | High | BVH depth varies per ray |
| `push` (stack) | 61.1% | 98.3% | High | BVH depth varies per ray |
| `intersect` (AABB) | 69.5% | 98.3% | **Highest** | 80M inst in mesh, heavy divergence |
| `fminf`/`fmaxf` | ~71% | ~98% | Medium | AABB math in divergent context |
| `getNextHit` | 84.3% | 99.2% | Medium | Traversal hits more uniform |

### Functions Where Mesh is More Efficient

| Function | Mesh Eff | Neural Eff | Cause |
|----------|----------|------------|-------|
| `hiprtSceneTraversalAnyHit` | 94.7% | 69.7% | Shadow rays diverge more in neural |
| `__cuda_sm3x_div_rn_noftz_f32_slowpath` | 93.6% | 69.7% | FP div on divergent shadow paths |

### Common Low-Efficiency Functions

| Function | Mesh Eff | Neural Eff | Cause |
|----------|----------|------------|-------|
| ~~`log2f`~~ | ~~12.5%~~ | ~~12.5%~~ | ~~Instance entropy calculation~~ **(REMOVED)** |

## Analysis of Specific Issues

### 1. The `log2f` Issue (RESOLVED)

The `log2f` at 12.5% efficiency was in the `warpInstanceEntropy()` function for instance entropy calculation. **This code has been removed** as the divergence metrics weren't providing useful insights. Re-profiling should show improved warp efficiency now that this low-efficiency code path is gone.

### 2. Shadow Ray Divergence in Neural (69.7% efficiency)

The `hiprtSceneTraversalAnyHit` functions run at 69.7% efficiency in neural vs 94.7% in mesh. This occurs because:

1. **Fewer shadow rays traced**: Neural has far fewer instructions in AnyHit (18,972 vs 278,256 in mesh). This means the function runs on a smaller, more divergent subset of rays.

2. **Hit distribution**: After primary ray hits, shadow rays are traced only from hit points. In neural scenes with sparse AABB geometry, hit points are more spatially clustered, leading to more divergent shadow ray paths.

3. **Mesh fills more space**: Triangle-based meshes have continuous surfaces that create more uniform shadow occlusion patterns within a warp.

**Note**: Despite lower efficiency, the neural AnyHit executes ~15x fewer instructions, so the absolute impact is small.

### 3. FP Division Slowpath

The `__cuda_sm3x_div_rn_noftz_f32_slowpath` at 69.7% efficiency in neural appears in:
- Light attenuation calculation: `float attenuation = (light.intensity) / (light_dist * light_dist + 1.0f) / 10.0f;`
- Normal normalization and other shading math

This is executed **only for rays that hit geometry**. Since neural scenes have sparser hit patterns (fewer but larger objects), the active thread mask during shading is less uniform than mesh. The 69.7% efficiency matches the AnyHit functions because they share the same active thread population.

### 4. Why is the Release Build Efficiency Gap Smaller?

The debug build shows neural at 98% vs mesh at 74% (24% gap), but release shows 80.8% vs 74.3% (6.5% gap). This is because:

1. **Compiler optimizations**: Release builds inline functions and optimize control flow, reducing the visibility of per-function divergence patterns.

2. **Different instruction mix**: Optimized code may execute different instruction sequences that aren't directly comparable to debug.

3. **Both builds show neural ahead**: The key finding (neural has better warp efficiency than mesh) holds in both cases.

## Recommendations

### 1. Shadow Ray Toggle (Implemented)

A `--no-shadows` / `-S` command-line flag has been added to disable shadow rays:

```bash
./build/bin/Release/renderer_hiprt scene.yaml output.ppm --no-shadows
```

This allows isolating the shadow ray contribution to warp efficiency degradation.

### 2. Consider Warp-Coherent Shadow Rays

For neural rendering, consider:
- Deferring shadow rays and regrouping by instance
- Using ray sorting/binning before shadow tracing

### 3. Instance Entropy Optimization (DONE)

~~The `log2f` call could be moved out of the kernel if entropy calculation isn't needed per-frame, or computed with a lookup table.~~

**Resolved**: The entire divergence measurement system including `warpInstanceEntropy()` and `measure_divergence()` has been removed from the kernel. This eliminates the 12.5% efficiency `log2f` calls.

## Profiling Methodology

Profiles collected using NVIDIA NSight Compute with:
- Debug build (`-G -lineinfo`) for source correlation
- `--set full` for comprehensive metrics
- `--source-folders` pointing to HIPRT and project directories
- Source-level SASS analysis via `--page source --csv`

## Files Referenced

- `/workspaces/LumiThreadDivergence/src/hiprt/kernel_source.h` - Render kernel implementation
- `/workspaces/LumiThreadDivergence/output/neural_4_debug.ncu-rep` - Neural profile
- `/workspaces/LumiThreadDivergence/output/mesh_4_debug.ncu-rep` - Mesh profile
