# Nsight Compute Profiling Guide

This guide explains how to profile the neural OptiX renderer to measure thread divergence when mixing traditional geometry with neural assets.

## Prerequisites

- NVIDIA Nsight Compute installed
- Completed Phase 3 (mixed geometry rendering)
- Test scene with both triangle meshes and neural assets

## Installation

### Linux
```bash
# Usually installed with CUDA toolkit
/usr/local/cuda/bin/ncu --version

# Or download standalone
# https://developer.nvidia.com/nsight-compute
```

### Windows
```bash
"C:\Program Files\NVIDIA Corporation\Nsight Compute\ncu.exe" --version
```

## Basic Profiling

### Profile the Renderer

```bash
# Basic profiling (default metrics)
ncu ./build/bin/renderer data/scenes/mixed.yaml

# Full metric collection
ncu --set full ./build/bin/renderer data/scenes/mixed.yaml

# Save report for later analysis
ncu --set full -o profile_report ./build/bin/renderer data/scenes/mixed.yaml
```

### Profile Specific Kernels

```bash
# Profile only OptiX kernels
ncu --kernel-name "optix::" ./build/bin/renderer data/scenes/mixed.yaml

# Profile neural network kernels
ncu --kernel-name-base regex:__raygen__ ./build/bin/renderer data/scenes/mixed.yaml
```

## Key Metrics for Thread Divergence

### Warp Execution Efficiency

The primary metric for measuring divergence:

```bash
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct \
    ./build/bin/renderer data/scenes/mixed.yaml
```

**Interpretation:**
- **100%:** All threads in a warp execute the same code path (perfect)
- **50-99%:** Moderate divergence
- **<50%:** High divergence (expected when mixing geometry types)

### Branch Efficiency

```bash
ncu --metrics smsp__sass_branch_targets_threads_uniform.pct,\
               smsp__sass_branch_targets.avg,\
               smsp__sass_average_branch_targets_threads_uniform.pct \
    ./build/bin/renderer data/scenes/mixed.yaml
```

### Full Divergence Analysis

```bash
ncu --set full \
    --section SpeedOfLight \
    --section WarpStateStats \
    --section Occupancy \
    -o divergence_report \
    ./build/bin/renderer data/scenes/mixed.yaml
```

## Analyzing Results

### View Interactive Report

```bash
# Linux
ncu-ui divergence_report.ncu-rep

# Or export to CSV
ncu --import divergence_report.ncu-rep --csv > results.csv
```

### Compare Scenarios

Profile different scene configurations to measure divergence impact:

```bash
# Scene 1: Only triangles
ncu -o triangles_only --set full ./build/bin/renderer data/scenes/triangles.yaml

# Scene 2: Only neural assets
ncu -o neural_only --set full ./build/bin/renderer data/scenes/neural.yaml

# Scene 3: Mixed geometry
ncu -o mixed --set full ./build/bin/renderer data/scenes/mixed.yaml

# Compare in UI
ncu-ui triangles_only.ncu-rep neural_only.ncu-rep mixed.ncu-rep
```

## Advanced Profiling

### Kernel-Specific Analysis

Focus on specific OptiX programs:

```bash
# Profile only closest-hit programs
ncu --kernel-name-base regex:__closesthit__ \
    --set full -o closesthit_profile \
    ./build/bin/renderer data/scenes/mixed.yaml

# Profile intersection programs (where divergence is expected)
ncu --kernel-name-base regex:__intersection__ \
    --set full -o intersection_profile \
    ./build/bin/renderer data/scenes/mixed.yaml
```

### Memory Access Patterns

Neural asset evaluation may have different memory access patterns:

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
               l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
               l1tex__m_xbar2l1tex_read_bytes.sum \
    --set full -o memory_profile \
    ./build/bin/renderer data/scenes/mixed.yaml
```

### Source-Level Analysis

Map metrics to source code:

```bash
ncu --set full \
    --source-level analysis \
    --source-dir ./programs \
    -o source_profile \
    ./build/bin/renderer data/scenes/mixed.yaml
```

## Interpreting Divergence in Mixed Scenes

### Expected Patterns

**High Divergence Scenarios:**
1. **Traversal:** Rays hitting different geometry types
2. **Intersection:** AABB test (neural) vs triangle test
3. **Shading:** MLP evaluation vs simple material evaluation

**Low Divergence Scenarios:**
1. Pure triangle scenes
2. Pure neural scenes
3. Well-clustered geometry (spatial coherence)

### Metrics to Monitor

| Metric | Good | Moderate | Poor |
|--------|------|----------|------|
| Branch Efficiency | >90% | 70-90% | <70% |
| Warp Execution | >95% | 80-95% | <80% |
| Occupancy | >50% | 30-50% | <30% |

### Example Analysis

```bash
# Generate divergence report
ncu --set full \
    --metrics smsp__sass_average_branch_targets_threads_uniform.pct,\
              smsp__sass_branch_targets.avg,\
              sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    ./build/bin/renderer data/scenes/mixed.yaml > divergence.csv

# Parse results
python scripts/analyze_divergence.py divergence.csv
```

## Optimization Strategies

Based on profiling results, consider:

1. **Spatial Sorting:** Group similar geometry types
2. **Batched Inference:** Collect neural queries, run in batch
3. **Instance Culling:** Early rejection of neural assets
4. **Adaptive Sampling:** Reduce neural queries in low-contribution areas

## Troubleshooting

### Permission Denied

```bash
# Profiling requires elevated permissions
sudo ncu ./build/bin/renderer data/scenes/mixed.yaml

# Or set capabilities
sudo setcap cap_sys_admin+ep /usr/local/cuda/bin/ncu
```

### Too Many Kernels

```bash
# Limit profiling to specific kernel launches
ncu --launch-count 1 ./build/bin/renderer data/scenes/mixed.yaml

# Or skip first N launches
ncu --launch-skip 10 --launch-count 5 ./build/bin/renderer data/scenes/mixed.yaml
```

### Large Report Files

```bash
# Collect only specific sections
ncu --section SpeedOfLight --section WarpStateStats \
    -o compact_report ./build/bin/renderer data/scenes/mixed.yaml
```

## Automated Benchmarking

Create a benchmarking script:

```bash
#!/bin/bash
# benchmark.sh

SCENES=("triangles.yaml" "neural.yaml" "mixed.yaml")
METRICS="smsp__sass_average_branch_targets_threads_uniform.pct,sm__warps_active.avg.pct_of_peak_sustained_active"

for scene in "${SCENES[@]}"; do
    echo "Profiling $scene..."
    ncu --metrics $METRICS \
        --csv \
        ./build/bin/renderer "data/scenes/$scene" > "results_${scene%.yaml}.csv"
done

echo "Benchmarking complete!"
```

## References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Best Practices](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [OptiX Profiling Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#profiling)

## Next Steps

After profiling:
1. Identify hotspots and divergence patterns
2. Experiment with scene layouts
3. Implement optimization strategies
4. Re-profile to measure improvements
5. Document performance characteristics
