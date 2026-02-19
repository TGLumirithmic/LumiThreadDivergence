# Scene Generator for OptiX Neural Renderer

This directory contains tools to generate scene files with varying numbers of spheres for performance testing and scaling analysis.

## Overview

The scene generator creates pairs of YAML scene files (mesh and neural variants) with configurable numbers of non-overlapping spheres. This is useful for:

- Testing how performance scales with the number of objects
- Comparing mesh vs neural rendering performance
- Profiling warp divergence under different scene complexities

## Files

- `generate_scenes.py` - Python script to generate scene file pairs
- `generate_scene_batch.sh` - Bash script to generate multiple scenes at once
- `data/obj/floor.obj` - Generated floor geometry
- `data/obj/walls.obj` - Generated wall geometry

## Quick Start

### Generate a Single Scene

```bash
# Generate scenes with 16 spheres
python3 generate_scenes.py 16

# This creates:
# - scene_mesh_16.yaml (spheres as OBJ meshes)
# - scene_neural_16.yaml (spheres as neural assets)
# - data/obj/floor.obj (configurable floor)
# - data/obj/walls.obj (configurable walls)
```

### Generate Multiple Scenes

```bash
# Generate scenes with 1, 2, 4, 8, 16, 32, and 64 spheres
./generate_scene_batch.sh
```

### Render Generated Scenes

```bash
# Render mesh variant
./build/bin/render scene_mesh_16.yaml output/mesh_16.ppm

# Render neural variant
./build/bin/render scene_neural_16.yaml output/neural_16.ppm
```

## Usage

```
python3 generate_scenes.py NUM_SPHERES [OPTIONS]

Required Arguments:
  NUM_SPHERES           Number of spheres to generate

Optional Arguments:
  --output-dir DIR      Output directory (default: current directory)
  --prefix PREFIX       Prefix for scene files (default: "scene")
  --arena-size SIZE     Arena size in world units (default: 10.0)
  --wall-height HEIGHT  Wall height in world units (default: 5.0)
  --min-radius RADIUS   Minimum sphere radius (default: 0.3)
  --max-radius RADIUS   Maximum sphere radius (default: 1.5)
  --min-height HEIGHT   Minimum sphere center height (default: 0.0)
  --max-height HEIGHT   Maximum sphere center height (default: 3.0)
  --seed SEED           Random seed for reproducibility
  --sphere-obj PATH     Path to sphere OBJ (default: data/obj/sphere.obj)
  --weights PATH        Path to neural weights (default: data/models/weights.bin)
```

## Examples

### Generate Dense Scene

```bash
# Many small spheres in a large arena
python3 generate_scenes.py 100 \
    --arena-size 20.0 \
    --min-radius 0.2 \
    --max-radius 0.5 \
    --seed 123
```

### Generate Sparse Scene

```bash
# Few large spheres
python3 generate_scenes.py 5 \
    --arena-size 15.0 \
    --min-radius 1.0 \
    --max-radius 2.0 \
    --seed 456
```

### Generate Reproducible Scenes

```bash
# Using a seed ensures identical sphere placement
python3 generate_scenes.py 20 --seed 42 --prefix reproducible
```

## Features

### Non-Overlapping AABBs

The generator uses spatial rejection sampling to ensure that no sphere AABBs overlap. This provides realistic scene complexity without degenerate geometry.

### Configurable Arena

The floor and walls are generated as OBJ files with configurable dimensions. The arena automatically scales to accommodate the requested number of spheres.

### Camera Positioning

The camera is automatically positioned to view the entire scene, with the distance and angle calculated based on the arena size.

### Paired Generation

Each invocation generates both mesh and neural variants of the same scene, ensuring fair performance comparisons.

## Scene File Format

Generated YAML files follow this structure:

```yaml
scene:
  camera:
    position: [x, y, z]
    look_at: [x, y, z]
    fov: 90

  light:
    type: point
    position: [x, y, z]
    color: [r, g, b]
    intensity: 100.0

  objects:
    - type: mesh
      file: data/obj/floor.obj

    - type: mesh
      file: data/obj/walls.obj

    - type: mesh  # or neural_asset
      file: data/obj/sphere.obj  # or weights: path/to/weights.bin
      has_bounds: true
      bounds:
        min: [-1.0, -1.0, -1.0]
        max: [1.0, 1.0, 1.0]
      transform:
        position: [x, y, z]
        scale: [sx, sy, sz]
```

## Performance Testing Workflow

1. **Generate scenes with varying complexity:**
   ```bash
   ./generate_scene_batch.sh
   ```

2. **Render each scene:**
   ```bash
   for n in 1 2 4 8 16 32 64; do
       ./build/bin/render scene_mesh_$n.yaml output/mesh_$n.ppm
       ./build/bin/render scene_neural_$n.yaml output/neural_$n.ppm
   done
   ```

3. **Analyze divergence metrics:**
   ```bash
   # Divergence data is written to *_divergence.bin files
   python3 analyze_divergence.py output/mesh_*_divergence.bin
   python3 analyze_divergence.py output/neural_*_divergence.bin
   ```

## Troubleshooting

### "Could only place N out of M spheres"

The arena is too small or the spheres are too large. Solutions:
- Increase `--arena-size`
- Decrease `--max-radius`
- Reduce the number of spheres

### Missing OBJ files

Ensure `data/obj/sphere.obj` exists. The floor and walls OBJ files are automatically generated.

### Scene appears empty

Check that:
- The sphere OBJ file path is correct
- For neural scenes, the weights file exists
- The camera position can see the spheres (check YAML camera settings)

## Implementation Details

### Sphere Placement Algorithm

1. Generate random position and radius within constraints
2. Compute transformed AABB for the sphere
3. Check for overlaps with all previously placed spheres
4. If no overlap, place sphere; otherwise retry
5. After max attempts, warn and return partial placement

### Transform Calculation

Spheres use a unit sphere (radius 1) and transform it via:
- `scale`: Uniform scaling to desired radius
- `position`: Translation to desired location
- AABB: Transformed bounds are `scale * unit_bounds + position`

### Floor and Walls Generation

- **Floor**: Horizontal quad at Y=-1, extends from (-size/2, -size/2) to (+size/2, +size/2)
- **Walls**: Vertical quad at Z=-size/2, extends from Y=-1 to Y=wall_height

The existing C++ code has hardcoded floor/walls, but will load OBJ files if the path ends with `.obj`, so our generated files work seamlessly.
