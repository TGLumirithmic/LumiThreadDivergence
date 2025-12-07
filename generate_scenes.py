#!/usr/bin/env python3
"""
Scene Generator for OptiX Neural Renderer

Generates pairs of scene files (mesh and neural variants) with varying numbers of spheres.
Spheres are placed with non-overlapping AABBs at different positions and sizes.
"""

import argparse
import math
import random
import yaml
import os
from typing import List, Tuple, Dict, Any
import numpy as np


def aabb_overlap(min1: List[float], max1: List[float],
                 min2: List[float], max2: List[float]) -> bool:
    """Check if two AABBs overlap."""
    return (min1[0] <= max2[0] and max1[0] >= min2[0] and
            min1[1] <= max2[1] and max1[1] >= min2[1] and
            min1[2] <= max2[2] and max1[2] >= min2[2])


def transform_aabb(bounds_min: List[float], bounds_max: List[float],
                   position: List[float], scale: List[float]) -> Tuple[List[float], List[float]]:
    """Transform an AABB by position and scale."""
    # Apply scale
    scaled_min = [bounds_min[i] * scale[i] for i in range(3)]
    scaled_max = [bounds_max[i] * scale[i] for i in range(3)]

    # Apply translation
    transformed_min = [scaled_min[i] + position[i] for i in range(3)]
    transformed_max = [scaled_max[i] + position[i] for i in range(3)]

    return transformed_min, transformed_max


def generate_sphere_configs_random(num_spheres: int,
                                    arena_size: float = 10.0,
                                    min_radius: float = 0.3,
                                    max_radius: float = 1.5,
                                    min_height: float = 0.0,
                                    max_height: float = 3.0,
                                    seed: int = None,
                                    max_attempts: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate non-overlapping sphere configurations using random placement.
    (Legacy method - tends to clump spheres together)

    Args:
        num_spheres: Number of spheres to generate
        arena_size: Size of the arena (floor will be arena_size x arena_size)
        min_radius: Minimum sphere radius
        max_radius: Maximum sphere radius
        min_height: Minimum Y position for sphere centers
        max_height: Maximum Y position for sphere centers
        seed: Random seed for reproducibility
        max_attempts: Maximum attempts to place each sphere

    Returns:
        List of sphere configurations with position, scale, and bounds
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    spheres = []
    placed_aabbs = []

    # Unit sphere bounds
    unit_min = [-1.0, -1.0, -1.0]
    unit_max = [1.0, 1.0, 1.0]

    for i in range(num_spheres):
        placed = False

        for attempt in range(max_attempts):
            # Generate random radius and position
            radius = random.uniform(min_radius, max_radius)

            # Position within arena, accounting for radius
            margin = arena_size / 2.0 - radius
            x = random.uniform(-margin, margin)
            z = random.uniform(-margin, margin)

            # Y position: center of sphere should be at least radius above min_height
            y = random.uniform(min_height + radius, max_height)

            position = [x, y, z]
            scale = [radius, radius, radius]

            # Compute transformed AABB
            aabb_min, aabb_max = transform_aabb(unit_min, unit_max, position, scale)

            # Check for overlaps with existing spheres
            overlaps = False
            for existing_min, existing_max in placed_aabbs:
                if aabb_overlap(aabb_min, aabb_max, existing_min, existing_max):
                    overlaps = True
                    break

            if not overlaps:
                spheres.append({
                    'position': position,
                    'scale': scale,
                    'bounds_min': unit_min.copy(),
                    'bounds_max': unit_max.copy()
                })
                placed_aabbs.append((aabb_min, aabb_max))
                placed = True
                break

        if not placed:
            print(f"Warning: Could only place {i} out of {num_spheres} spheres after {max_attempts} attempts")
            break

    return spheres


def generate_sphere_configs_grid(num_spheres: int,
                                  arena_size: float = 10.0,
                                  min_radius: float = 0.3,
                                  max_radius: float = 1.5,
                                  min_height: float = 0.0,
                                  max_height: float = 3.0,
                                  seed: int = None) -> List[Dict[str, Any]]:
    """
    Generate non-overlapping sphere configurations using grid-based placement.
    Spreads spheres evenly across X, Y, Z dimensions with random jitter.

    Args:
        num_spheres: Number of spheres to generate
        arena_size: Size of the arena (floor will be arena_size x arena_size)
        min_radius: Minimum sphere radius
        max_radius: Maximum sphere radius
        min_height: Minimum Y position for sphere centers
        max_height: Maximum Y position for sphere centers
        seed: Random seed for reproducibility

    Returns:
        List of sphere configurations with position, scale, and bounds
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate grid dimensions
    # Use cube root to get cells per dimension, ensuring we have enough cells
    cells_per_dim = math.ceil(num_spheres ** (1/3))

    # Cell sizes
    cell_size_xz = arena_size / cells_per_dim
    height_range = max_height - min_height
    cell_size_y = height_range / max(1, cells_per_dim)

    # Generate all cell indices (i, j, k) for X, Y, Z
    cells = [(i, j, k) for i in range(cells_per_dim)
                       for j in range(cells_per_dim)
                       for k in range(cells_per_dim)]

    # Shuffle for random selection order
    random.shuffle(cells)

    spheres = []
    placed_aabbs = []

    # Unit sphere bounds
    unit_min = [-1.0, -1.0, -1.0]
    unit_max = [1.0, 1.0, 1.0]

    for idx in range(min(num_spheres, len(cells))):
        i, j, k = cells[idx]

        # Cell center position
        cx = -arena_size / 2.0 + (i + 0.5) * cell_size_xz
        cy = min_height + (j + 0.5) * cell_size_y
        cz = -arena_size / 2.0 + (k + 0.5) * cell_size_xz

        # Random radius (constrained by cell size to avoid overflow)
        max_possible_radius = min(cell_size_xz, cell_size_y) / 2.0 * 0.9  # 90% of half-cell
        effective_max_radius = min(max_radius, max_possible_radius)
        effective_min_radius = min(min_radius, effective_max_radius)
        radius = random.uniform(effective_min_radius, effective_max_radius)

        # Jitter within cell (constrained by radius to stay in cell)
        max_jitter_xz = max(0, (cell_size_xz / 2.0) - radius)
        max_jitter_y = max(0, (cell_size_y / 2.0) - radius)

        jitter_x = random.uniform(-max_jitter_xz, max_jitter_xz)
        jitter_y = random.uniform(-max_jitter_y, max_jitter_y)
        jitter_z = random.uniform(-max_jitter_xz, max_jitter_xz)

        # Final position
        x = cx + jitter_x
        y = cy + jitter_y
        z = cz + jitter_z

        # Clamp Y to valid range (sphere center must allow full sphere above floor)
        y = max(min_height + radius, min(y, max_height))

        # Clamp X, Z to arena bounds
        half_arena = arena_size / 2.0
        x = max(-half_arena + radius, min(x, half_arena - radius))
        z = max(-half_arena + radius, min(z, half_arena - radius))

        position = [x, y, z]
        scale = [radius, radius, radius]

        # Compute transformed AABB
        aabb_min, aabb_max = transform_aabb(unit_min, unit_max, position, scale)

        # Safety check for overlaps (shouldn't happen with proper grid sizing)
        overlaps = False
        for existing_min, existing_max in placed_aabbs:
            if aabb_overlap(aabb_min, aabb_max, existing_min, existing_max):
                overlaps = True
                break

        if not overlaps:
            spheres.append({
                'position': position,
                'scale': scale,
                'bounds_min': unit_min.copy(),
                'bounds_max': unit_max.copy()
            })
            placed_aabbs.append((aabb_min, aabb_max))

    if len(spheres) < num_spheres:
        print(f"Warning: Could only place {len(spheres)} out of {num_spheres} spheres in grid")

    return spheres


def generate_sphere_configs(num_spheres: int,
                            arena_size: float = 10.0,
                            min_radius: float = 0.3,
                            max_radius: float = 1.5,
                            min_height: float = 0.0,
                            max_height: float = 3.0,
                            seed: int = None,
                            spread_mode: str = 'grid',
                            max_attempts: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate non-overlapping sphere configurations.

    Args:
        num_spheres: Number of spheres to generate
        arena_size: Size of the arena (floor will be arena_size x arena_size)
        min_radius: Minimum sphere radius
        max_radius: Maximum sphere radius
        min_height: Minimum Y position for sphere centers
        max_height: Maximum Y position for sphere centers
        seed: Random seed for reproducibility
        spread_mode: 'grid' for even distribution, 'random' for legacy behavior
        max_attempts: Maximum attempts to place each sphere (random mode only)

    Returns:
        List of sphere configurations with position, scale, and bounds
    """
    if spread_mode == 'grid':
        return generate_sphere_configs_grid(
            num_spheres, arena_size, min_radius, max_radius,
            min_height, max_height, seed
        )
    else:
        return generate_sphere_configs_random(
            num_spheres, arena_size, min_radius, max_radius,
            min_height, max_height, seed, max_attempts
        )


def compute_scene_aabb(sphere_configs: List[Dict[str, Any]],
                       arena_size: float = 10.0,
                       wall_height: float = 5.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute axis-aligned bounding box of the entire scene (arena + spheres).

    Args:
        sphere_configs: List of sphere dicts with 'position' and 'scale'
        arena_size: Size of the arena (floor extent)
        wall_height: Height of the back wall

    Returns:
        tuple: (aabb_min, aabb_max, centroid) where each is [x, y, z]
    """
    # Arena bounds (floor + wall)
    half_arena = arena_size / 2.0
    arena_min = [-half_arena, -1.0, -half_arena]
    arena_max = [half_arena, wall_height, half_arena]

    if not sphere_configs:
        # Just arena bounds
        centroid = [
            (arena_min[i] + arena_max[i]) / 2.0 for i in range(3)
        ]
        return arena_min, arena_max, centroid

    # Compute sphere bounds
    positions = [s['position'] for s in sphere_configs]
    radii = [s['scale'][0] for s in sphere_configs]  # Uniform scale

    xs = [(p[0] - r, p[0] + r) for p, r in zip(positions, radii)]
    ys = [(p[1] - r, p[1] + r) for p, r in zip(positions, radii)]
    zs = [(p[2] - r, p[2] + r) for p, r in zip(positions, radii)]

    sphere_min = [min(x[0] for x in xs), min(y[0] for y in ys), min(z[0] for z in zs)]
    sphere_max = [max(x[1] for x in xs), max(y[1] for y in ys), max(z[1] for z in zs)]

    # Union of arena and sphere bounds
    aabb_min = [min(arena_min[i], sphere_min[i]) for i in range(3)]
    aabb_max = [max(arena_max[i], sphere_max[i]) for i in range(3)]

    # Centroid of combined AABB
    centroid = [
        (aabb_min[i] + aabb_max[i]) / 2.0 for i in range(3)
    ]

    return aabb_min, aabb_max, centroid


def calculate_camera_distance_from_fov(extent_width: float, extent_height: float,
                                       fov_degrees: float, aspect_ratio: float = 1.0,
                                       padding: float = 1.2) -> float:
    """
    Calculate camera distance needed to fit a rectangular extent in view.

    Args:
        extent_width: Scene width (max of X and Z extent)
        extent_height: Scene height (Y extent)
        fov_degrees: Vertical field of view in degrees
        aspect_ratio: width/height of output image
        padding: Safety margin (1.2 = 20% padding)

    Returns:
        float: Required camera distance
    """
    fov_rad = math.radians(fov_degrees)

    # Vertical FOV constraint
    dist_for_height = extent_height / (2 * math.tan(fov_rad / 2))

    # Horizontal FOV constraint
    dist_for_width = extent_width / (2 * aspect_ratio * math.tan(fov_rad / 2))

    # Take maximum (more restrictive), apply padding
    min_distance = max(dist_for_height, dist_for_width)
    return min_distance * padding


def calculate_optimal_camera(sphere_configs: List[Dict[str, Any]], arena_size: float = 10.0,
                            wall_height: float = 5.0,
                            fov: float = 90, aspect_ratio: float = 1.0, padding: float = 1.2,
                            camera_angle_deg: float = 35.0, azimuth_deg: float = 30.0) -> Tuple[List[float], List[float]]:
    """
    Calculate optimal camera position and look-at point for a scene.

    Args:
        sphere_configs: List of sphere configuration dicts
        arena_size: Size of the arena
        wall_height: Height of the walls
        fov: Vertical field of view in degrees
        aspect_ratio: Image width/height
        padding: Padding factor to avoid edge clipping
        camera_angle_deg: Elevation angle above horizon
        azimuth_deg: Rotation around vertical axis from +Z

    Returns:
        tuple: (camera_position, look_at_position) as [x, y, z] lists
    """
    # Compute scene bounds (union of arena and spheres)
    aabb_min, aabb_max, centroid = compute_scene_aabb(sphere_configs, arena_size, wall_height)

    # Scene extent
    extent = [aabb_max[i] - aabb_min[i] for i in range(3)]
    extent_width = max(extent[0], extent[2])  # Max of X and Z
    extent_height = extent[1]

    # Calculate required distance
    distance = calculate_camera_distance_from_fov(
        extent_width, extent_height, fov, aspect_ratio, padding
    )

    # Ensure minimum distance
    min_distance = max(extent) * 0.5
    distance = max(distance, min_distance)

    # Position camera using spherical coordinates
    camera_angle_rad = math.radians(camera_angle_deg)
    azimuth_rad = math.radians(azimuth_deg)

    cam_x = centroid[0] + distance * math.cos(camera_angle_rad) * math.sin(azimuth_rad)
    cam_y = centroid[1] + distance * math.sin(camera_angle_rad)
    cam_z = centroid[2] + distance * math.cos(camera_angle_rad) * math.cos(azimuth_rad)

    camera_position = [cam_x, cam_y, cam_z]
    look_at_position = centroid

    return camera_position, look_at_position


def write_floor_obj(filepath: str, size: float):
    """Write a floor quad as OBJ file."""
    half_size = size / 2.0
    y = -1.0

    with open(filepath, 'w') as f:
        f.write("# Floor quad\n")
        f.write(f"v {-half_size} {y} {-half_size}\n")
        f.write(f"v {half_size} {y} {-half_size}\n")
        f.write(f"v {half_size} {y} {half_size}\n")
        f.write(f"v {-half_size} {y} {half_size}\n")

        f.write("vn 0.0 1.0 0.0\n")
        f.write("vn 0.0 1.0 0.0\n")
        f.write("vn 0.0 1.0 0.0\n")
        f.write("vn 0.0 1.0 0.0\n")

        f.write("f 1//1 2//2 3//3\n")
        f.write("f 1//1 3//3 4//4\n")


def write_walls_obj(filepath: str, size: float, height: float):
    """Write back wall as OBJ file."""
    half_size = size / 2.0
    y_bottom = -1.0
    y_top = height
    z = -size / 2.0

    with open(filepath, 'w') as f:
        f.write("# Back wall\n")
        f.write(f"v {-half_size} {y_bottom} {z}\n")
        f.write(f"v {half_size} {y_bottom} {z}\n")
        f.write(f"v {half_size} {y_top} {z}\n")
        f.write(f"v {-half_size} {y_top} {z}\n")

        f.write("vn 0.0 0.0 1.0\n")
        f.write("vn 0.0 0.0 1.0\n")
        f.write("vn 0.0 0.0 1.0\n")
        f.write("vn 0.0 0.0 1.0\n")

        f.write("f 1//1 2//2 3//3\n")
        f.write("f 1//1 3//3 4//4\n")


def generate_scene_yaml(output_path: str,
                        spheres: List[Dict[str, Any]],
                        arena_size: float,
                        wall_height: float,
                        sphere_type: str,
                        sphere_file: str,
                        camera_distance: float = None,
                        fov: float = 90,
                        aspect_ratio: float = 1.0,
                        camera_padding: float = 1.2,
                        use_optimal_camera: bool = True,
                        camera_angle: float = 35.0,
                        camera_azimuth: float = 30.0):
    """
    Generate a scene YAML file.

    Args:
        output_path: Path to write YAML file
        spheres: List of sphere configurations
        arena_size: Size of the arena
        wall_height: Height of the walls
        sphere_type: Either 'mesh' or 'neural_asset'
        sphere_file: For mesh: path to OBJ file, for neural: path to weights
        camera_distance: Distance of camera from origin (deprecated, for backward compat)
        fov: Camera field of view in degrees (default: 90)
        aspect_ratio: Image width/height (default: 1.0)
        camera_padding: Padding factor for camera distance (default: 1.2)
        use_optimal_camera: Use optimal camera positioning (default: True)
        camera_angle: Camera elevation angle in degrees (default: 35)
        camera_azimuth: Camera azimuth angle in degrees (default: 30)
    """
    # Calculate camera position
    if use_optimal_camera:
        cam_pos, look_at = calculate_optimal_camera(
            spheres, arena_size, wall_height, fov, aspect_ratio, camera_padding,
            camera_angle, camera_azimuth
        )
    else:
        # Fallback to current behavior
        if camera_distance is None:
            camera_distance = arena_size * 0.75
        cam_height = arena_size * 0.4
        cam_pos = [camera_distance * 0.6, cam_height, camera_distance]
        look_at = [0.0, arena_size * 0.2, 0.0]

    # Light positioned above the scene
    light_height = max(wall_height, arena_size * 0.5)

    scene = {
        'scene': {
            'camera': {
                'position': cam_pos,
                'look_at': look_at,
                'fov': 90
            },
            'light': {
                'type': 'point',
                'position': [0.0, light_height, arena_size * 0.3],
                'color': [1.0, 1.0, 1.0],
                'intensity': 100.0
            },
            'objects': []
        }
    }

    # Add floor
    scene['scene']['objects'].append({
        'type': 'mesh',
        'file': 'data/obj/floor.obj'
    })

    # Add walls
    scene['scene']['objects'].append({
        'type': 'mesh',
        'file': 'data/obj/walls.obj'
    })

    # Add spheres
    for sphere in spheres:
        sphere_obj = {
            'type': sphere_type,
            'has_bounds': True,
            'bounds': {
                'min': sphere['bounds_min'],
                'max': sphere['bounds_max']
            },
            'transform': {
                'position': sphere['position'],
                'scale': sphere['scale']
            }
        }

        if sphere_type == 'mesh':
            sphere_obj['file'] = sphere_file
        else:  # neural_asset
            sphere_obj['weights'] = sphere_file

        scene['scene']['objects'].append(sphere_obj)

    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(scene, f, default_flow_style=None, sort_keys=False)

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate scene files with varying numbers of spheres'
    )
    parser.add_argument('num_spheres', type=int,
                        help='Number of spheres to generate')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for scene files (default: current directory)')
    parser.add_argument('--prefix', type=str, default='scene',
                        help='Prefix for output files (default: scene)')
    parser.add_argument('--arena-size', type=float, default=10.0,
                        help='Size of the arena (default: 10.0)')
    parser.add_argument('--wall-height', type=float, default=5.0,
                        help='Height of the walls (default: 5.0)')
    parser.add_argument('--min-radius', type=float, default=0.3,
                        help='Minimum sphere radius (default: 0.3)')
    parser.add_argument('--max-radius', type=float, default=1.5,
                        help='Maximum sphere radius (default: 1.5)')
    parser.add_argument('--min-height', type=float, default=0.0,
                        help='Minimum sphere center height (default: 0.0)')
    parser.add_argument('--max-height', type=float, default=3.0,
                        help='Maximum sphere center height (default: 3.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--sphere-obj', type=str, default='data/obj/sphere.obj',
                        help='Path to sphere OBJ file (default: data/obj/sphere.obj)')
    parser.add_argument('--weights', type=str, default='data/models/weights.bin',
                        help='Path to neural weights file (default: data/models/weights.bin)')
    parser.add_argument('--camera-padding', type=float, default=1.2,
                        help='Camera distance padding factor (default: 1.2)')
    parser.add_argument('--no-optimal-camera', action='store_true',
                        help='Use legacy camera positioning instead of optimal')
    parser.add_argument('--camera-angle', type=float, default=35.0,
                        help='Camera elevation angle in degrees (default: 35)')
    parser.add_argument('--camera-azimuth', type=float, default=30.0,
                        help='Camera azimuth angle in degrees (default: 30)')
    parser.add_argument('--spread-mode', type=str, default='grid', choices=['grid', 'random'],
                        help='Sphere placement mode: grid (spread out) or random (legacy) (default: grid)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data', 'obj'), exist_ok=True)

    # Generate sphere configurations
    print(f"Generating {args.num_spheres} spheres with non-overlapping AABBs (mode: {args.spread_mode})...")
    spheres = generate_sphere_configs(
        num_spheres=args.num_spheres,
        arena_size=args.arena_size,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_height=args.min_height,
        max_height=args.max_height,
        seed=args.seed,
        spread_mode=args.spread_mode
    )

    if len(spheres) < args.num_spheres:
        print(f"Warning: Only generated {len(spheres)} spheres (requested {args.num_spheres})")
        print("Consider increasing --arena-size or decreasing --max-radius")

    # Write floor and walls OBJ files
    floor_path = os.path.join(args.output_dir, 'data', 'obj', 'floor.obj')
    walls_path = os.path.join(args.output_dir, 'data', 'obj', 'walls.obj')

    print(f"Writing floor to {floor_path}")
    write_floor_obj(floor_path, args.arena_size)

    print(f"Writing walls to {walls_path}")
    write_walls_obj(walls_path, args.arena_size, args.wall_height)

    # Generate mesh scene
    mesh_scene_path = os.path.join(args.output_dir, f'{args.prefix}_mesh_{args.num_spheres}.yaml')
    generate_scene_yaml(
        mesh_scene_path,
        spheres,
        args.arena_size,
        args.wall_height,
        sphere_type='mesh',
        sphere_file=args.sphere_obj,
        camera_padding=args.camera_padding,
        use_optimal_camera=not args.no_optimal_camera,
        camera_angle=args.camera_angle,
        camera_azimuth=args.camera_azimuth
    )

    # Generate neural scene
    neural_scene_path = os.path.join(args.output_dir, f'{args.prefix}_neural_{args.num_spheres}.yaml')
    generate_scene_yaml(
        neural_scene_path,
        spheres,
        args.arena_size,
        args.wall_height,
        sphere_type='neural_asset',
        sphere_file=args.weights,
        camera_padding=args.camera_padding,
        use_optimal_camera=not args.no_optimal_camera,
        camera_angle=args.camera_angle,
        camera_azimuth=args.camera_azimuth
    )

    print(f"\nSuccessfully generated scene pair with {len(spheres)} spheres:")
    print(f"  Mesh scene:   {mesh_scene_path}")
    print(f"  Neural scene: {neural_scene_path}")
    print(f"  Floor:        {floor_path}")
    print(f"  Walls:        {walls_path}")

    # Print some statistics
    if spheres:
        radii = [s['scale'][0] for s in spheres]
        heights = [s['position'][1] for s in spheres]
        print(f"\nSphere statistics:")
        print(f"  Radius range: {min(radii):.2f} - {max(radii):.2f}")
        print(f"  Height range: {min(heights):.2f} - {max(heights):.2f}")
        print(f"  Arena size:   {args.arena_size}x{args.arena_size}")
        print(f"  Wall height:  {args.wall_height}")


def test_camera_calculations():
    """Test camera calculation functions."""
    print("Running camera calculation tests...")

    # Test 1: Empty scene - uses arena bounds only
    # Arena: arena_size=10, wall_height=5
    # AABB: min=[-5, -1, -5], max=[5, 5, 5]
    # Centroid: [0, 2, 0]
    cam_pos, look_at = calculate_optimal_camera([], arena_size=10.0, wall_height=5.0)
    assert abs(look_at[0]) < 0.01, f"Empty scene look_at X should be 0: got {look_at[0]}"
    assert abs(look_at[1] - 2.0) < 0.01, f"Empty scene look_at Y should be 2: got {look_at[1]}"
    assert abs(look_at[2]) < 0.01, f"Empty scene look_at Z should be 0: got {look_at[2]}"
    print("  ✓ Empty scene test passed")

    # Test 2: compute_scene_aabb with arena only
    aabb_min, aabb_max, centroid = compute_scene_aabb([], arena_size=10.0, wall_height=5.0)
    assert aabb_min == [-5.0, -1.0, -5.0], f"Arena-only AABB min wrong: got {aabb_min}"
    assert aabb_max == [5.0, 5.0, 5.0], f"Arena-only AABB max wrong: got {aabb_max}"
    assert centroid == [0.0, 2.0, 0.0], f"Arena-only centroid wrong: got {centroid}"
    print("  ✓ Arena-only AABB test passed")

    # Test 3: compute_scene_aabb with spheres inside arena
    spheres = [
        {'position': [-2.0, 1.0, 0.0], 'scale': [0.5, 0.5, 0.5]},
        {'position': [2.0, 1.0, 0.0], 'scale': [0.5, 0.5, 0.5]},
    ]
    # Sphere bounds: x=[-2.5, 2.5], y=[0.5, 1.5], z=[-0.5, 0.5]
    # Arena bounds: x=[-5, 5], y=[-1, 5], z=[-5, 5]
    # Union: x=[-5, 5], y=[-1, 5], z=[-5, 5] (arena dominates)
    aabb_min, aabb_max, centroid = compute_scene_aabb(spheres, arena_size=10.0, wall_height=5.0)
    assert aabb_min == [-5.0, -1.0, -5.0], f"Union AABB min wrong: got {aabb_min}"
    assert aabb_max == [5.0, 5.0, 5.0], f"Union AABB max wrong: got {aabb_max}"
    assert centroid == [0.0, 2.0, 0.0], f"Union centroid wrong: got {centroid}"
    print("  ✓ Spheres-inside-arena AABB test passed")

    # Test 4: compute_scene_aabb with spheres outside arena
    spheres = [
        {'position': [0.0, 8.0, 0.0], 'scale': [1.0, 1.0, 1.0]},  # Above wall_height
    ]
    # Sphere bounds: x=[-1, 1], y=[7, 9], z=[-1, 1]
    # Arena bounds: x=[-5, 5], y=[-1, 5], z=[-5, 5]
    # Union: x=[-5, 5], y=[-1, 9], z=[-5, 5]
    aabb_min, aabb_max, centroid = compute_scene_aabb(spheres, arena_size=10.0, wall_height=5.0)
    assert aabb_max[1] == 9.0, f"Union should extend to sphere top: got {aabb_max[1]}"
    expected_centroid_y = (-1.0 + 9.0) / 2.0  # = 4.0
    assert abs(centroid[1] - expected_centroid_y) < 0.01, f"Centroid Y wrong: got {centroid[1]}"
    print("  ✓ Spheres-outside-arena AABB test passed")

    # Test 5: calculate_camera_distance_from_fov
    dist = calculate_camera_distance_from_fov(
        extent_width=4.0, extent_height=2.0, fov_degrees=90,
        aspect_ratio=1.0, padding=1.0
    )
    # For FOV=90°, to fit height=2: distance = 2/(2*tan(45°)) = 2/2 = 1.0
    # For width=4: distance = 4/(2*1*tan(45°)) = 4/2 = 2.0
    # Should take max(1.0, 2.0) * 1.0 = 2.0
    expected_dist = 2.0
    assert abs(dist - expected_dist) < 0.01, f"Camera distance wrong: expected {expected_dist}, got {dist}"
    print("  ✓ Camera distance calculation test passed")

    print("\n✓ All camera calculation tests passed!")


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        test_camera_calculations()
        sys.exit(0)
    main()
