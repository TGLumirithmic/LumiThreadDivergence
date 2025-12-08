#!/usr/bin/env python3
"""
Compare hit positions between mesh and neural rendering methods.
Generates an interactive 3D HTML visualization using Plotly.

Usage:
    python scripts/compare_hit_positions.py output/mesh_test output/neural_test

This will look for:
    - output/mesh_test_hit_position.bin
    - output/mesh_test_instance_id.bin
    - output/neural_test_hit_position.bin
    - output/neural_test_instance_id.bin

And generate:
    - output/hit_position_comparison.html
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Try to import plotly, install if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


def load_hit_positions(filepath: str, width: int = 512, height: int = 512) -> np.ndarray:
    """Load hit positions from binary file.

    Format: width*height*3 floats (12 bytes per pixel), no header.
    Returns: (height, width, 3) array of XYZ world coordinates.
    """
    data = np.fromfile(filepath, dtype=np.float32)
    expected_size = width * height * 3
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} floats, got {len(data)}")
    return data.reshape((height, width, 3))


def load_instance_ids(filepath: str, width: int = 512, height: int = 512) -> np.ndarray:
    """Load instance IDs from binary file.

    Format: width*height int32s (4 bytes per pixel), no header.
    Returns: (height, width) array of instance IDs (-1 for miss).
    """
    data = np.fromfile(filepath, dtype=np.int32)
    expected_size = width * height
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} int32s, got {len(data)}")
    return data.reshape((height, width))


def subsample_points(positions: np.ndarray, instance_ids: np.ndarray,
                     max_points: int = 20000, filter_miss: bool = True) -> tuple:
    """Subsample point cloud for interactive visualization.

    Args:
        positions: (H, W, 3) array of hit positions
        instance_ids: (H, W) array of instance IDs
        max_points: Maximum number of points to keep
        filter_miss: If True, exclude rays that missed (instance_id == -1)

    Returns:
        (xyz, ids) - subsampled positions and instance IDs
    """
    # Flatten
    pos_flat = positions.reshape(-1, 3)
    ids_flat = instance_ids.flatten()

    # Filter miss rays
    if filter_miss:
        mask = ids_flat >= 0
        pos_flat = pos_flat[mask]
        ids_flat = ids_flat[mask]

    # Subsample if needed
    n_points = len(pos_flat)
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        pos_flat = pos_flat[indices]
        ids_flat = ids_flat[indices]

    return pos_flat, ids_flat


def create_comparison_html(mesh_prefix: str, neural_prefix: str, output_path: str,
                           width: int = 512, height: int = 512, max_points: int = 20000):
    """Create interactive 3D comparison visualization."""

    # Load data
    print(f"Loading mesh data from {mesh_prefix}...")
    mesh_pos = load_hit_positions(f"{mesh_prefix}_hit_position.bin", width, height)
    mesh_ids = load_instance_ids(f"{mesh_prefix}_instance_id.bin", width, height)

    print(f"Loading neural data from {neural_prefix}...")
    neural_pos = load_hit_positions(f"{neural_prefix}_hit_position.bin", width, height)
    neural_ids = load_instance_ids(f"{neural_prefix}_instance_id.bin", width, height)

    # Subsample for visualization
    print(f"Subsampling to {max_points} points...")
    mesh_xyz, mesh_id = subsample_points(mesh_pos, mesh_ids, max_points)
    neural_xyz, neural_id = subsample_points(neural_pos, neural_ids, max_points)

    print(f"Mesh points: {len(mesh_xyz)}, Neural points: {len(neural_xyz)}")

    # Get unique instance IDs for color mapping
    all_ids = np.unique(np.concatenate([mesh_id, neural_id]))
    n_instances = len(all_ids)
    print(f"Unique instances: {n_instances}")

    # Create color scale
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # Create figure with two 3D subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Mesh Rendering', 'Neural Rendering'],
        horizontal_spacing=0.05
    )

    # Add mesh points (colored by instance ID)
    for i, inst_id in enumerate(all_ids):
        mask = mesh_id == inst_id
        if mask.sum() > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=mesh_xyz[mask, 0],
                    y=mesh_xyz[mask, 1],
                    z=mesh_xyz[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    name=f'Instance {inst_id}',
                    legendgroup=f'inst_{inst_id}',
                    showlegend=True
                ),
                row=1, col=1
            )

    # Add neural points (colored by instance ID)
    for i, inst_id in enumerate(all_ids):
        mask = neural_id == inst_id
        if mask.sum() > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=neural_xyz[mask, 0],
                    y=neural_xyz[mask, 1],
                    z=neural_xyz[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    name=f'Instance {inst_id}',
                    legendgroup=f'inst_{inst_id}',
                    showlegend=False  # Already in legend from mesh
                ),
                row=1, col=2
            )

    # Compute axis ranges to sync both plots
    all_xyz = np.vstack([mesh_xyz, neural_xyz])
    x_range = [all_xyz[:, 0].min() - 0.5, all_xyz[:, 0].max() + 0.5]
    y_range = [all_xyz[:, 1].min() - 0.5, all_xyz[:, 1].max() + 0.5]
    z_range = [all_xyz[:, 2].min() - 0.5, all_xyz[:, 2].max() + 0.5]

    # Update layout
    fig.update_layout(
        title=dict(
            text='Hit Position Comparison: Mesh vs Neural',
            x=0.5,
            font=dict(size=20)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        height=700,
        width=1400,
        margin=dict(l=0, r=150, t=50, b=0)
    )

    # Sync camera and axes for both subplots
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5)
    )

    scene_config = dict(
        xaxis=dict(range=x_range, title='X'),
        yaxis=dict(range=y_range, title='Y'),
        zaxis=dict(range=z_range, title='Z'),
        camera=camera,
        aspectmode='cube'
    )

    fig.update_layout(
        scene=scene_config,
        scene2=scene_config
    )

    # Write to HTML
    print(f"Writing HTML to {output_path}...")
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    # Inject JavaScript to sync camera between the two 3D subplots
    camera_sync_js = """
<script>
// Wait for the plot to be fully rendered before attaching event listener
function setupCameraSync() {
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd || !gd.on) {
        // Plot not ready yet, try again
        setTimeout(setupCameraSync, 100);
        return;
    }

    var syncing = false;

    gd.on('plotly_relayout', function(eventData) {
        if (syncing) return;
        syncing = true;

        var update = {};

        // Sync scene -> scene2
        if (eventData['scene.camera']) {
            update['scene2.camera'] = eventData['scene.camera'];
        }
        // Sync scene2 -> scene
        if (eventData['scene2.camera']) {
            update['scene.camera'] = eventData['scene2.camera'];
        }

        if (Object.keys(update).length > 0) {
            Plotly.relayout(gd, update).then(function() {
                syncing = false;
            });
        } else {
            syncing = false;
        }
    });

    console.log('Camera sync enabled');
}

// Start checking when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(setupCameraSync, 500);
    });
} else {
    setTimeout(setupCameraSync, 500);
}
</script>
</body>"""

    # Read the generated HTML and inject the script before </body>
    with open(output_path, 'r') as f:
        html_content = f.read()

    html_content = html_content.replace('</body>', camera_sync_js)

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Done! Open {output_path} in a browser to view.")

    # Also print some statistics
    print("\n=== Statistics ===")
    print(f"Mesh hit count: {(mesh_ids >= 0).sum()} / {width*height}")
    print(f"Neural hit count: {(neural_ids >= 0).sum()} / {width*height}")

    # Compute position differences where both hit
    both_hit = (mesh_ids >= 0) & (neural_ids >= 0)
    if both_hit.sum() > 0:
        diff = np.abs(mesh_pos[both_hit] - neural_pos[both_hit])
        print(f"\nPosition differences (where both hit):")
        print(f"  Mean: {diff.mean():.4f}")
        print(f"  Max:  {diff.max():.4f}")
        print(f"  Std:  {diff.std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare hit positions between mesh and neural rendering'
    )
    parser.add_argument('mesh_prefix', help='Prefix for mesh output files (e.g., output/mesh_test)')
    parser.add_argument('neural_prefix', help='Prefix for neural output files (e.g., output/neural_test)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML file (default: output/hit_position_comparison.html)')
    parser.add_argument('-w', '--width', type=int, default=512, help='Image width')
    parser.add_argument('-H', '--height', type=int, default=512, help='Image height')
    parser.add_argument('-n', '--max-points', type=int, default=20000,
                        help='Maximum points per plot (default: 20000)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = 'output/hit_position_comparison.html'

    create_comparison_html(
        args.mesh_prefix,
        args.neural_prefix,
        args.output,
        args.width,
        args.height,
        args.max_points
    )


if __name__ == '__main__':
    main()
