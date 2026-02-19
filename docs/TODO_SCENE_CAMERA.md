# TODO: Optimize Camera Positioning for Scene Generator

## Goal
Modify the scene generator script to automatically adjust camera position so that the generated scene is centered in the view and occupies as much of the output image as possible.

## Current Behavior
- Camera distance is calculated as `arena_size * 0.75`
- Camera height is `arena_size * 0.4`
- Camera position: `[arena_size * 0.6, height, arena_size]`
- Look-at point: `[0.0, arena_size * 0.2, 0.0]`

These are basic heuristics that may not optimally frame scenes with different sphere distributions.

## Proposed Improvements

### 1. Calculate Scene Bounding Box
- Compute the actual AABB of all placed spheres (including floor/walls)
- Use this to determine the scene's true extent

### 2. Camera Positioning Strategies

**Option A: Fit scene to view frustum**
- Calculate the scene's bounding sphere
- Position camera to fit the bounding sphere within the FOV
- Center the look-at point on the scene centroid

**Option B: Maximize sphere coverage**
- Ignore floor/walls (they're just context)
- Focus on the sphere AABB centroid
- Adjust camera distance to fit all spheres with minimal padding

**Option C: Fixed aspect optimization**
- For a given FOV (default 90°), calculate minimum distance to fit scene
- Apply padding factor (e.g., 1.1x) to avoid edge clipping

### 3. Implementation Location
File: `generate_scenes.py`
Function: `generate_scene_yaml()` around line 150

```python
def calculate_optimal_camera(spheres, arena_size, fov=90, padding=1.2):
    """
    Calculate optimal camera position to frame the scene.

    Args:
        spheres: List of sphere configurations
        arena_size: Size of the arena
        fov: Camera field of view in degrees
        padding: Padding factor (1.2 = 20% margin)

    Returns:
        (camera_position, look_at_position)
    """
    # Compute scene bounds including all spheres
    # Calculate centroid
    # Compute required camera distance based on FOV
    # Return optimized camera parameters
```

### 4. Considerations
- Should the camera position change with sphere count?
- Should we consider vertical vs horizontal FOV for non-square images?
- Do we want consistent camera positions for performance comparison?
  - If yes: base on arena_size only (current approach)
  - If no: base on actual sphere distribution (better framing)

### 5. Testing
After implementation, generate scenes and visually verify:
```bash
python3 generate_scenes.py 4 --seed 42
python3 generate_scenes.py 16 --seed 42
python3 generate_scenes.py 64 --seed 42 --arena-size 20
# Render and check that spheres fill the frame appropriately
```

## Priority
Medium - Current camera positioning works but isn't optimal for visual analysis or presentations.

## Related Files
- `generate_scenes.py` - Main implementation
- `scene_mesh.yaml`, `scene_neural.yaml` - Example reference scenes
- `src/main.cpp:628-638` - Camera basis vector calculation
