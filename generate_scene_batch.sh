#!/bin/bash
# Batch scene generation script
# Generates scenes with different numbers of spheres for performance testing

# Array of sphere counts to generate
SPHERE_COUNTS=(1 2 4 8 16 32 64)

# Configuration
ARENA_SIZE=15.0
WALL_HEIGHT=8.0
MIN_RADIUS=0.3
MAX_RADIUS=1.0
SEED=42

echo "Generating scene files for performance testing..."
echo "================================================"

for count in "${SPHERE_COUNTS[@]}"; do
    echo ""
    echo "Generating scenes with $count spheres..."
    python3 generate_scenes.py $count \
        --arena-size $ARENA_SIZE \
        --wall-height $WALL_HEIGHT \
        --min-radius $MIN_RADIUS \
        --max-radius $MAX_RADIUS \
        --seed $SEED \
        --prefix scene

    if [ $? -ne 0 ]; then
        echo "Error generating scene with $count spheres"
        exit 1
    fi
done

echo ""
echo "================================================"
echo "Scene generation complete!"
echo ""
echo "Generated files:"
ls -1 scene_*_*.yaml
