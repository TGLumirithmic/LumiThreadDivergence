#!/bin/bash
# Post-create script for Neural OptiX Renderer dev container
# Runs once after the container is created

set -e

echo "========================================="
echo "Neural OptiX Renderer - Post-Create Setup"
echo "========================================="

# Verify CUDA installation
echo ""
echo "Verifying CUDA installation..."
nvcc --version

# Verify GPU access
echo ""
echo "Checking GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "Warning: nvidia-smi failed. GPU may not be accessible."
else
    echo "Warning: nvidia-smi not found. Running without GPU access?"
fi

# Verify OptiX installation
echo ""
echo "Verifying OptiX SDK..."
if [ -d "$OptiX_INSTALL_DIR/include" ]; then
    echo "OptiX SDK found at: $OptiX_INSTALL_DIR"
    ls -la "$OptiX_INSTALL_DIR/include" | head -5
else
    echo "Warning: OptiX SDK not found at $OptiX_INSTALL_DIR"
fi

# Verify tiny-cuda-nn installation
echo ""
echo "Verifying tiny-cuda-nn..."
if [ -d "/usr/local/include/tiny-cuda-nn" ]; then
    echo "tiny-cuda-nn headers found"
else
    echo "Warning: tiny-cuda-nn headers not found"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p /workspace/build
mkdir -p /workspace/output
mkdir -p /workspace/data/{models,test,scenes,meshes}

# Set proper permissions
echo "Setting permissions..."
sudo chown -R developer:developer /workspace/build /workspace/output /workspace/data

# Configure CMake (if CMakeLists.txt exists)
if [ -f "/workspace/CMakeLists.txt" ]; then
    echo ""
    echo "Configuring CMake build..."
    cd /workspace
    cmake -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DOptiX_INSTALL_DIR=/opt/optix \
        || echo "Warning: CMake configuration failed. You may need to run it manually."
else
    echo "Warning: CMakeLists.txt not found. Skipping CMake configuration."
fi

# Print helpful information
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Environment Variables:"
echo "  OptiX_INSTALL_DIR: $OptiX_INSTALL_DIR"
echo "  TCNN_DIR: $TCNN_DIR"
echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo ""
echo "Quick Start:"
echo "  1. Build the project:"
echo "     cd /workspace/build && ninja"
echo ""
echo "  2. Convert PyTorch checkpoint:"
echo "     python3 scripts/convert_checkpoint.py model.pth data/models/model.bin"
echo ""
echo "  3. Run Phase 1 test:"
echo "     ./build/bin/test_network data/models/model.bin"
echo ""
echo "  4. View results:"
echo "     ls -lh output/"
echo ""
echo "Useful Commands:"
echo "  - Rebuild: cd /workspace/build && ninja clean && ninja"
echo "  - Run tests: cd /workspace/build && ctest"
echo "  - Profile: ncu ./build/bin/test_network <args>"
echo ""
echo "========================================="
