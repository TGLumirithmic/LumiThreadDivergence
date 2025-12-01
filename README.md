# Neural OptiX Renderer

An OptiX-based ray tracer that renders scenes containing both traditional triangle meshes and neural assets (Instant-NGP style NeRFs using tiny-cuda-nn). This project demonstrates SIMD/thread divergence when mixing geometry types, measurable via Nsight Compute profiling.

## Project Status

**Phase 1 (Current):** Neural Network Integration - Proof of Concept
- Load PyTorch weights into tiny-cuda-nn
- Initialize multi-decoder architecture (visibility, normal, depth)
- Run test queries and validate outputs

**Phase 2:** Basic OptiX + tiny-cuda-nn integration
**Phase 3:** Mixed geometry types (triangles + neural assets)
**Phase 4:** Scene loading from YAML

See [SPECIFICATION.md](SPECIFICATION.md) for full details.

## Architecture

The project uses a multi-decoder neural architecture:
- **Shared Hash Encoding:** Instant-NGP style multi-resolution hash grid
- **Shared Encoder:** MLP that processes encoded features
- **Three Decoders:**
  - **Visibility Decoder:** 1D output for any-hit testing (opacity)
  - **Normal Decoder:** 3D output for surface normals
  - **Depth Decoder:** 1D output for closest-hit distance

## Requirements

- **CUDA:** 11.x or 12.x
- **OptiX:** 7.x or 8.x (Phase 2+)
- **CMake:** 3.18+
- **GPU:** NVIDIA GPU with compute capability 8.6+ (Ampere or newer recommended)
- **Dependencies:**
  - [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
  - yaml-cpp (Phase 4)
  - tinyobjloader (Phase 3)

## Installation

### Option 1: Using Dev Container (Recommended)

The easiest way to get started is using the VSCode Dev Container, which provides a complete development environment with all dependencies pre-installed.

**Prerequisites:**
- Docker
- NVIDIA Container Toolkit
- VSCode with Dev Containers extension
- OptiX SDK installer (download from NVIDIA)

See [.devcontainer/README.md](.devcontainer/README.md) for detailed setup instructions.

**Quick Start:**
```bash
# 1. Download OptiX SDK from NVIDIA and place in .devcontainer/docker/
# 2. Open project in VSCode
code .
# 3. Click "Reopen in Container" when prompted
# 4. Build and run
cd /workspace/build
ninja
./bin/test_network ../data/models/model.bin
```

### Option 2: Manual Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd LumiThreadDivergence
```

#### 2. Install tiny-cuda-nn

```bash
# Clone into external directory
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn external/tiny-cuda-nn

# OR install system-wide
cd /tmp
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn
cmake . -B build
cmake --build build --config Release
sudo cmake --install build
```

#### 3. Build the Project

```bash
mkdir build
cd build

# Configure
cmake ..

# Build
make -j$(nproc)
```

Build outputs:
- `build/bin/test_network` - Phase 1 test program
- `build/lib/` - Compiled PTX files (Phase 2+)

> **Note:** Using the Dev Container is strongly recommended as it handles all dependencies automatically, including OptiX SDK installation and tiny-cuda-nn compilation.

## Usage

### Phase 1: Testing Neural Network

#### 1. Prepare Your Model Weights

Convert a PyTorch checkpoint to binary format:

```bash
# Assuming you have a PyTorch checkpoint
python scripts/convert_checkpoint.py model.pth data/models/model.bin

# Check what's in the checkpoint first
python scripts/convert_checkpoint.py model.pth --info
```

See [docs/weight_format.md](docs/weight_format.md) for details on the expected PyTorch model structure.

#### 2. Run the Test Program

```bash
# Basic test (visualizes network outputs)
./build/bin/test_network data/models/model.bin

# Test with ground truth predictions
./build/bin/test_network data/models/model.bin data/test/predictions.bin

# Specify output directory
./build/bin/test_network data/models/model.bin "" output/
```

#### 3. View Results

The test program generates visualization images in PPM format:
- `output/visibility.ppm` - Visibility/opacity field
- `output/normals.ppm` - Normal vectors (as RGB)
- `output/depth.ppm` - Depth values

Convert to PNG if needed:
```bash
convert output/visibility.ppm output/visibility.png
```

## Project Structure

```
neural-optix-renderer/
├── src/                      # Host-side source code
│   ├── neural/              # Neural network integration
│   │   ├── network.cpp/h    # tiny-cuda-nn wrapper
│   │   ├── weight_loader.cpp/h  # PyTorch weight loading
│   │   └── config.cpp/h     # Network configuration
│   └── utils/               # Utility functions
│       ├── cuda_utils.h     # CUDA helpers
│       └── error.h          # Error checking macros
├── tests/                   # Test programs
│   ├── test_network.cu      # Phase 1 test
│   └── neural_proxy_predictions.h  # Prediction file format
├── scripts/                 # Python utilities
│   └── convert_checkpoint.py  # PyTorch → binary converter
├── data/                    # Runtime data
│   ├── models/             # Neural network weights
│   ├── test/               # Ground truth predictions
│   ├── scenes/             # YAML scene files (Phase 4)
│   └── meshes/             # Geometry files (Phase 3)
└── docs/                   # Documentation
    ├── weight_format.md    # Weight conversion guide
    └── nsight_profiling.md # Profiling instructions
```

## Development Roadmap

### Phase 1 (Current): Neural Network Integration ✓
- [x] Set up tiny-cuda-nn integration
- [x] Create weight loading system
- [x] Implement multi-decoder architecture
- [x] Test harness for validation
- [ ] Load actual weights into tiny-cuda-nn (TODO)
- [ ] Implement real inference calls (TODO)

### Phase 2: OptiX Integration
- [ ] Create OptiX pipeline
- [ ] Implement neural asset as custom primitive
- [ ] Ray-AABB intersection
- [ ] Call tiny-cuda-nn from closest-hit program

### Phase 3: Mixed Geometry
- [ ] Triangle mesh support
- [ ] TLAS with multiple instance types
- [ ] Basic path tracing
- [ ] Mixed scene rendering

### Phase 4: Scene Loading
- [ ] YAML parser
- [ ] Mesh loading (OBJ)
- [ ] Dynamic scene construction

## Network Configuration

The default Instant-NGP configuration:

```cpp
neural::NetworkConfig config;
config.n_levels = 16;
config.n_features_per_level = 2;
config.log2_hashmap_size = 19;
config.base_resolution = 16.0f;
config.max_resolution = 2048.0f;
config.n_hidden_layers = 2;
config.n_neurons = 64;
```

Decoders:
- **Visibility:** 1D output, sigmoid activation
- **Normal:** 3D output, normalized
- **Depth:** 1D output, linear

## Troubleshooting

### Build Issues

**CMake can't find CUDA:**
```bash
export CUDA_PATH=/usr/local/cuda
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**tiny-cuda-nn not found:**
```bash
# Make sure it's in external/ or installed system-wide
ls external/tiny-cuda-nn  # Should exist

# Or install system-wide
cd external/tiny-cuda-nn
cmake . -B build
sudo cmake --install build
```

### Runtime Issues

**"Failed to load weights":**
- Make sure you converted the PyTorch checkpoint using `scripts/convert_checkpoint.py`
- Check that the file path is correct
- Verify the file has the "TCNN" magic header

**CUDA out of memory:**
- Reduce batch size in test queries
- Use a smaller network configuration
- Check GPU memory with `nvidia-smi`

## Profiling

For Nsight Compute profiling (Phase 3+):

```bash
ncu --set full --target-processes all ./renderer scene.yaml
```

Key metrics for divergence analysis:
- `smsp__sass_average_branch_targets_threads_uniform.pct`
- Warp execution efficiency
- Branch divergence during neural intersection/hit programs

See [docs/nsight_profiling.md](docs/nsight_profiling.md) for detailed profiling instructions.

## Contributing

This is a research/demo project. For major changes, please open an issue first.

## License

[Specify license]

## References

- [Instant Neural Graphics Primitives (Instant-NGP)](https://nvlabs.github.io/instant-ngp/)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [OptiX Documentation](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf)

## Contact

[Your contact information]
