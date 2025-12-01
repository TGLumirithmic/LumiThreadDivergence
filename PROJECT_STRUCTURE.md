# Project Structure

```
neural-optix-renderer/
├── CMakeLists.txt                 # Main build configuration
├── README.md                      # Project overview and build instructions
├── SPECIFICATION.md               # Detailed project specification
│
├── src/                           # Host-side source code
│   ├── main.cpp                   # Entry point, argument parsing
│   │
│   ├── optix/                     # OptiX pipeline management
│   │   ├── context.cpp/h          # OptiX context initialization
│   │   ├── pipeline.cpp/h         # Pipeline and module creation
│   │   ├── sbt.cpp/h              # Shader binding table management
│   │   └── geometry.cpp/h         # BLAS/TLAS construction
│   │
│   ├── neural/                    # Neural network integration
│   │   ├── network.cpp/h          # tiny-cuda-nn wrapper/interface
│   │   ├── weight_loader.cpp/h    # Load PyTorch weights
│   │   └── config.cpp/h           # Network architecture configuration
│   │
│   ├── scene/                     # Scene management
│   │   ├── scene.cpp/h            # Scene container and management
│   │   ├── yaml_loader.cpp/h      # YAML scene file parser
│   │   ├── mesh_loader.cpp/h      # OBJ/mesh file loading
│   │   ├── camera.cpp/h           # Camera setup and controls
│   │   └── light.cpp/h            # Light sources
│   │
│   ├── render/                    # Rendering coordination
│   │   ├── renderer.cpp/h         # Main render loop
│   │   └── output.cpp/h           # Image output (PNG/EXR)
│   │
│   └── utils/                     # Utility functions
│       ├── math.h                 # Vector math, transforms
│       ├── cuda_utils.h           # CUDA helper functions
│       └── error.h                # Error checking macros
│
├── programs/                      # OptiX device programs (.cu files)
│   ├── common.h                   # Shared structures (LaunchParams, etc.)
│   ├── raygen.cu                  # Ray generation programs
│   ├── miss.cu                    # Miss programs
│   │
│   ├── triangle/                  # Triangle mesh programs
│   │   ├── closest_hit.cu         # Triangle closest hit
│   │   └── any_hit.cu             # Triangle any hit (optional)
│   │
│   └── neural/                    # Neural asset programs
│       ├── intersection.cu        # AABB intersection
│       └── closest_hit.cu         # Neural field sampling
│
├── include/                       # Public headers
│   └── launch_params.h            # Shared launch parameters structure
│
├── tests/                         # Test programs
│   ├── test_network.cu            # Phase 1: Network loading test
│   ├── neural_proxy_predictions.h # Prediction file format and loader
│   └── CMakeLists.txt             # Test build config
│
├── data/                          # Runtime data
│   ├── scenes/                    # YAML scene definitions
│   │   ├── simple.yaml            # Single neural asset
│   │   ├── mixed.yaml             # Mixed geometry types
│   │   └── complex.yaml           # Full test scene
│   │
│   ├── models/                    # Neural network weights
│   │   ├── test_ngp.pth           # Test Instant-NGP weights
│   │   └── README.md              # Weight format documentation
│   │
│   ├── test/                      # Test/reference data
│   │   ├── predictions.bin        # Ground truth predictions from PyTorch
│   │   └── README.md              # Test data format documentation
│   │
│   └── meshes/                    # Geometry files
│       ├── cube.obj               # Simple test geometry
│       └── room.obj               # Test room
│
├── output/                        # Rendered images (gitignored)
│   └── .gitkeep
│
├── build/                         # Build artifacts (gitignored)
│   └── .gitkeep
│
├── external/                      # Third-party dependencies
│   ├── CMakeLists.txt             # External dependency management
│   ├── tiny-cuda-nn/              # Git submodule or local copy
│   ├── yaml-cpp/                  # Git submodule or system install
│   └── tinyobjloader/             # Header-only, copied directly
│
└── docs/                          # Additional documentation
    ├── weight_format.md           # PyTorch → tiny-cuda-nn conversion
    ├── nsight_profiling.md        # Profiling instructions
    └── scene_format.md            # YAML scene schema reference
```

## Key Files Description

### Build System
- **CMakeLists.txt**: Main build file that finds OptiX SDK, CUDA, and links dependencies
- **external/CMakeLists.txt**: Handles third-party library integration

### Host Code
- **src/main.cpp**: Parses command line args, loads scene, runs render loop
- **src/optix/**: All OptiX initialization and setup (context, modules, pipeline, SBT)
- **src/neural/**: Wraps tiny-cuda-nn, handles weight loading from PyTorch
- **src/scene/**: Scene graph, YAML parsing, asset loading
- **src/render/**: High-level rendering logic and output

### Device Code
- **programs/common.h**: Structures shared between host and device (LaunchParams, ray payloads)
- **programs/raygen.cu**: Generates camera rays
- **programs/miss.cu**: Background color
- **programs/triangle/**: Traditional mesh hit programs
- **programs/neural/**: Neural asset intersection and sampling

### Data
- **data/scenes/**: YAML scene files for different test cases
- **data/models/**: Neural network weights exported from PyTorch
- **data/test/**: Ground truth predictions for validation (binary format)
- **data/meshes/**: Simple test geometry (cubes, rooms, etc.)

### Tests
- **tests/test_network.cu**: Phase 1 standalone test - loads weights, runs inference, outputs image
- **tests/neural_proxy_predictions.h**: Defines binary format for prediction data and provides loading utilities

## Build Output Structure
```
build/
├── bin/
│   ├── renderer                   # Main executable
│   └── test_network               # Phase 1 test
├── lib/
│   └── *.ptx                      # Compiled OptiX programs
└── CMakeFiles/
```

## Notes
- OptiX device programs (.cu in programs/) get compiled to PTX at build time
- PTX files are loaded at runtime by the OptiX pipeline
- Network weights in data/models/ are loaded at runtime
- Test predictions in data/test/ are used for Phase 1 validation (compare tiny-cuda-nn output against PyTorch ground truth)
- Scene YAML files reference meshes and weights by relative paths
- Output images go to output/ directory (user can override via command line)