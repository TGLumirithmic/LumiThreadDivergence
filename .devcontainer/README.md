# Development Container Configuration

This directory contains the VSCode Dev Container configuration for the Neural OptiX Renderer project.

## Overview

The development container provides a complete, reproducible development environment with:
- CUDA Toolkit 12.6
- OptiX SDK 9.0.0
- tiny-cuda-nn (pre-built)
- All necessary dependencies
- Development tools (CMake, GDB, etc.)
- Python environment with PyTorch

## Prerequisites

### Required

1. **Docker** - Install from [docker.com](https://docs.docker.com/get-docker/)
2. **NVIDIA Container Toolkit** - For GPU access
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **VSCode** with **Dev Containers extension**
   - Install VSCode: [code.visualstudio.com](https://code.visualstudio.com/)
   - Install extension: `ms-vscode-remote.remote-containers`

4. **OptiX SDK Installer**
   - Download from [NVIDIA OptiX Downloads](https://developer.nvidia.com/designworks/optix/downloads)
   - Place in `docker/` directory
   - See [docker/README.md](docker/README.md) for details

### Optional

- **NVIDIA GPU** with recent drivers (recommended for testing)
- At least **16GB RAM** (for building)
- **20GB free disk space** (for container image and build artifacts)

## Quick Start

### 1. Prepare OptiX Installer

Download the OptiX SDK installer and place it in the `docker/` directory:

```bash
# The file should be named:
.devcontainer/docker/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh
```

### 2. Open in VSCode

```bash
# Open the project
code /home/tom/work/LumiThreadDivergence

# VSCode will detect the devcontainer config and prompt:
# "Folder contains a Dev Container configuration file. Reopen folder to develop in a container?"

# Click "Reopen in Container" or run command:
# Ctrl+Shift+P -> "Dev Containers: Reopen in Container"
```

### 3. Wait for Build

The first build will take 10-20 minutes as it:
- Downloads base images (~3GB)
- Installs OptiX SDK
- Clones and builds tiny-cuda-nn
- Installs all dependencies

Subsequent opens are instant (container is cached).

### 4. Verify Setup

Once the container is ready, open a terminal in VSCode and run:

```bash
# Check CUDA
nvcc --version

# Check GPU access
nvidia-smi

# Check OptiX
ls $OptiX_INSTALL_DIR/include

# Check tiny-cuda-nn
ls /usr/local/include/tiny-cuda-nn
```

### 5. Build the Project

```bash
cd /workspace/build
ninja

# Or use CMake Tools extension (Ctrl+Shift+P -> "CMake: Build")
```

## Directory Structure

```
.devcontainer/
├── devcontainer.json       # Main configuration file
├── Dockerfile              # Container image definition
├── post-create.sh          # Setup script (runs after container creation)
├── docker/                 # Docker resources
│   ├── README.md           # Instructions for OptiX SDK
│   ├── NVIDIA-OptiX-SDK-*.sh  # OptiX installer (you must download)
│   └── Dockerfile.original # Original Dockerfile (reference)
└── README.md              # This file
```

## Container Features

### Installed Software

**Build Tools:**
- CMake 3.28+
- Ninja build system
- GCC/G++ compiler
- Clang tools (format, tidy)

**CUDA & Graphics:**
- CUDA Toolkit 12.6
- OptiX SDK 9.0.0
- OpenGL libraries
- GLFW, GLM

**Python:**
- Python 3.12
- PyTorch 2.x
- NumPy, Matplotlib, Pillow

**Development:**
- GDB debugger
- Valgrind
- Git, GitHub CLI
- Vim, Nano

### VSCode Extensions

Automatically installed:
- C/C++ IntelliSense
- CMake Tools
- NVIDIA Nsight VSCode Edition
- Python support
- Makefile Tools

### Environment Variables

- `OptiX_INSTALL_DIR=/opt/optix`
- `TCNN_DIR=/opt/tiny-cuda-nn`
- `CUDA_HOME=/usr/local/cuda`
- `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`

## Workspace Layout

The container mounts your local directory to `/workspace`:

```
/workspace/               # Your local project directory
├── build/               # CMake build directory (Docker volume)
├── output/              # Rendered images (Docker volume)
├── src/                 # Source code
├── data/                # Runtime data (models, scenes, etc.)
└── ...
```

**Note:** `build/` and `output/` use Docker volumes for better performance.

## Common Tasks

### Build the Project

```bash
# Configure (first time only)
cd /workspace
cmake -B build -G Ninja

# Build
cd build
ninja

# Or rebuild from scratch
ninja clean && ninja
```

### Run Phase 1 Test

```bash
# Convert PyTorch checkpoint
python3 scripts/convert_checkpoint.py model.pth data/models/model.bin

# Run test
./build/bin/test_network data/models/model.bin

# View outputs
ls -lh output/
```

### Debug with GDB

```bash
gdb ./build/bin/test_network
(gdb) run data/models/model.bin
(gdb) bt  # backtrace on crash
```

### Profile with Nsight Compute

```bash
ncu --set full ./build/bin/test_network data/models/model.bin
```

## Customization

### Change CUDA Version

Edit `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04
# Change to desired CUDA version
```

### Change OptiX Version

Edit `devcontainer.json`:
```json
"args": {
  "OPTIX_VERSION": "9.0.0"  // Change to desired version
}
```

Then update the installer filename in `docker/`.

### Add Python Packages

Edit `Dockerfile` to add to the `pip3 install` command, or install interactively:
```bash
pip3 install <package-name>
```

### Modify VSCode Settings

Edit `devcontainer.json` under `customizations.vscode.settings`.

## Troubleshooting

### Container Fails to Build

**OptiX installer not found:**
```
Error: COPY docker/NVIDIA-OptiX-SDK-*.sh /tmp/
```
Solution: Download OptiX SDK from NVIDIA and place in `docker/` directory.

**Out of disk space:**
```
Error: no space left on device
```
Solution: Clean up Docker:
```bash
docker system prune -a --volumes
```

### GPU Not Accessible

**nvidia-smi fails in container:**
```bash
# On host, verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

If this works but devcontainer doesn't, rebuild:
```bash
# In VSCode: Ctrl+Shift+P -> "Dev Containers: Rebuild Container"
```

### Build Errors

**CMake can't find OptiX:**
```
CMake Error: Could not find OptiX
```
Check environment variable:
```bash
echo $OptiX_INSTALL_DIR
ls $OptiX_INSTALL_DIR/include
```

**tiny-cuda-nn not found:**
```bash
ls /usr/local/include/tiny-cuda-nn
ls /usr/local/lib/libtiny-cuda-nn*
```

If missing, rebuild container from scratch:
```bash
docker system prune -a
# Reopen in VSCode
```

## Performance Notes

- **First build:** 10-20 minutes (installs everything)
- **Subsequent builds:** Instant (cached container)
- **Project compilation:** ~1-2 minutes for full build
- **Container size:** ~8-10GB

## Using the Image Outside VSCode

The dev container builds an image named `neural-optix-renderer:latest` which you can use directly with Docker or docker-compose:

### Option 1: Using docker-compose

```bash
# Start the container
cd /home/tom/work/LumiThreadDivergence
docker-compose -f .devcontainer/docker-compose.yml up -d

# Access the container
docker exec -it neural-optix-dev bash

# Inside container
cd /workspace/build
ninja
./bin/test_network ../data/models/model.bin

# Stop when done
docker-compose -f .devcontainer/docker-compose.yml down
```

### Option 2: Using Docker directly

```bash
# Use the image built by VSCode
docker run --gpus all -it --rm \
  --name neural-optix-dev \
  -v $(pwd):/workspace \
  -w /workspace \
  neural-optix-renderer:latest bash

# Or build manually with a clean name
docker build -f .devcontainer/Dockerfile \
  -t neural-optix-renderer:latest \
  --build-arg OPTIX_VERSION=9.0.0 \
  .

# Run
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  neural-optix-renderer:latest
```

### List Available Images

```bash
# See the image
docker images | grep neural-optix

# Should show:
# neural-optix-renderer   latest    <id>   <time>   <size>
# neural-optix-renderer   phase1    <id>   <time>   <size>
```

## References

- [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Documentation](https://docs.docker.com/)
- [OptiX Documentation](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
