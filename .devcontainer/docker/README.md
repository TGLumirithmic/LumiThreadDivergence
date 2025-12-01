# Docker Resources

This directory contains Docker-related resources for the Neural OptiX Renderer.

## Contents

- `NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh` - OptiX SDK installer (required for build)
- `Dockerfile.original` - Original Dockerfile (kept for reference)

## OptiX SDK Installer

The OptiX SDK installer is required to build the development container. Due to NVIDIA's licensing terms, this file cannot be distributed directly and must be obtained from NVIDIA.

### Obtaining the OptiX SDK

1. Visit the [NVIDIA OptiX Download Page](https://developer.nvidia.com/designworks/optix/downloads)
2. Sign in with your NVIDIA Developer account (free registration required)
3. Accept the license agreement
4. Download **OptiX SDK 9.0.0 for Linux**
5. Place the `.sh` file in this directory

The file should be named exactly: `NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh`

### Alternative Versions

If you want to use a different OptiX version:

1. Download the desired version from NVIDIA
2. Update the `OPTIX_VERSION` build arg in `../.devcontainer/devcontainer.json`
3. Update line 26 in `../Dockerfile` to reference the correct filename

## Building the Container

The container is automatically built when you open the project in VSCode with the Dev Containers extension.

### Manual Build

If you need to build manually:

```bash
# From the project root
cd /home/tom/work/LumiThreadDivergence

# Build the image
docker build \
  -f .devcontainer/Dockerfile \
  -t neural-optix-renderer:latest \
  --build-arg OPTIX_VERSION=9.0.0 \
  .

# Run the container
docker run --gpus all -it \
  -v $(pwd):/workspace \
  neural-optix-renderer:latest
```

## Notes

- The OptiX installer file is large (~500MB) and is excluded from git via `.gitignore`
- The Dockerfile uses a multi-stage build to keep the final image size manageable
- GPU access requires the NVIDIA Container Toolkit to be installed on the host
