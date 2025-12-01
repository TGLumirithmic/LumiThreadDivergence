# Docker Usage Quick Reference

This project uses docker-compose for better image naming and container management.

## Image and Container Names

- **Image name:** `neural-optix-renderer:latest` and `neural-optix-renderer:phase1`
- **Container name:** `neural-optix-dev`

These clean, short names make it easy to use the container outside of VSCode.

## Quick Commands

### Using docker-compose (Recommended)

```bash
# Start container in background
docker-compose -f .devcontainer/docker-compose.yml up -d

# Access the running container
docker exec -it neural-optix-dev bash

# View logs
docker-compose -f .devcontainer/docker-compose.yml logs -f

# Stop container
docker-compose -f .devcontainer/docker-compose.yml down

# Rebuild image
docker-compose -f .devcontainer/docker-compose.yml build --no-cache

# Stop and remove everything (including volumes)
docker-compose -f .devcontainer/docker-compose.yml down -v
```

### Using Docker directly

```bash
# Run interactively (exits when you exit)
docker run --gpus all -it --rm \
  --name neural-optix-dev \
  -v $(pwd):/workspace \
  -w /workspace \
  neural-optix-renderer:latest

# Run in background
docker run --gpus all -d \
  --name neural-optix-dev \
  -v $(pwd):/workspace \
  -w /workspace \
  neural-optix-renderer:latest sleep infinity

# Access running container
docker exec -it neural-optix-dev bash

# Stop and remove
docker stop neural-optix-dev
docker rm neural-optix-dev
```

## Image Management

```bash
# List images
docker images | grep neural-optix

# Remove old images
docker rmi neural-optix-renderer:latest

# Build manually
docker build -f .devcontainer/Dockerfile \
  -t neural-optix-renderer:latest \
  -t neural-optix-renderer:phase1 \
  --build-arg OPTIX_VERSION=9.0.0 \
  .
```

## Volume Management

The docker-compose setup uses named volumes for build and output directories:

```bash
# List volumes
docker volume ls | grep neural-optix

# Inspect volume
docker volume inspect neural-optix-build-cache

# Remove volumes (clears build cache)
docker volume rm neural-optix-build-cache neural-optix-output-cache

# Or remove with docker-compose
docker-compose -f .devcontainer/docker-compose.yml down -v
```

## Development Workflow

### Option 1: VSCode (Easiest)

1. Open project in VSCode: `code .`
2. Click "Reopen in Container"
3. Develop as normal

The image will be named `neural-optix-renderer:latest` automatically.

### Option 2: docker-compose CLI

```bash
# Start
docker-compose -f .devcontainer/docker-compose.yml up -d

# Develop
docker exec -it neural-optix-dev bash
cd /workspace/build
ninja
./bin/test_network ../data/models/model.bin

# Stop
docker-compose -f .devcontainer/docker-compose.yml down
```

### Option 3: Pure Docker

```bash
# Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  neural-optix-renderer:latest bash

# Build and run
cd /workspace/build
cmake -B . -G Ninja
ninja
./bin/test_network ../data/models/model.bin
```

## GPU Access Verification

```bash
# Inside container
nvidia-smi

# Should show your GPU
# If not, check NVIDIA Container Toolkit installation on host
```

## Troubleshooting

### Container name already in use

```bash
# Remove existing container
docker rm -f neural-optix-dev

# Or use a different name
docker run --name my-neural-optix ...
```

### Image not found

```bash
# Check if image exists
docker images | grep neural-optix

# If not, build it
docker-compose -f .devcontainer/docker-compose.yml build

# Or build with Docker
docker build -f .devcontainer/Dockerfile -t neural-optix-renderer:latest .
```

### GPU not accessible

```bash
# Verify on host
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Volume data persists

The build and output volumes persist between container runs. To reset:

```bash
# Remove volumes
docker-compose -f .devcontainer/docker-compose.yml down -v

# Next run will have fresh volumes
docker-compose -f .devcontainer/docker-compose.yml up -d
```

## Tips

1. **Use docker-compose** for consistent naming and configuration
2. **Named volumes** (`neural-optix-build-cache`) persist between runs for faster builds
3. **Container name** (`neural-optix-dev`) makes it easy to reference: `docker exec -it neural-optix-dev bash`
4. **Image tags** (`latest`, `phase1`) let you track different versions
5. **VSCode integration** automatically uses the same image name

## Cleanup

```bash
# Full cleanup (removes everything)
docker-compose -f .devcontainer/docker-compose.yml down -v
docker rmi neural-optix-renderer:latest neural-optix-renderer:phase1
docker volume prune
docker system prune -a
```
