# Neural OptiX Renderer Development Container
# Multi-stage build for optimal image size

# Base image configuration
# For GTX 980 / lower resource: nvidia/cuda:11.6.1-devel-ubuntu20.04
# For RTX 3090/4090 / high resource: nvidia/cuda:12.3.2-devel-ubuntu22.04 or nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG CUDA_BASE_IMAGE=nvidia/cuda:11.6.1-devel-ubuntu20.04

# Stage 1: Build OptiX SDK and dependencies
FROM ${CUDA_BASE_IMAGE} as builder

ARG OPTIX_VERSION=9.0.0
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    gnupg \
    cmake \
    build-essential \
    ninja-build \
    libglm-dev \
    libglfw3-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libomp-dev \
    libboost-all-dev \
    lsb-release \
    python3 \
    python3-pip \
    python3-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# Add Kitware repo manually (without add-apt-repository)
# Automatically detects Ubuntu version (focal/jammy/noble)
RUN UBUNTU_CODENAME=$(lsb_release -cs) && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/kitware.list

# Install latest CMake
RUN apt-get update && apt-get install -y cmake

# Install Python dependencies for checkpoint conversion
RUN pip3 install --no-cache-dir \
    torch \
    numpy

# Install OptiX SDK
RUN mkdir -p /opt/optix /tmp/optix-installer
COPY .devcontainer/docker/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh /tmp/optix-installer/
RUN chmod +x /tmp/optix-installer/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh && \
    /tmp/optix-installer/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh \
        --skip-license \
        --prefix=/opt/optix && \
    rm -rf /tmp/optix-installer

# Note: The Kitware CMake repository URL is automatically configured for the Ubuntu version

# # Build OptiX SDK samples (optional)
# RUN cd /opt/optix && \
#     mkdir -p build && cd build && \
#     cmake ../SDK && make -j$(nproc) install

# Stage 2: Development environment
FROM ${CUDA_BASE_IMAGE}

ARG CUDA_BASE_IMAGE=nvidia/cuda:11.6.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# GPU configuration arguments
# For GTX 980: CUDA_ARCH=52, ENABLE_HALF_PRECISION=OFF
# For RTX 3090/4090: CUDA_ARCH=86, ENABLE_HALF_PRECISION=ON
# For RTX 2080 Ti: CUDA_ARCH=75, ENABLE_HALF_PRECISION=ON
ARG CUDA_ARCH=52
ARG ENABLE_HALF_PRECISION=OFF

# Install runtime and development dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    build-essential \
    git \
    libglm-dev \
    libglfw3-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libomp-dev \
    libyaml-cpp-dev \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    python3 \
    python3-pip \
    python3-venv \
    imagemagick \
    vim \
    nano \
    wget \
    curl \
    zip \
    unzip \
    htop \
    sudo \
    software-properties-common \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# Add Kitware repo manually (without add-apt-repository)
# Automatically detects Ubuntu version (focal/jammy/noble)
RUN UBUNTU_CODENAME=$(lsb_release -cs) && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/kitware.list

# Install latest CMake
RUN apt-get update && apt-get install -y cmake


# Clone and build tiny-cuda-nn with configurable GPU support
# Note: GTX 980 (sm_52) doesn't support half precision in tiny-cuda-nn
# Must build with TCNN_HALF_PRECISION=OFF to use float precision
# Note we don't do a system install as historically this has caused stale versions
# lingering that get out of sync and cause the project build to fail
WORKDIR /opt
RUN git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git /opt/tiny-cuda-nn && \
    cd /opt/tiny-cuda-nn && \
    cmake . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DTCNN_BUILD_BENCHMARK=OFF \
        -DTCNN_BUILD_EXAMPLES=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DTCNN_MIN_GPU_ARCH=${CUDA_ARCH} \
        -DTCNN_ENABLE_HALF_PRECISION=${ENABLE_HALF_PRECISION} \
        -DTCNN_DISABLE_TENSOR_CORES=$(if [ "${CUDA_ARCH}" = "52" ]; then echo "ON"; else echo "OFF"; fi) && \
    cmake --build build --config Release -j $(nproc)

# Python packages
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    matplotlib \
    pillow \
    pyyaml

# Clone and build HIPRT with NVIDIA/CUDA support via Orochi
# HIPRT automatically detects CUDA through Orochi when available
# For NVIDIA-only builds: compile bitcode with --no-amd --nvidia flags
WORKDIR /opt
RUN git clone --recursive https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git /opt/hiprt-src && \
    cd /opt/hiprt-src && \
    mkdir -p build && \
    # Configure with BITCODE=ON to enable precompiled kernel support
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBITCODE=ON \
        -DNO_UNITTEST=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/hiprt && \
    # Build the library
    cmake --build build --config Release -j $(nproc) && \
    # Compile NVIDIA bitcode kernels (--no-amd to skip AMD, --nvidia to enable NVIDIA)
    cd /opt/hiprt-src/scripts/bitcodes && \
    python3 compile.py --no-amd --nvidia && \
    cd /opt/hiprt-src && \
    # Manual install since HIPRT doesn't have a standard install target
    mkdir -p /opt/hiprt/include /opt/hiprt/lib /opt/hiprt/bin && \
    # Copy main headers
    cp -r /opt/hiprt-src/hiprt/*.h /opt/hiprt/include/ && \
    # Copy implementation headers (needed for custom intersect functions)
    mkdir -p /opt/hiprt/include/hiprt && \
    cp -r /opt/hiprt-src/hiprt/impl /opt/hiprt/include/hiprt/ && \
    # Copy Orochi headers and sources (needed for CUDA/HIP abstraction)
    cp -r /opt/hiprt-src/contrib/Orochi /opt/hiprt/include/ && \
    # Copy contrib directory (cuew/hipew for CUDA/HIP wrappers)
    cp -r /opt/hiprt-src/contrib /opt/hiprt/include/ && \
    # Copy built library (look for versioned library name)
    find /opt/hiprt-src/build -name "libhiprt*.so*" -exec cp {} /opt/hiprt/lib/ \; && \
    # Copy compiled bitcode/fatbin files if they exist
    find /opt/hiprt-src -name "*.hipfb" -exec cp {} /opt/hiprt/bin/ \; 2>/dev/null || true && \
    find /opt/hiprt-src -name "*nvidia*.fatbin" -exec cp {} /opt/hiprt/bin/ \; 2>/dev/null || true && \
    # Create symlink if needed
    cd /opt/hiprt/lib && \
    if [ -f libhiprt0300064.so ]; then ln -sf libhiprt0300064.so libhiprt.so; fi && \
    # Cleanup source to save space
    rm -rf /opt/hiprt-src

# Copy OptiX SDK from builder
COPY --from=builder /opt/optix /opt/optix

# Copy tiny-cuda-nn from builder
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/ /usr/local/include/

RUN ldconfig

# Environment variables
ENV OptiX_INSTALL_DIR=/opt/optix
ENV TCNN_DIR=/opt/tiny-cuda-nn/
ENV HIPRT_DIR=/opt/hiprt
ENV LD_LIBRARY_PATH=/opt/hiprt/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    usermod -aG video,sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace
RUN chown -R $USERNAME:$USERNAME /workspace

USER $USERNAME

RUN mkdir -p /workspace/output /workspace/build /workspace/data/models /workspace/data/test

CMD ["/bin/bash"]
