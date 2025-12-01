# Neural OptiX Renderer Development Container
# Multi-stage build for optimal image size

# Stage 1: Build OptiX SDK and dependencies
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 as builder

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
RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" \
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

# # Build OptiX SDK samples (optional)
# RUN cd /opt/optix && \
#     mkdir -p build && cd build && \
#     cmake ../SDK && make -j$(nproc) install

# Stage 2: Development environment
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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
RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" \
        > /etc/apt/sources.list.d/kitware.list

# Install latest CMake
RUN apt-get update && apt-get install -y cmake


# Clone and build tiny-cuda-nn for GTX 980 (sm_52)
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
        -DCMAKE_CUDA_ARCHITECTURES=52 \
        -DTCNN_MIN_GPU_ARCH=52 \
        -DTCNN_ENABLE_HALF_PRECISION=OFF \
        -DTCNN_DISABLE_TENSOR_CORES=ON && \
    cmake --build build --config Release -j $(nproc)

# Python packages
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    matplotlib \
    pillow \
    pyyaml

# Copy OptiX SDK from builder
COPY --from=builder /opt/optix /opt/optix

# Copy tiny-cuda-nn from builder
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/ /usr/local/include/

RUN ldconfig

# Environment variables
ENV OptiX_INSTALL_DIR=/opt/optix
ENV TCNN_DIR=/opt/tiny-cuda-nn/
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
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
