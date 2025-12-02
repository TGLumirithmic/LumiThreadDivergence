#! /bin/bash
set -euxo pipefail

echo "==== DLVM CUDA 12.4 (Debian 11) GPU Docker setup starting ===="

############################################
# 0. Work around dead bullseye-backports repo
############################################
# Some older Debian 11 images still have bullseye-backports configured
# at a URL that no longer has a Release file. That makes apt-get update
# exit with code 100, which would kill this script under set -e.
# We comment those entries out once, up-front.

for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
  if [ -f "$f" ]; then
    # Comment out any lines mentioning bullseye-backports
    sed -i '/bullseye-backports/s/^deb/# deb/' "$f" || true
  fi
done

############################################
# 1. Basic packages
############################################

apt-get update -y
apt-get install -y curl gnupg2 ca-certificates

# Optional: log whether Docker is already present
if command -v docker >/dev/null 2>&1; then
  echo "Docker is already installed on this image; not installing docker.io."
else
  echo "Docker is not installed; skipping automatic docker.io install to avoid containerd.io/containerd conflicts."
fi

# Let a normal user use docker (handy for VS Code, Jupyter, etc.)
DEV_USER=""
if id "jupyter" &>/dev/null; then
  DEV_USER="jupyter"
elif id "ubuntu" &>/dev/null; then
  DEV_USER="ubuntu"
else
  # Fallback: first non-system user (typically uid 1000)
  DEV_USER="$(getent passwd 1000 | cut -d: -f1 || true)"
fi

if [ -n "${DEV_USER}" ]; then
  usermod -aG docker "${DEV_USER}" || true
fi

############################################
# 2. Wait for DLVM's NVIDIA driver install
############################################
# Deep Learning VM images run their own driver installer at first boot.
# We just wait until nvidia-smi works so we don't race it.

for i in {1..30}; do
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver is ready."
    break
  fi
  echo "Waiting for NVIDIA driver to be ready (attempt $i)..."
  sleep 10
done

############################################
# 3. Install NVIDIA Container Toolkit (strict)
############################################
# Only install if it's not already present.

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "Installing NVIDIA Container Toolkit..."

  # Add NVIDIA repo
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update -y
  apt-get install -y nvidia-container-toolkit
fi

# Wire Docker up to use NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

############################################
# 4. Optional sanity checks
############################################

echo "==== Host-level nvidia-smi ===="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "==== Docker + GPU sanity check (non-fatal) ===="
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi || true

echo "==== DLVM CUDA 12.4 (Debian 11) GPU Docker setup finished ===="
