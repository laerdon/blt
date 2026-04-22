#!/bin/bash
# sets up blt on a fresh ubuntu 22.04 box with an nvidia gpu.
# idempotent: safe to rerun. skips steps that are already done.
#
# requirements:
#   - sudo access (for apt and cuda toolkit install)
#   - nvidia driver already present and supporting cuda 12.x (check nvidia-smi)
#   - run from the repo root (the directory containing pyproject.toml)

set -euo pipefail

start_time=$(date +%s)

# where to download the cuda runfile (needs ~5gb free; /tmp often too small)
CUDA_DOWNLOAD_DIR="${CUDA_DOWNLOAD_DIR:-/mnt}"
CUDA_INSTALL_PATH="/usr/local/cuda-12.1"
CUDA_RUNFILE="cuda_12.1.1_530.30.02_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${CUDA_RUNFILE}"

# gpu arch. a6000=8.6, a100=8.0, rtx 4090=8.9, h100=9.0. adjust if needed.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "[1/7] installing system build dependencies"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  gcc-12 \
  g++-12 \
  ninja-build \
  wget \
  curl \
  git

# point gcc/g++ at version 12 for the cuda build. cuda 12.1 supports gcc <= 12.
export CC=gcc-12
export CXX=g++-12

echo "[2/7] checking cuda 12.1 toolkit"
if [ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]; then
  echo "installing cuda 12.1 toolkit to ${CUDA_INSTALL_PATH}"
  mkdir -p "${CUDA_DOWNLOAD_DIR}"
  if [ ! -f "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" ]; then
    wget --continue -O "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" "${CUDA_URL}"
  fi
  mkdir -p "${CUDA_DOWNLOAD_DIR}/cuda_tmp"
  sudo sh "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" \
    --silent \
    --toolkit \
    --toolkitpath="${CUDA_INSTALL_PATH}" \
    --override \
    --tmpdir="${CUDA_DOWNLOAD_DIR}/cuda_tmp"
  # the installer sometimes leaves things root-only readable
  sudo chmod -R a+rX "${CUDA_INSTALL_PATH}"
else
  echo "cuda 12.1 already installed at ${CUDA_INSTALL_PATH}"
fi

export CUDA_HOME="${CUDA_INSTALL_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "nvcc version:"
nvcc --version

echo "[3/7] installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "[4/7] creating venv"
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo "venv python: $(which python)"

echo "[5/7] installing pre_build group (torch, setuptools, ninja)"
# pre_build must land before compile_xformers so the xformers build sees torch.
uv pip install --group pre_build --no-build-isolation

echo "[6/7] building xformers from source with cuda"
# flags consumed by xformers' setup.py during the compile step
export FORCE_CUDA=1
export XFORMERS_BUILD_TYPE=Release
export MAX_JOBS="${MAX_JOBS:-4}"

uv pip install --group compile_xformers --no-build-isolation

echo "[7/7] syncing remaining project dependencies"
uv sync

echo "verification:"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -m xformers.info | grep -E "build.cuda_version|TORCH_CUDA_ARCH_LIST|cutlassF:|cutlassB:|fa2F|triton_splitK" || true

end_time=$(date +%s)
elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "done in ${elapsed_minutes} minutes"
echo "remember to 'source .venv/bin/activate' in new shells"
echo "and optionally add the cuda env vars to ~/.bashrc:"
echo "  export CUDA_HOME=${CUDA_INSTALL_PATH}"
echo "  export PATH=${CUDA_INSTALL_PATH}/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=${CUDA_INSTALL_PATH}/lib64:\$LD_LIBRARY_PATH"
