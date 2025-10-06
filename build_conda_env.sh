#!/usr/bin/env bash
set -euo pipefail

# Script to create a conda environment with dependencies for finetune_mms_lid.py
# Usage: ./build_conda_env.sh [ENV_NAME]
# Default environment name is mms-lid-finetune

ENV_NAME="${1:-mms-lid-finetune}"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Anaconda or Miniconda before running this script." >&2
  exit 1
fi

# Detect whether we're running inside a conda base environment
# and ensure base is activated to allow environment creation.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME" && {
  echo "[INFO] Removing existing environment: $ENV_NAME"
  conda env remove -y -n "$ENV_NAME"
}

echo "[INFO] Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Prefer conda-forge for better binary coverage on SageMaker
conda config --env --add channels conda-forge
conda config --env --set channel_priority flexible

# Core scientific stack
conda install -y numpy pandas scipy ffmpeg

# Hugging Face stack; install torch via pip to ensure CUDA/cuDNN compatibility on SageMaker DLCs
pip install --upgrade pip
pip install -r requirements.txt

# Optional: accelerate for distributed training on multi-GPU SageMaker instances
pip install accelerate

echo "[INFO] Environment '$ENV_NAME' is ready. Activate it with:"
echo "       conda activate $ENV_NAME"
echo "[INFO] To launch fine-tuning:"
echo "       python finetune_mms_lid.py --help"
