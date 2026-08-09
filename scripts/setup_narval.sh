#!/bin/bash
# One-time environment setup on Narval.
# Run this ONCE before submitting any jobs:
#   bash scripts/setup_narval.sh
#
# Assumes you are in the project root directory.

set -e

PROJECT_DIR="$PWD"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== Loading modules ==="
module load python/3.11
# OpenCV MUST be loaded before activating the venv (Alliance Canada requirement)
module load gcc opencv

echo "=== Setting up Python virtualenv at $VENV_DIR ==="
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# PyTorch with CUDA 11.8 (matches Narval A100 drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo ""
echo "Next step: download data with  bash scripts/download_data.sh"
