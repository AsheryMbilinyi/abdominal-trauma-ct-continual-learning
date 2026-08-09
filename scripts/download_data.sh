#!/bin/bash
# Download the RSNA 2023 Abdominal Trauma Detection dataset to $SCRATCH.
# Requires the Kaggle API token (~/.kaggle/kaggle.json) to already be in place.
#
# Usage (run from project root, NOT as a SLURM job):
#   bash scripts/download_data.sh
#
# After this finishes, DATA_DIR=$SCRATCH/micad/data should contain:
#   train.csv
#   RSNA2023ProcessedImages/<patient_id>/<series_id>/*.png

set -e

# Pick up KAGGLE_TOKEN if defined in .bashrc but not yet exported
[ -f ~/.bashrc ] && source ~/.bashrc

DEST="$SCRATCH/micad/data"
mkdir -p "$DEST"

echo "=== Downloading RSNA 2023 Abdominal Trauma dataset to $DEST ==="

source "$PWD/venv/bin/activate"
pip install -q kaggle

if [ -z "$KAGGLE_TOKEN" ]; then
    echo "ERROR: KAGGLE_TOKEN is not set. Run: export KAGGLE_TOKEN=your_token"
    exit 1
fi

kaggle competitions download -c rsna-2023-abdominal-trauma-detection -p "$DEST"

echo "Unzipping..."
cd "$DEST"
unzip -q rsna-2023-abdominal-trauma-detection.zip
rm -f rsna-2023-abdominal-trauma-detection.zip

echo ""
echo "=== Download complete ==="
echo "Data directory: $DEST"
echo ""
echo "Next: submit jobs with  bash scripts/submit_all.sh"
