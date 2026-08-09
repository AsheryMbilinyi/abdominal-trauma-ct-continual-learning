#!/bin/bash
# Download the RSNA 2023 Abdominal Trauma Detection dataset into DATA_DIR.
# Requires the Kaggle API token (~/.kaggle/kaggle.json) to already be in place,
# or a KAGGLE_TOKEN environment variable set beforehand.
#
# Usage (run from project root):
#   DATA_DIR=./data bash scripts/download_data.sh
#
# After this finishes, $DATA_DIR should contain:
#   train.csv
#   train_images/<patient_id>/<series_id>/*.dcm

set -e

DEST="${DATA_DIR:-data}"
mkdir -p "$DEST"

echo "=== Downloading RSNA 2023 Abdominal Trauma dataset to $DEST ==="

pip install -q kaggle

if [ -z "$KAGGLE_TOKEN" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: No Kaggle credentials found."
    echo "Either set KAGGLE_TOKEN or place your API token at ~/.kaggle/kaggle.json"
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
echo "Next: convert DICOM to PNG with  DATA_DIR=$DEST python scripts/preprocess_dicom.py"
