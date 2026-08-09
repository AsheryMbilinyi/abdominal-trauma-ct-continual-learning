#!/usr/bin/env python3
"""
Convert RSNA 2023 DICOM files to PNG for use by the experiment scripts.

Saves three CT-window versions so DI experiments can load pre-windowed PNGs
instead of trying to re-window 8-bit values at runtime.

Input:  $DATA_DIR/train_images/<patient_id>/<series_id>/<instance>.dcm
Output: $DATA_DIR/RSNA2023ProcessedImages/soft/<patient_id>/...   (soft-tissue, default for CI)
        $DATA_DIR/RSNA2023ProcessedImages/brain/<patient_id>/...  (brain window)
        $DATA_DIR/RSNA2023ProcessedImages/lung/<patient_id>/...   (lung window)

Usage:
    python scripts/preprocess_dicom.py
    DATA_DIR=/path/to/data python scripts/preprocess_dicom.py
"""

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

DATA_DIR        = Path(os.environ.get('DATA_DIR', 'data'))
TRAIN_DIR       = DATA_DIR / 'train_images'
OUT_DIR         = DATA_DIR / 'RSNA2023ProcessedImages'
LABELS_CSV      = DATA_DIR / 'train.csv'
MAX_PER_CLASS   = int(os.environ.get('MAX_PER_CLASS', '100'))
MAX_PER_PATIENT = int(os.environ.get('MAX_PER_PATIENT', '20'))
TARGET_SIZE     = (256, 256)

# Three CT windows: (name, WL, WW)
# soft is the default used by CI experiments; brain+lung+soft are the DI task domains
WINDOWS = [
    ('soft',   40,  400),   # WL=40, WW=400 — same as original preprocessing
    ('brain',  40,   80),   # WL=40, WW=80
    ('lung', -600, 1500),   # WL=-600, WW=1500
]


def dicom_to_png(dcm_path, out_path, wl, ww):
    dcm = pydicom.dcmread(str(dcm_path))
    pixels = dcm.pixel_array.astype(np.float32)

    slope     = float(getattr(dcm, 'RescaleSlope',     1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    pixels    = pixels * slope + intercept  # convert to Hounsfield Units

    lo, hi = wl - ww // 2, wl + ww // 2
    pixels = np.clip(pixels, lo, hi)
    pixels = ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)

    if pixels.shape[:2] != TARGET_SIZE:
        pixels = cv2.resize(pixels, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(out_path), pixels)


def main():
    if not LABELS_CSV.exists():
        print(f'ERROR: {LABELS_CSV} not found. Download the data first.')
        sys.exit(1)

    df = pd.read_csv(LABELS_CSV)

    # Handle both old and new CSV column names
    label_col = 'any_injury' if 'any_injury' in df.columns else df.columns[-1]
    pid_col   = 'patient_id' if 'patient_id' in df.columns else df.columns[0]

    pid_label = {int(r[pid_col]): int(r[label_col])
                 for _, r in df[[pid_col, label_col]].dropna().iterrows()}

    by_class = {}
    for pid, lbl in pid_label.items():
        by_class.setdefault(lbl, []).append(pid)

    rng = random.Random(42)
    selected = []
    for lbl, pids in sorted(by_class.items()):
        rng.shuffle(pids)
        chosen = pids[:MAX_PER_CLASS]
        selected.extend(chosen)
        print(f'  Class {lbl}: {len(chosen)} patients selected')

    print(f'\nTotal: {len(selected)} patients  |  max {MAX_PER_PATIENT} slices each')
    print(f'Output base: {OUT_DIR}\n')

    for window_name, wl, ww in WINDOWS:
        window_dir = OUT_DIR / window_name
        window_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n--- Window: {window_name} (WL={wl}, WW={ww}) ---')

        skipped = errors = converted = 0
        for pid in tqdm(selected, desc=f'Patients [{window_name}]'):
            pid_dir = TRAIN_DIR / str(pid)
            if not pid_dir.exists():
                skipped += 1
                continue

            for series_dir in sorted(pid_dir.iterdir()):
                if not series_dir.is_dir():
                    continue

                dcm_files = sorted(series_dir.glob('*.dcm'))[:MAX_PER_PATIENT]
                if not dcm_files:
                    continue

                out_series = window_dir / str(pid) / series_dir.name
                out_series.mkdir(parents=True, exist_ok=True)

                for dcm_path in dcm_files:
                    out_path = out_series / (dcm_path.stem + '.png')
                    if out_path.exists():
                        continue
                    try:
                        dicom_to_png(dcm_path, out_path, wl=wl, ww=ww)
                        converted += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f'\nWarning [{window_name}]: {dcm_path.name}: {e}')

        print(f'  Converted: {converted}  Skipped patients: {skipped}  Errors: {errors}')

    print(f'\nDone. PNGs saved under: {OUT_DIR}')


if __name__ == '__main__':
    main()
