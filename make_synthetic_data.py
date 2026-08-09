# Generates synthetic CT-like data for smoke-testing the pipeline.
# Creates the exact folder structure and CSV the experiments expect,
# using random-noise grayscale images instead of real CT scans.
#
# Run:  python make_synthetic_data.py
# Then: export DEBUG_RUN=1 && python src/experiments/multi_seed_ci.py

import random
from pathlib import Path

import numpy as np
import cv2

DATA_ROOT  = Path('data/RSNA2023ProcessedImages')
LABELS_CSV = Path('data/train.csv')

N_PATIENTS_PER_CLASS = 10   # 10 healthy + 10 injured
N_SERIES_PER_PATIENT = 1
N_SLICES_PER_SERIES  = 8
IMAGE_SIZE           = (256, 256)

random.seed(0)
np.random.seed(0)

DATA_ROOT.mkdir(parents=True, exist_ok=True)

rows = []
patient_id = 10000

for label in [0, 1]:                              # 0 = healthy, 1 = injured
    for _ in range(N_PATIENTS_PER_CLASS):
        patient_dir = DATA_ROOT / str(patient_id)
        series_dir  = patient_dir / '99001'
        series_dir.mkdir(parents=True, exist_ok=True)

        for s in range(N_SLICES_PER_SERIES):
            # Random noise image that vaguely resembles a CT slice
            img = np.random.randint(0, 256, IMAGE_SIZE, dtype=np.uint8)
            # Add a soft circular blob so it looks slightly structured
            cx, cy = IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2
            Y, X = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < (80 + label * 20) ** 2
            img[mask] = np.clip(img[mask].astype(int) + 40, 0, 255).astype(np.uint8)

            cv2.imwrite(str(series_dir / f'{s:03d}.png'), img)

        rows.append({'patient_id': patient_id, 'any_injury': label})
        patient_id += 1

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(LABELS_CSV, index=False)

print(f'Created {len(rows)} patients ({N_PATIENTS_PER_CLASS} per class)')
print(f'Each patient: {N_SLICES_PER_SERIES} slices')
print(f'Images in   : {DATA_ROOT}')
print(f'Labels in   : {LABELS_CSV}')
print()
print('Ready for smoke test:')
print('  export DEBUG_RUN=1')
print('  python src/experiments/multi_seed_ci.py')
