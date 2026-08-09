#!/bin/bash
#SBATCH --job-name=extract_patients
#SBATCH --account=def-a2nyi4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load python/3.11
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

export ZIP="$SCRATCH/micad/data/rsna-2023-abdominal-trauma-detection.zip"
export OUT="$SCRATCH/micad/data"
export KEEP="$SCRATCH/micad/extract_patients.txt"

echo "Selective extraction started at $(date)"
echo "Zip: $ZIP"
echo "Output: $OUT"

python3 - <<'EOF'
import zipfile, os, sys
from pathlib import Path

zip_path  = os.environ['ZIP']
out_dir   = Path(os.environ['OUT'])
keep_file = os.environ['KEEP']

with open(keep_file) as f:
    keep_pids = {line.strip() for line in f if line.strip()}

print(f'Patients to extract: {len(keep_pids)}')

extracted = skipped = 0
with zipfile.ZipFile(zip_path) as zf:
    all_members = zf.namelist()
    print(f'Total entries in zip: {len(all_members)}')
    for member in all_members:
        parts = Path(member).parts
        # Handle both: train_images/<pid>/... and <topdir>/train_images/<pid>/...
        for i, part in enumerate(parts):
            if part == 'train_images' and i + 1 < len(parts):
                pid = parts[i + 1]
                if pid in keep_pids:
                    zf.extract(member, out_dir)
                    extracted += 1
                    if extracted % 2000 == 0:
                        print(f'  Extracted {extracted} files so far...')
                else:
                    skipped += 1
                break

print(f'Done. Extracted: {extracted}  Skipped: {skipped}')
EOF

echo "Extraction finished at $(date)"
echo "Patient folders now in $OUT/train_images:"
ls "$OUT/train_images" | wc -l
