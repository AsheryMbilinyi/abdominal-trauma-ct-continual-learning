#!/bin/bash
#SBATCH --job-name=preprocess_dicom
#SBATCH --account=def-a2nyi4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load python/3.11
module load gcc opencv
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

export DATA_DIR="$SCRATCH/micad/data"
export MAX_PER_CLASS=100
export MAX_PER_PATIENT=20

cd "$SLURM_SUBMIT_DIR"
echo "Preprocessing started at $(date)"
python scripts/preprocess_dicom.py
echo "Preprocessing finished at $(date)"
