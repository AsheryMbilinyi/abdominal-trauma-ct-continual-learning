#!/bin/bash
#SBATCH --job-name=separate_head
#SBATCH --account=def-a2nyi4
#SBATCH --time=08:00:00               # MultiHead vs SharedHead, 3 seeds, 100 patients/class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
module load python/3.11
module load gcc opencv
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

export DATA_DIR="$SCRATCH/micad/data"
export LOG_DIR="$SCRATCH/micad/logs"
mkdir -p "$LOG_DIR"

# ── run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID started at $(date)"

python src/experiments/separate_head_baseline.py

echo "Job finished at $(date)"

cp "$LOG_DIR"/separate_head_*.{log,csv} "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
