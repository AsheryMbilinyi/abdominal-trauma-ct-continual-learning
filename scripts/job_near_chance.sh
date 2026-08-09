#!/bin/bash
#SBATCH --job-name=near_chance
#SBATCH --account=def-a2nyi4
#SBATCH --time=12:00:00               # CI + DI near-chance diagnostic scan, 100 patients/class
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

# Optional: lower the near-chance threshold (default 55%)
# export CHANCE_THRESHOLD=60.0

# ── run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID started at $(date)"

python src/experiments/near_chance_analysis.py

echo "Job finished at $(date)"

cp "$LOG_DIR"/near_chance_analysis_*.{log,csv} "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
