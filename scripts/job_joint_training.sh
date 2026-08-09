#!/bin/bash
#SBATCH --job-name=joint_training_ub
#SBATCH --account=def-a2nyi4
#SBATCH --time=04:00:00               # CI + DI joint training, 5 epochs each
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

python src/experiments/joint_training_upper_bound.py

echo "Job finished at $(date)"

cp "$LOG_DIR"/joint_training_upper_bound_*.{log,csv} "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
