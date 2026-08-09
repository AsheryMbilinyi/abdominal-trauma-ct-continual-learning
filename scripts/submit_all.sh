#!/bin/bash
# Submit all 4 experiment jobs to the Narval scheduler.
# Run from the project root after setup_narval.sh and download_data.sh.
#
#   bash scripts/submit_all.sh
#
# All 4 jobs are independent and run in parallel.
# Check status with:  squeue -u $USER

set -e

cd "$(dirname "$0")/.."   # project root

echo "Submitting jobs..."

JID1=$(sbatch --parsable scripts/job_multi_seed_ci.sh)
echo "  multi_seed_ci        → job $JID1"

JID2=$(sbatch --parsable scripts/job_joint_training.sh)
echo "  joint_training_ub    → job $JID2"

JID3=$(sbatch --parsable scripts/job_separate_head.sh)
echo "  separate_head        → job $JID3"

JID4=$(sbatch --parsable scripts/job_near_chance.sh)
echo "  near_chance          → job $JID4"

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Logs will appear in \$SCRATCH/micad/logs/ and be copied back to logs/ on completion."
