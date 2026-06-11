#!/bin/bash
#SBATCH --job-name=ohja_sweep
#SBATCH --partition="cpu-zen4"
#SBATCH --array=0-8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out

echo "## Loading module"
echo "## Current working directory is $(pwd)"
module load ALICE/default
source .venv/bin/activate

set -e
export PYTHONUNBUFFERED=1

COMMANDS=(
    "python -u train_lora_ohja_parametric.py --run-name sanity_100 --steps 101  --rank 8  --batch-size 1"
    "python -u train_lora_ohja_parametric.py --run-name baseline   --steps 1001 --rank 8  --batch-size 1"
    "python -u train_lora_ohja_parametric.py --run-name rank_2     --steps 1001 --rank 2  --batch-size 1"
    "python -u train_lora_ohja_parametric.py --run-name rank_4     --steps 1001 --rank 4  --batch-size 1"
    "python -u train_lora_ohja_parametric.py --run-name rank_16    --steps 1001 --rank 16 --batch-size 1"
    "python -u train_lora_ohja_parametric.py --run-name lr_5e5     --steps 1001 --rank 8  --batch-size 1 --lr 5e-5"
    "python -u train_lora_ohja_parametric.py --run-name lr_5e4     --steps 1001 --rank 8  --batch-size 1 --lr 5e-4"
    "python -u train_lora_ohja_parametric.py --run-name steps_5000 --steps 5001 --rank 8  --batch-size 1"
)

echo "===== START $(date) ====="
echo "RUNNING: ${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

echo "===== DONE $(date) ====="

