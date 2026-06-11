#!/bin/bash
#SBATCH --job-name=lora_sweep
#SBATCH --partition="cpu-zen4"
#SBATCH --array=0-8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%A_%a.out

echo "## Loading module"
echo "## Current working directory is $(pwd)"
module load ALICE/default
source .venv/bin/activate

COMMANDS=(
"uv run train_lora_parametric.py --run-name steps_500  --steps 501  --rank 8  --batch-size 16"
"uv run train_lora_parametric.py --run-name steps_5000 --steps 5001 --rank 8  --batch-size 16"
"uv run train_lora_parametric.py --run-name rank_2     --steps 1001 --rank 2  --batch-size 16"
"uv run train_lora_parametric.py --run-name rank_4     --steps 1001 --rank 4  --batch-size 16"
"uv run train_lora_parametric.py --run-name rank_16    --steps 1001 --rank 16 --batch-size 16"
"uv run train_lora_parametric.py --run-name batch_32   --steps 1001 --rank 8  --batch-size 32"
"uv run train_lora_parametric.py --run-name batch_64   --steps 1001 --rank 8  --batch-size 64"
"uv run train_lora_parametric.py --run-name lr_5e5     --steps 1001 --rank 8  --batch-size 16 --lr 5e-5"
"uv run train_lora_parametric.py --run-name lr_5e4     --steps 1001 --rank 8  --batch-size 16 --lr 5e-4"
)

echo "Running task ${SLURM_ARRAY_TASK_ID}"
echo "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
