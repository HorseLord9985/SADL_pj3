#!/bin/bash -l
#SBATCH --job-name=test_gpu
#SBATCH --mail-user="<your-email-address>"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:30:00
#SBATCH --partition=testing
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"

echo "## Loading module"
module load ALICE/default
source .venv/bin/activate

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

echo "## Running test"
uv run train_lora_parametric.py --run-name baseline --steps 1000 --rank 8 --batch-size 16

echo "## Test finished. Goodbye"
