#!/usr/bin/env bash

set -e

PYTHON=python
SCRIPT=train_lora_parametric.py

echo "===== START $(date) ====="

run_exp () {
    echo ""
    echo "=================================================="
    echo "RUNNING: $*"
    echo "START: $(date)"
    echo "=================================================="

    $PYTHON $SCRIPT "$@"

    echo "FINISHED: $(date)"
}

# --------------------------------------------------
# 500 steps (quick sanity)
# --------------------------------------------------

run_exp \
    --run-name steps_500 \
    --steps 500 \
    --rank 8 \
    --batch-size 16

# --------------------------------------------------
# Baseline
# --------------------------------------------------

run_exp \
    --run-name baseline \
    --steps 1000 \
    --rank 8 \
    --batch-size 16

# --------------------------------------------------
# Rank ablation
# --------------------------------------------------

run_exp \
    --run-name rank_2 \
    --steps 1000 \
    --rank 2 \
    --batch-size 16

run_exp \
    --run-name rank_4 \ #i arrive till herer
    --steps 1000 \
    --rank 4 \
    --batch-size 16

run_exp \
    --run-name rank_16 \
    --steps 1000 \
    --rank 16 \
    --batch-size 16

# --------------------------------------------------
# Batch size ablation
# --------------------------------------------------

run_exp \
    --run-name batch_32 \
    --steps 1000 \
    --rank 8 \
    --batch-size 32

run_exp \
    --run-name batch_64 \
    --steps 1000 \
    --rank 8 \
    --batch-size 64

# --------------------------------------------------
# Learning rate ablation
# --------------------------------------------------

run_exp \
    --run-name lr_5e5 \
    --steps 1000 \
    --rank 8 \
    --batch-size 16 \
    --lr 5e-5

run_exp \
    --run-name lr_5e4 \
    --steps 1000 \
    --rank 8 \
    --batch-size 16 \
    --lr 5e-4

# --------------------------------------------------
# Long run LAST
# --------------------------------------------------

run_exp \
    --run-name steps_5000 \
    --steps 5000 \
    --rank 8 \
    --batch-size 16

echo ""
echo "===== ALL EXPERIMENTS COMPLETED $(date) ====="