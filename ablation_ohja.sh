#!/usr/bin/env bash

set -e

PYTHON=python
SCRIPT=train_lora_ohja_parametric.py

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
# Quick sanity
# --------------------------------------------------

run_exp \
    --run-name sanity_100 \
    --steps 101 \
    --rank 8 \
    --batch-size 1

# --------------------------------------------------
# Baseline
# --------------------------------------------------

run_exp \
    --run-name baseline \
    --steps 1001 \
    --rank 8 \
    --batch-size 1

# --------------------------------------------------
# Rank ablation
# --------------------------------------------------

run_exp \
    --run-name rank_2 \
    --steps 1001 \
    --rank 2 \
    --batch-size 1

run_exp \
    --run-name rank_4 \
    --steps 1001 \
    --rank 4 \
    --batch-size 1

run_exp \
    --run-name rank_16 \
    --steps 1001 \
    --rank 16 \
    --batch-size 1

# --------------------------------------------------
# Learning rate ablation
# --------------------------------------------------

run_exp \
    --run-name lr_5e5 \
    --steps 1001 \
    --rank 8 \
    --batch-size 1 \
    --lr 5e-5

run_exp \
    --run-name lr_5e4 \
    --steps 1001 \
    --rank 8 \
    --batch-size 1 \
    --lr 5e-4

# --------------------------------------------------
# Long run LAST
# --------------------------------------------------

run_exp \
    --run-name steps_5000 \
    --steps 5001 \
    --rank 8 \
    --batch-size 1

echo ""
echo "===== ALL EXPERIMENTS COMPLETED $(date) ====="