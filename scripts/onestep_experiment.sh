#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SWEEP_ARG="configs/sweep_config_experiment.yaml"

# 1. Extract stdout 和 stderr
OUTPUT=$(wandb sweep "$SWEEP_ARG" 2>&1)

# 2. Find out the word "wandb agent" line 
LINE=$(printf '%s\n' "$OUTPUT" | grep -m1 'wandb agent')

# 3. Use bash parameter expansion, and remain the words "wandb agent ..."
AGENT_CMD=$(printf '%s\n' "$LINE" | sed -E 's/.*(wandb agent .*)$/\1/')

# 4. Run 
CUDA_VISIBLE_DEVICES="$GPU_ID" $AGENT_CMD
