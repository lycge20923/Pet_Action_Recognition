#!/usr/bin/env bash

GPU_ID="$1"
FOLD_NUM="$2"
MODE="$3"
USE_VEL="$4"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.infogcn.main \
    --fold_num "$FOLD_NUM" \
    --mode "$MODE" \
    --use_vel "$USE_VEL"
