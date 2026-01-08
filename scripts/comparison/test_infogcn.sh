#!/usr/bin/env bash

GPU_ID="$1"
WORK_DIR="$2" 
MODE="$3" #joint, bone
USE_VEL="$4" #True, False
FOLD_NUM="$5"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.infogcn.main \
    --phase test \
    --weights "$WORK_DIR"/best_model.pt \
    --mode "$MODE" \
    --use_vel "$USE_VEL" \
    --fold_num "$FOLD_NUM"

