#!/usr/bin/env bash

GPU_ID="$1"
FOLD_NUM="$2"
JOINT_SCORE="$3"
JOINTV_SCORE="$4"
BONE_SCORE="$5"
BONEV_SCORE="$6"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.tdgcn.ensemble \
    --fold_num "$FOLD_NUM" \
    --joint_Score "$JOINT_SCORE" \
    --jointmotion_Score "$JOINTV_SCORE" \
    --bone_Score "$BONE_SCORE" \
    --bonemotion_Score "$BONEV_SCORE"