#!/usr/bin/env bash

GPU_ID="$1"
FOLD_NUM="$2"
JOINT_DIR="$3"
JOINTV_DIR="$4"
BONE_DIR="$5"
BONEV_DIR="$6"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.CTRGCN.ensemble \
    --fold_num "$FOLD_NUM" \
    --joint-dir "$JOINT_DIR" \
    --joint-motion-dir "$JOINTV_DIR" \
    --bone-dir "$BONE_DIR" \
    --bone-motion-dir "$BONEV_DIR"