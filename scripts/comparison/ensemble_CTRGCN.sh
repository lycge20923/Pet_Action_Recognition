#!/usr/bin/env bash

GPU_ID="$1"         # 第一個參數：GPU ID 
JOINT_DIR="$2"
JOINTV_DIR="$3"
BONE_DIR="$4"
BONEV_DIR="$5"


CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.CTRGCN.ensemble \
    --joint-dir "$JOINT_DIR" \
    --joint-motion-dir "$JOINTV_DIR" \
    --bone-dir "$BONE_DIR" \
    --bone-motion-dir "$BONEV_DIR"