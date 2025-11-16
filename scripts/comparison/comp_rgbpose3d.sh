#!/usr/bin/env bash

GPU_ID="$1"

# RGB
CUDA_VISIBLE_DEVICES="$GPU_ID" bash comparison/pyskl/tools/dist_train.sh configs/comp/PetAction_rgb_only.py 1 \
    --validate --test-last --test-best --seed 42

# # Joints
# CUDA_VISIBLE_DEVICES="$GPU_ID" bash comparison/pyskl/tools/dist_train.sh configs/comp/PetAction_pose_only.py 1 \
#     --validate --test-last --test-best --seed 42