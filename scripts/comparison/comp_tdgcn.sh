#!/usr/bin/env bash

GPU_ID="$1"         # 第一個參數：GPU ID 

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.tdgcn.main \
    --config configs/comp/tdgcn_PetAction.yaml \