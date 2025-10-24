#!/usr/bin/env bash

GPU_ID="$1"         # 第一個參數：GPU ID
DATA_NAME="$2"      # 第二個參數：資料集名稱 

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.data_processing.stabilization \
    --for_comparison \
    --dataset_name "${DATASET_NAME}"