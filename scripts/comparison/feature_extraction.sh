#!/usr/bin/env bash

GPU_ID="$1"         # 第一個參數：GPU ID
DATASET_NAME="$2"      

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.data_processing.feature_extraction \
    --for_comparison \
    --dataset_name "${DATASET_NAME}"