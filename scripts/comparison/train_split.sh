#!/usr/bin/env bash

DATASET_NAME="$1"      # 第二個參數：資料集名稱

# 執行 Python 腳本
python -m src.data_processing.train_split \
    --for_comparison \
    --dataset_name "${DATASET_NAME}"