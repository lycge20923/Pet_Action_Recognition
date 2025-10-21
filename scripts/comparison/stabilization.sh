#!/usr/bin/env bash

GPU_ID="$1"         # 第一個參數：GPU ID
DATA_NAME="$2"      # 第二個參數：資料集名稱

# 根據資料集設定對應的路徑
if [ "$DATA_NAME" == "BaboonLand" ]; then
    PATH_TO_DATA="BaboonLand/charades"
elif [ "$DATA_NAME" == "KABR" ]; then
    PATH_TO_DATA="KABR/KABR_files"
else
    echo "❌ Unknown dataset name: $DATA_NAME"
    exit 1
fi

# 執行 Python 腳本
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.data_processing.stabilization \
    --data_dir data/others \
    --seg_dir_name "raw/${PATH_TO_DATA}/dataset/video" \
    --stabilized_dir_name "${DATA_NAME}/stabilized"