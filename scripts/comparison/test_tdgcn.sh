#!/usr/bin/env bash

GPU_ID="$1"
WORK_DIR="$2" 

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m comparison.tdgcn.main \
  --config "$WORK_DIR"/config.yaml \
  --phase test \
  --save-score True \
  --weights "$WORK_DIR"/best_model.pt 