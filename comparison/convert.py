#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert your keypoint dataset (annotation.json + *_kp.npy of shape (T, 17, 3))
into NTU-style .npz that CTR-GCN expects.

CTR-GCN expects:
    x_train: (N_train, T, 150)
    y_train: (N_train, num_classes)  # one-hot
    x_test:  (N_test,  T, 150)
    y_test:  (N_test,  num_classes)

Where 150 = 2 (persons) * 25 (joints) * 3 (coords).

Here we assume:
    - Each sample in annotation.json looks like:
        {
            "sample_id": 2245,
            "kp_feature_file": "data/main/train_split/npy/002245_kp.npy",
            "of_feature_file": "...",
            "video_name": "...",
            "source_video_id": 58,
            "action_id": 6,
            "window_index": 1,
            "fold": 1
        }
    - kp_feature_file is (T, V=17, C=3)
    - Only 1 animal per clip (M=1).
    - We pad joints to 25, and pad a 2nd person with all zeros.
"""

import os
import json
import argparse
import numpy as np
import logging

from src.utils.cli_args import DataArguments, ComparisonArguments
from src.utils.logging_utils import setup_logger

def build_x_y(samples, num_classes, target_joints, strict_shape=False):
    """
    Build x, y for a split in CTR-GCN format.

    Input:
        samples: list of dict from annotation.json
        num_classes: number of classes for one-hot
        target_joints: we pad your joints to this number (NTU uses 25)
    Output:
        x: (N, T, 2 * target_joints * 3)  # e.g. (N, T, 150)
        y: (N, num_classes)               # one-hot
    """
    if not samples:
        x = np.zeros((0, 0, target_joints * 3), dtype=np.float32)
        y = np.zeros((0, num_classes), dtype=np.uint8)
        return x, y

    xs = []
    ys = []

    # Sort by sample_id for deterministic ordering
    samples = sorted(samples, key=lambda s: s.get("sample_id", 0))

    for s in samples:
        kp_path = os.path.join("/home/r12922166/Pet_Action_Recognition", s["kp_feature_file"])
        if not os.path.isfile(kp_path):
            raise FileNotFoundError(f"kp_feature_file not found: {kp_path}")

        kp = np.load(kp_path)  # expected shape (T, V, 3)
        if kp.ndim != 3:
            raise ValueError(
                f"kp shape must be 3D (T, V, C), got {kp.shape} for {kp_path}"
            )

        T, V, C = kp.shape
        if strict_shape:
            if C != 3:
                raise ValueError(
                    f"Expected C=3 (x,y,conf or 3D coords), got C={C} for {kp_path}"
                )
            if V > target_joints:
                raise ValueError(
                    f"V={V} > target_joints={target_joints} for {kp_path}"
                )

        # If V < target_joints, pad extra joints with zeros
        if V < target_joints:
            pad_joints = target_joints - V
            pad = np.zeros((T, pad_joints, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad], axis=1)  # (T, target_joints, C)
            V = target_joints
        elif V > target_joints:
            # If not strict, we can simply truncate extra joints
            kp = kp[:, :target_joints, :]
            V = target_joints

        # Flatten joints of person 1: (T, V, C) -> (T, V*C)
        feat = kp.reshape(T, V * C).astype(np.float32)  # (T, target_joints * 3)

        xs.append(feat)

        # Build one-hot label
        label_idx = int(s["action_id"])
        if not (0 <= label_idx < num_classes):
            raise ValueError(
                f"action_id {label_idx} out of range for num_classes={num_classes}"
            )
        y_onehot = np.zeros(num_classes, dtype=np.uint8)
        y_onehot[label_idx] = 1
        ys.append(y_onehot)

    x = np.stack(xs, axis=0)  # (N, T, 2 * target_joints * 3)
    y = np.stack(ys, axis=0)  # (N, num_classes)
    return x, y

def main():
    
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    data_params = DataArguments()
    data_dir = os.path.join(data_params.data_dir, data_params.trainsplit_dir_name)
    annotation_path = os.path.join(data_dir, "annotation_windows_metadata.json")
    num_classes = data_params.num_classes
    num_folds = data_params.num_folds
    target_joints = data_params.num_nodes
    
    comp_paras = ComparisonArguments()
    output_dir = os.path.join(data_params.data_dir, comp_paras.new_data_dir_name)
    
    # load annotation
    with open(annotation_path, "r") as f:
        annotation = json.load(f)
    logger.info(f"Total samples in annotation: {len(annotation)}")
    
    for fold_num in range(num_folds):
        logger.info(f"Start to run for fold num: {fold_num}")
        test_folds = set([fold_num])
        train_folds = set([i for i in range(num_folds) if i != fold_num])
        
        train_samples = [s for s in annotation if int(s["fold"]) in train_folds]
        test_samples = [s for s in annotation if int(s["fold"]) in test_folds]
        if not train_samples:
            message = f"No samples found for train_folds={sorted(train_folds)}"
            logger.error(message)
            raise ValueError(message)
        if not test_samples:
            logger.warning(f"[convert] WARNING: no samples found for test_folds={sorted(test_folds)}")
        logger.info(f"[convert] TRAIN folds {sorted(train_folds)} -> {len(train_samples)} samples")
        logger.info(f"[convert] TEST  folds {sorted(test_folds)}  -> {len(test_samples)} samples")
        logger.info(f"[convert] Using user-specified num_classes={num_classes}")

        # Build train split
        print("Building x_train, y_train...")
        x_train, y_train = build_x_y(
            train_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

        # Build test split
        print("Building x_test, y_test...")
        x_test, y_test = build_x_y(
            test_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        logger.info(f"[convert] x_test shape:  {x_test.shape}, y_test shape:  {y_test.shape}")

        # Ensure output dir exists
        out_dir = os.path.dirname(output_dir)
        if out_dir:
            os.makedirs(output_dir, exist_ok=True)
        npz_file_path = os.path.join(output_dir, f"data_fold_{fold_num}.npz")
        np.savez(
            npz_file_path,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        logger.info(f"[convert] Saving NTU-style npz to: {npz_file_path}")


if __name__ == "__main__":
    main()
