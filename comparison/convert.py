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
import argparse

from src.utils.cli_args import DataArguments, ComparisonArguments, DEFAULT_SKELETON_LIST
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
        kp_path = os.path.join("./", s["kp_feature_file"])
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

def build_x_y_bone(samples, num_classes, target_joints, strict_shape=False):
    """
    跟 build_x_y 幾乎一樣，但輸出的是 bone features。
    """
    if not samples:
        x = np.zeros((0, 0, target_joints * 3), dtype=np.float32)
        y = np.zeros((0, num_classes), dtype=np.uint8)
        return x, y

    xs = []
    ys = []

    parents = get_parent_indices(target_joints)
    samples = sorted(samples, key=lambda s: s.get("sample_id", 0))

    for s in samples:
        kp_path = os.path.join("/home/r12922166/Pet_Action_Recognition", s["kp_feature_file"])
        if not os.path.isfile(kp_path):
            raise FileNotFoundError(f"kp_feature_file not found: {kp_path}")

        kp = np.load(kp_path)  # (T, V, C)
        if kp.ndim != 3:
            raise ValueError(f"kp shape must be 3D (T, V, C), got {kp.shape} for {kp_path}")

        T, V, C = kp.shape
        if strict_shape:
            if C != 3:
                raise ValueError(f"Expected C=3, got C={C} for {kp_path}")
            if V > target_joints:
                raise ValueError(f"V={V} > target_joints={target_joints} for {kp_path}")

        if V < target_joints:
            pad_joints = target_joints - V
            pad = np.zeros((T, pad_joints, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad], axis=1)
            V = target_joints
        elif V > target_joints:
            kp = kp[:, :target_joints, :]
            V = target_joints

        # 這裡改成 bone
        bone = compute_bone(kp.astype(np.float32), parents)  # (T, V, C)
        feat = bone.reshape(T, V * C).astype(np.float32)

        xs.append(feat)

        label_idx = int(s["action_id"])
        if not (0 <= label_idx < num_classes):
            raise ValueError(f"action_id {label_idx} out of range for num_classes={num_classes}")
        y_onehot = np.zeros(num_classes, dtype=np.uint8)
        y_onehot[label_idx] = 1
        ys.append(y_onehot)

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    return x, y

def build_x_y_vel(samples, num_classes, target_joints, strict_shape=False):
    """
    使用 joint velocity (關節速度) 作為輸出 features。
    """
    if not samples:
        x = np.zeros((0, 0, target_joints * 3), dtype=np.float32)
        y = np.zeros((0, num_classes), dtype=np.uint8)
        return x, y

    xs = []
    ys = []

    samples = sorted(samples, key=lambda s: s.get("sample_id", 0))

    for s in samples:
        kp_path = os.path.join("/home/r12922166/Pet_Action_Recognition", s["kp_feature_file"])
        if not os.path.isfile(kp_path):
            raise FileNotFoundError(f"kp_feature_file not found: {kp_path}")

        kp = np.load(kp_path)  # (T, V, C)
        if kp.ndim != 3:
            raise ValueError(f"kp shape must be 3D (T, V, C), got {kp.shape} for {kp_path}")

        T, V, C = kp.shape
        if strict_shape:
            if C != 3:
                raise ValueError(f"Expected C=3, got C={C} for {kp_path}")
            if V > target_joints:
                raise ValueError(f"V={V} > target_joints={target_joints} for {kp_path}")

        if V < target_joints:
            pad_joints = target_joints - V
            pad = np.zeros((T, pad_joints, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad], axis=1)
            V = target_joints
        elif V > target_joints:
            kp = kp[:, :target_joints, :]
            V = target_joints

        vel = compute_velocity(kp.astype(np.float32))  # (T, V, C)
        feat = vel.reshape(T, V * C).astype(np.float32)

        xs.append(feat)

        label_idx = int(s["action_id"])
        if not (0 <= label_idx < num_classes):
            raise ValueError(f"action_id {label_idx} out of range for num_classes={num_classes}")
        y_onehot = np.zeros(num_classes, dtype=np.uint8)
        y_onehot[label_idx] = 1
        ys.append(y_onehot)

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    return x, y

def build_x_y_bone_vel(samples, num_classes, target_joints, strict_shape=False):
    """
    使用 bone velocity (骨向量速度) 作為輸出 features。
    """
    if not samples:
        x = np.zeros((0, 0, target_joints * 3), dtype=np.float32)
        y = np.zeros((0, num_classes), dtype=np.uint8)
        return x, y

    xs = []
    ys = []

    parents = get_parent_indices(target_joints)
    samples = sorted(samples, key=lambda s: s.get("sample_id", 0))

    for s in samples:
        kp_path = os.path.join("/home/r12922166/Pet_Action_Recognition", s["kp_feature_file"])
        if not os.path.isfile(kp_path):
            raise FileNotFoundError(f"kp_feature_file not found: {kp_path}")

        kp = np.load(kp_path)  # (T, V, C)
        if kp.ndim != 3:
            raise ValueError(f"kp shape must be 3D (T, V, C), got {kp.shape} for {kp_path}")

        T, V, C = kp.shape
        if strict_shape:
            if C != 3:
                raise ValueError(f"Expected C=3, got C={C} for {kp_path}")
            if V > target_joints:
                raise ValueError(f"V={V} > target_joints={target_joints} for {kp_path}")

        if V < target_joints:
            pad_joints = target_joints - V
            pad = np.zeros((T, pad_joints, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad], axis=1)
            V = target_joints
        elif V > target_joints:
            kp = kp[:, :target_joints, :]
            V = target_joints

        bone = compute_bone(kp.astype(np.float32), parents)      # (T, V, C)
        bone_vel = compute_velocity(bone)                        # (T, V, C)
        feat = bone_vel.reshape(T, V * C).astype(np.float32)

        xs.append(feat)

        label_idx = int(s["action_id"])
        if not (0 <= label_idx < num_classes):
            raise ValueError(f"action_id {label_idx} out of range for num_classes={num_classes}")
        y_onehot = np.zeros(num_classes, dtype=np.uint8)
        y_onehot[label_idx] = 1
        ys.append(y_onehot)

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    return x, y

def get_parent_indices(num_joints: int, 
                       bone_pairs=[[0, 1], [0, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10], [4, 14], [14, 15], [15, 16], [4, 11], [11, 12], [12, 13]]):
    """
    num_joints : 關節總數 (V)
    bone_pairs : list of [parent, child] 的 pair，例如：
                 [[0,1], [1,2], [2,3]]
                 
    回傳 parent list，長度 = num_joints
    parent[v] = 該關節的 parent index
                如果無 parent → parent[v] = -1
    """
    parents = [-1] * num_joints  # 預設 root = -1
    
    for parent, child in bone_pairs:
        if 0 <= child < num_joints:
            parents[child] = parent
    
    return parents

def compute_bone(kp: np.ndarray, parents):
    """
    kp: (T, V, C)
    parents: list[int] of length V, parent[v] = parent index or -1

    回傳 bone: (T, V, C)，其中
        bone[..., v, :] = kp[..., v, :] - kp[..., parent[v], :] (若有 parent)
        若 parent[v] == -1，則 bone[..., v, :] = 0
    """
    T, V, C = kp.shape
    bone = np.zeros_like(kp, dtype=np.float32)
    for v in range(V):
        p = parents[v]
        if p < 0 or p >= V:
            # root 或無效 parent → bone 向量設為 0
            continue
        bone[:, v, :] = kp[:, v, :] - kp[:, p, :]
    return bone

def compute_velocity(arr: np.ndarray):
    """
    arr: (T, V, C)  # 可以是 joint 或 bone
    回傳 vel: (T, V, C)
        vel[0]   = 0
        vel[t]   = arr[t] - arr[t-1] for t >= 1
    """
    vel = np.zeros_like(arr, dtype=np.float32)
    vel[1:] = arr[1:] - arr[:-1]
    return vel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--KABR", action="store_true", help="Use KABR dataset")
    temp_args = parser.parse_args()
    
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    
    comp_paras = ComparisonArguments()
    data_params = DataArguments()
    target_joints = data_params.num_nodes
    
    if not temp_args.KABR:
        data_dir = os.path.join(data_params.data_dir, data_params.trainsplit_dir_name)
        annotation_path = os.path.join(data_dir, "annotation_windows_metadata.json")
        num_classes = data_params.num_classes
        num_folds = data_params.num_folds
        output_dir = os.path.join(data_params.data_dir, comp_paras.new_data_dir_name)
        
    else:
        data_dir = comp_paras.other_data_dir_name
        annotation_path = os.path.join(data_dir, "train_split", "annotation_windows_metadata.json")
        num_classes = 8
        num_folds = 1
        output_dir = os.path.join(data_dir, comp_paras.new_data_dir_name)
    
    # load annotation
    with open(annotation_path, "r") as f:
        annotation = json.load(f)
    logger.info(f"Total samples in annotation: {len(annotation)}")
    
    for fold_num in range(num_folds):
        logger.info(f"Start to run for fold num: {fold_num}")
        test_folds = set([fold_num])
        if temp_args.KABR:
            train_folds = set([i for i in range(num_folds + 1) if i != fold_num])
        else:
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
        
        # # for tdgcn
        # y_test_idx = np.where(y_test > 0)[1]
        # # val_sample_txt_path = os.path.join(output_dir, f"val_sample_fold_{fold_num}.txt")
        # # with open(val_sample_txt_path, "w") as f:
        # #     for cls in y_test_idx:
        # #         # cls 是 0-based，所以要 +1，再補成 3 位數
        # #         f.write(f"{cls + 1:03d}\n")

        # logger.info(f"[convert] Saving val_sample txt to: {val_sample_txt_path}")

        # Ensure output dir exists
        out_dir = os.path.dirname(output_dir)
        if out_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1) joint stream 
        joint_npz_file_path = os.path.join(output_dir, f"data_joint_fold_{fold_num}.npz")
        np.savez(
            joint_npz_file_path,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        logger.info(f"[convert] Saving JOINT npz to: {joint_npz_file_path}")

        # 2) bone stream
        print("Building x_train_bone, y_train_bone...")
        x_train_bone, y_train_bone = build_x_y_bone(
            train_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        print("Building x_test_bone, y_test_bone...")
        x_test_bone, y_test_bone = build_x_y_bone(
            test_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        bone_npz_file_path = os.path.join(output_dir, f"data_bone_fold_{fold_num}.npz")
        np.savez(
            bone_npz_file_path,
            x_train=x_train_bone,
            y_train=y_train_bone,
            x_test=x_test_bone,
            y_test=y_test_bone,
        )
        logger.info(f"[convert] Saving BONE npz to: {bone_npz_file_path}")

        # 3) joint velocity stream
        print("Building x_train_vel, y_train_vel...")
        x_train_vel, y_train_vel = build_x_y_vel(
            train_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        print("Building x_test_vel, y_test_vel...")
        x_test_vel, y_test_vel = build_x_y_vel(
            test_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        vel_npz_file_path = os.path.join(output_dir, f"data_joint_vel_fold_{fold_num}.npz")
        np.savez(
            vel_npz_file_path,
            x_train=x_train_vel,
            y_train=y_train_vel,
            x_test=x_test_vel,
            y_test=y_test_vel,
        )
        logger.info(f"[convert] Saving VEL npz to: {vel_npz_file_path}")

        # 4) bone velocity stream
        print("Building x_train_bone_vel, y_train_bone_vel...")
        x_train_bone_vel, y_train_bone_vel = build_x_y_bone_vel(
            train_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        print("Building x_test_bone_vel, y_test_bone_vel...")
        x_test_bone_vel, y_test_bone_vel = build_x_y_bone_vel(
            test_samples,
            num_classes=num_classes,
            target_joints=target_joints,
            strict_shape=True,
        )
        bone_vel_npz_file_path = os.path.join(output_dir, f"data_bone_vel_fold_{fold_num}.npz")
        np.savez(
            bone_vel_npz_file_path,
            x_train=x_train_bone_vel,
            y_train=y_train_bone_vel,
            x_test=x_test_bone_vel,
            y_test=y_test_bone_vel,
        )
        logger.info(f"[convert] Saving BONE_VEL npz to: {bone_vel_npz_file_path}")


if __name__ == "__main__":
    main()
