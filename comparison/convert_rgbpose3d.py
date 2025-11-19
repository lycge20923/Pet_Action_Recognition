#!/usr/bin/env python
import os
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert custom pet dataset into NTU-style ann_file + videos for PySkl / RGBPoseConv3D"
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default="data/main/train_split/annotation_windows_metadata.json",
        help="Path to your annotation JSON file",
    )
    parser.add_argument(
        "--video_output_root",
        type=str,
        default="data/main/comp_data_rgbpose3d/PetAction_videos",
        help="Directory to save per-sample mp4 clips (for RGB stream).",
    )
    parser.add_argument(
        "--pkl_output_root",
        type=str,
        default="data/main/comp_data_rgbpose3d/PetAction_ann",
        help="Directory to save NTU-style skeleton ann pkl files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="FPS used when writing mp4 clips.",
    )
    parser.add_argument(
        "--export_videos",
        action="store_true",
        help="If set, export mp4 videos from rgb_feature_file.",
    )
    parser.add_argument(
        "--no_export_videos",
        dest="export_videos",
        action="store_false",
        help="Disable exporting videos.",
    )
    parser.set_defaults(export_videos=True)
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_annotation(annotation_json):
    with open(annotation_json, "r") as f:
        data = json.load(f)
    return data


def rgb_npy_to_mp4(rgb_npy_path, dst_path, fps=25):
    """
    rgb_npy shape expected: (T, 1, 3, H, W)
    We convert it to (T, H, W, 3) BGR and write as mp4 using OpenCV.
    """
    rgb = np.load(rgb_npy_path)  # (T, 1, 3, H, W)
    if rgb.ndim != 5:
        raise ValueError(f"Unexpected rgb shape {rgb.shape} in {rgb_npy_path}")

    T, M, C, H, W = rgb.shape
    if M != 1 or C != 3:
        raise ValueError(f"Expected shape (T,1,3,H,W), got {rgb.shape} in {rgb_npy_path}")

    # Remove person dimension: (T, 3, H, W)
    rgb_clip = rgb[:, 0]  # (T, 3, H, W)

    # Convert to (H, W, 3) and to uint8 BGR
    # 假設目前是 [0,1] 或 [0,255]，這邊保守處理一下
    # 如果你確定是 [0,255] uint8，可以簡化。
    if rgb_clip.dtype != np.uint8:
        # 嘗試 clip 到 [0,1] 再放大到 [0,255]
        rgb_clip = np.clip(rgb_clip, 0.0, 1.0) * 255.0
    rgb_clip = rgb_clip.astype(np.uint8)

    # OpenCV 期待 BGR 順序
    # rgb_clip: (T, 3, H, W) -> (T, H, W, 3)
    rgb_frames = np.transpose(rgb_clip, (0, 2, 3, 1))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(os.path.dirname(dst_path))
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (W, H))

    for i in range(T):
        frame_rgb = rgb_frames[i]  # (H, W, 3), RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


def build_ntu_style_annotations(samples, img_h=256, img_w=256):
    """
    將你的 annotation list 轉成 NTU-style 結構：
    {
        'split': {
            'xsub_train': [...frame_dir...],
            'xsub_val':   [...frame_dir...],
        },
        'annotations': [ {frame_dir, total_frames, img_shape, original_shape, label, keypoint, keypoint_score}, ...]
    }

    我們會為每個 sample 建一筆 annotation，frame_dir = f"{sample_id:06d}"
    """
    annotations = []
    frame_dirs = []
    folds = []
    for s in tqdm(samples, desc="Building NTU-style annotations"):
        sample_id = int(s["sample_id"])
        filename = f"{sample_id:06d}.mp4"

        kp_path = s["kp_feature_file"]
        kp = np.load(kp_path)  # (T, V, 3)
        if kp.ndim != 3 or kp.shape[2] != 3:
            raise ValueError(f"Unexpected kp shape {kp.shape} in {kp_path}")

        T, V, C = kp.shape
        xy = kp[..., :2]      # (T, V, 2)
        xy = xy * 256.0
        score = kp[..., 2]    # (T, V)

        # Add person dimension: M = 1
        keypoint = xy[None, ...]         # (1, T, V, 2)
        keypoint_score = score[None, ...]  # (1, T, V)

        action_id = int(s["action_id"])
        fold = int(s["fold"])

        ann = dict(
            filename=filename,
            total_frames=T,
            img_shape=(img_h, img_w),
            original_shape=(img_h, img_w),
            label=action_id,
            keypoint=keypoint.astype(np.float32),
            keypoint_score=keypoint_score.astype(np.float32),
            # 保留 fold, video_name 等資訊方便 debug，
            # PySkl 只會用到已知 keys，其他 keys 會被忽略。
            fold=fold,
            video_name=s.get("video_name", ""),
            source_video_id=s.get("source_video_id", None),
            window_index=s.get("window_index", None),
        )

        annotations.append(ann)
        frame_dirs.append(filename)
        folds.append(fold)

    return annotations, frame_dirs, folds


def save_fold_pkls(annotations, frame_dirs, folds, output_root):
    """
    依據 folds 產生每個 fold 一個 pkl：
    - 某一 fold = val，其餘 fold = train
    - 結構：
      {
        'split': {
            'xsub_train': [...],
            'xsub_val': [...]
        },
        'annotations': annotations
      }
    """
    ensure_dir(output_root)
    unique_folds = sorted(set(folds))
    print(f"Found folds: {unique_folds}")

    # 建立 frame_dir -> fold 的查表
    fd2fold = {fd: f for fd, f in zip(frame_dirs, folds)}

    for val_fold in unique_folds:
        train_list = [fd for fd in frame_dirs if fd2fold[fd] != val_fold]
        val_list = [fd for fd in frame_dirs if fd2fold[fd] == val_fold]

        data_dict = {
            "split": {
                "xsub_train": train_list,
                "xsub_val": val_list,
            },
            "annotations": annotations,
        }

        out_path = os.path.join(output_root, f"pet_hrnet_fold{val_fold}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Saved fold {val_fold} pkl to {out_path}")


def main():
    args = parse_args()
    ensure_dir(args.video_output_root)
    ensure_dir(args.pkl_output_root)

    samples = load_annotation(args.annotation_json)

    # 1. 先建立 NTU-style annotations（skeleton 部分）
    annotations, frame_dirs, folds = build_ntu_style_annotations(
        samples, img_h=256, img_w=256
    )

    # 2. （選擇性）為每個 sample 匯出 mp4 clip (RGB stream)
    if args.export_videos:
        print("Exporting mp4 videos from rgb_feature_file ...")
        frame_dir_to_sample = defaultdict(list)
        for s in samples:
            sample_id = int(s["sample_id"])
            frame_dir = f"{sample_id:06d}"
            frame_dir_to_sample[frame_dir].append(s)

        for s in tqdm(samples, desc="Writing videos"):
            sample_id = int(s["sample_id"])
            frame_dir = f"{sample_id:06d}"
            rgb_path = s["rgb_feature_file"]
            dst_path = os.path.join(args.video_output_root, frame_dir + ".mp4")
            if os.path.exists(dst_path):
                # 已經寫過就略過（避免重複寫）
                continue
            rgb_npy_to_mp4(rgb_path, dst_path, fps=args.fps)

    # 3. 根據 fold 產生多個 pkl（每個 fold 一個）
    save_fold_pkls(annotations, frame_dirs, folds, args.pkl_output_root)


if __name__ == "__main__":
    main()
