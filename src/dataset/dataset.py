import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from ..utils.cli_args import DataArguments, ModelArguments, AugmentationArguments
from .aug_func import *

class KpOfDataset(Dataset):
    def __init__(self,
                 data_params: DataArguments,
                 aug_params: AugmentationArguments,
                 model_params: ModelArguments,
                 istrain: bool = True,
                 fold_num: int = 0):
        super().__init__()
        self.data_params = data_params
        self.aug_params = aug_params
        self.model_params = model_params
        self.kps_and_flow = model_params.add_I3D_branch or (model_params.gcn_model_name == "degcn" and model_params.degcn_add_of_A)
        self.only_flow = model_params.only_I3D_branch
        # self.extend_flow_to_kps = model_params.gcn_model_name == "degcn" and model_params.degcn_two_streams

        self.datatype = "train" if istrain else "val"
        # Adjust trainsplit_dir_name if it's not in your data_params
        trainsplit_dir_name = getattr(data_params, 'trainsplit_dir_name', 'train_split')
        annotation_path = os.path.join(self.data_params.data_dir, trainsplit_dir_name, "annotation_windows_metadata.json")
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        if istrain:
            self.annotations = [sample for sample in self.annotations if sample['fold'] != fold_num]
        else:
            self.annotations = [sample for sample in self.annotations if sample['fold'] == fold_num]
        
        self.T = self.data_params.num_samples
        self.num_joints = self.data_params.num_nodes
        self.num_coords = self.data_params.num_coords # Should be 3 (x,y,conf)
        
        # self._load_annotation() # This might call _generate_keypoints                

    def extract_local_flow_stats(self, keypoints: np.ndarray,
                                optical_flows: np.ndarray,
                                block_size: int) -> np.ndarray:
        """
        Args:
            keypoints:      (T, J, 3) – normalized (x, y, conf), but x/y may be outside [0,1].
                            If [x,y,conf] == [0,0,0], treat as missing and skip.
            optical_flows:  (2, T, H, W) – flow u/v.
            block_size:     int – patch side length.

        Returns:
            features: (T, J, 7) – concatenation of (x, y, conf) and (mean_u, mean_v, std_u, std_v).
        """
        T, J, _ = keypoints.shape
        _, _, H, W = optical_flows.shape

        stats = np.zeros((T, J, 4), dtype=np.float32)
        half = block_size // 2

        for t in range(T):
            fx = optical_flows[0, t]  # (H, W)
            fy = optical_flows[1, t]  # (H, W)

            for j in range(J):
                x_norm, y_norm, conf = keypoints[t, j]

                # 1) 如果没有检测到(conf==0 且 x,y=0)，跳过
                if conf == 0 and x_norm == 0 and y_norm == 0:
                    continue

                # 2) clamp 坐标到 [0,1]
                x_clamped = min(max(x_norm, 0.0), 1.0)
                y_clamped = min(max(y_norm, 0.0), 1.0)

                # 3) 转到像素坐标
                cx = int(round(x_clamped * (W - 1)))
                cy = int(round(y_clamped * (H - 1)))

                # 4) 取 patch
                x1 = max(cx - half, 0)
                x2 = min(cx + half + 1, W)
                y1 = max(cy - half, 0)
                y2 = min(cy + half + 1, H)

                patch_x = fx[y1:y2, x1:x2]
                patch_y = fy[y1:y2, x1:x2]

                # 5) 计算均值和标准差
                stats[t, j, 0] = patch_x.mean()
                stats[t, j, 1] = patch_y.mean()
                stats[t, j, 2] = patch_x.std()
                stats[t, j, 3] = patch_y.std()

        # 拼接原始 keypoints[x,y,conf] 和 stats → (T, J, 7)
        features = np.concatenate([keypoints[..., :3], stats], axis=2)
        return features
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load initial data from annotations
        # And 'action_id' is the label
        # load keypoints from the .npz window file
        feat_path = self.annotations[index]["feature_file"]
        data = np.load(feat_path)
        
        # --- optical flows & keypoints ---
        if self.kps_and_flow or self.only_flow: # or self.extend_flow_to_kps:
            optical_flows_np = data["optical_flows"].astype(np.float32) # T, 1, 2, H, W
            optical_flows_np = optical_flows_np.squeeze(axis=1) # T, 2, H, W
            optical_flows_np = optical_flows_np.transpose(1, 0, 2, 3) # 2, T, H, W
        else:
            optical_flows_np = None
        if not self.only_flow:
            keypoints_np = data["keypoints"].astype(np.float32)   # shape (T, V, 3)
        else:
            keypoints_np = None 
            
        # --- augmentation --- 
        if self.aug_params.augment and self.datatype == "train":
            # rotation
            if random.random() < 0.5:
                angle = random.uniform(-self.aug_params.rot_max, self.aug_params.rot_max)
                keypoints_np, optical_flows_np = rotate(
                    keypoints_np,
                    optical_flows_np,
                    angle,
                    self.aug_params.valid_kpt_confidence_thresh
                )
            # scale
            if random.random() < 0.5:
                sf = random.uniform(self.aug_params.scale_min, self.aug_params.scale_max)
                keypoints_np, optical_flows_np = scale(
                    keypoints_np, optical_flows_np,
                    sf,
                    self.aug_params.valid_kpt_confidence_thresh
                )
            # translate
            if random.random() < 0.5:
                tx = random.uniform(-self.aug_params.trans_max, self.aug_params.trans_max)
                ty = random.uniform(-self.aug_params.trans_max, self.aug_params.trans_max)
                keypoints_np, optical_flows_np = translate(
                    keypoints_np, optical_flows_np, tx, ty
                )
            # shear
            if random.random() < 0.3:
                sf = random.uniform(-self.aug_params.shear_max, self.aug_params.shear_max)
                sx = sf if random.random() < 0.5 else 0.0
                sy = sf if random.random() < 0.5 else 0.0
                keypoints_np, optical_flows_np = shear(
                    keypoints_np, optical_flows_np, sx, sy,
                    self.aug_params.valid_kpt_confidence_thresh
                )
            # horizontal flip
            if random.random() < self.aug_params.flow_hflip_prob:
                keypoints_np, optical_flows_np = hflip(
                    keypoints_np, optical_flows_np
                )
            # frame drop
            if random.random() < self.aug_params.frame_drop_prob:
                keypoints_np, optical_flows_np = frame_drop(
                    keypoints_np, optical_flows_np,
                    self.aug_params.frame_drop_prob
                )
            # temporal jitter
            if random.random() < self.aug_params.temporal_jitter_prob:
                keypoints_np, optical_flows_np = temporal_jitter(
                    keypoints_np, optical_flows_np,
                    self.aug_params.temporal_jitter_prob
                )
            # flow-only noise/occlusion/blur
            if optical_flows_np is not None:
                if random.random() < self.aug_params.flow_noise_std:
                    optical_flows_np = add_flow_noise(
                        optical_flows_np,
                        self.aug_params.flow_noise_std
                    )
                if random.random() < self.aug_params.flow_occl_ratio:
                    optical_flows_np = random_flow_occlusion(
                        optical_flows_np,
                        self.aug_params.flow_occl_ratio
                    )
                if random.random() < 0.5:
                    optical_flows_np = gaussian_blur_flow(
                        optical_flows_np,
                        self.aug_params.flow_blur_ksize,
                        self.aug_params.flow_blur_sigma
                    )
            
        # --- transform to tensor --- 
        # label
        label = torch.tensor(self.annotations[index]["action_id"], dtype=torch.long)
        
        if keypoints_np is not None:
            keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
            # Permute to (C, T, V) -> Channels (coords), Time (frames), Vertices (joints)
            keypoints_tensor = keypoints_tensor.permute(2, 0, 1).contiguous()
        else:
            keypoints_tensor = None
        if optical_flows_np is not None:
            optical_flows_tensor = torch.from_numpy(optical_flows_np).float()
        else:
            optical_flows_tensor = None

        return keypoints_tensor, optical_flows_tensor, label

class SiameseKpOfDataset(Dataset):
    """
    Wrapper to produce pairs for joint CrossEntropy + Contrastive training.
    Returns:
      (x1, x2), (lab1, lab2), y
      lab1, lab2: labels for CE loss
      y: binary (0: same, 1: different) for ContrastiveLoss
    """
    def __init__(self, base_dataset: KpOfDataset, data_args:DataArguments):
        self.base = base_dataset
        # Build mapping label -> indices for sampling
        self.label_to_indices = {}
        with open(os.path.join(data_args.data_dir, data_args.trainsplit_dir_name, "annotation_windows_metadata.json"), 'r') as f:
            self.annotations = json.load(f)
        for idx, sample_annotation in enumerate(self.base.annotations):
            lab = int(sample_annotation['action_id'])
            self.label_to_indices.setdefault(lab, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        kp1, flow1, lab1 = self.base[index]
        lab1_int = int(lab1.item())

        # obtain another sample
        if random.random() < 0.5:
            idx2 = random.choice(self.label_to_indices[lab1_int])
            y = 0.0
        else:
            neg_label = random.choice([l for l in self.labels if l != lab1_int])
            idx2 = random.choice(self.label_to_indices[neg_label])
            y = 1.0

        kp2, flow2, lab2 = self.base[idx2]
        
        y = torch.tensor(y, dtype=torch.float32)
        
        return (kp1, kp2), (flow1, flow2), (lab1, lab2), y