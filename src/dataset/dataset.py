import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import time
from tqdm import tqdm
import gc

from ..utils.cli_args import DataArguments, AugmentationArguments
from .aug_func import *

class KpOfDataset(Dataset):
    def __init__(self,
                 data_params: DataArguments,
                 aug_params: AugmentationArguments,
                 load_kps: bool = True,
                 load_flows: bool = False,
                 istrain: bool = True,
                 fold_num: int = 0,
                 data_seg_num: int = 6,
                 for_test = False,
                 test_len = 32
                 ):
        super().__init__()
        self.data_params = data_params
        self.aug_params = aug_params
        
        self.load_flows = load_flows
        self.load_kps = load_kps

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
        
        if for_test:
            self.annotations = self.annotations[:test_len]
        
        # trial loading complete at first time
        self.data_seg_num = data_seg_num
        for i in range(data_seg_num):
            setattr(self, f"dataset_{str(i)}", [])
        
        for idx, annotation in tqdm(enumerate(self.annotations), total=len(self.annotations)):
            sample = dict()
                        
            if self.load_flows:
                optical_flows_np = np.load(annotation["of_feature_file"])
                optical_flows_np = optical_flows_np.squeeze(axis=1) # T, 2, H, W
                optical_flows_np = optical_flows_np.transpose(1, 0, 2, 3) # 2, T, H, W
                sample["optical_flows_np"] = optical_flows_np
                
            if self.load_kps:
                keypoints_np = np.load(annotation["kp_feature_file"]).astype(np.float32, copy=False)
                sample["keypoints_np"] = keypoints_np
            sample["video_name"] = annotation["video_name"]
            getattr(self, f"dataset_{str(idx % data_seg_num)}").append(sample)
        if self.load_flows:
            del optical_flows_np
        if self.load_kps:
            del keypoints_np 
        gc.collect()    
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load initial data from annotations
        # And 'action_id' is the label
        # load keypoints from the .npz window file
        
        # trial loading complete at first time
        sample = getattr(self, f"dataset_{str(index % self.data_seg_num)}")[index // self.data_seg_num]
        video_name = sample["video_name"]
        label = torch.tensor(self.annotations[index]["action_id"], dtype=torch.long)
        
        optical_flows_np = sample["optical_flows_np"] if self.load_flows else None
        keypoints_np = sample["keypoints_np"] if self.load_kps else None
        
        # --- transform to tensor --- 
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

        return keypoints_tensor, optical_flows_tensor, label, video_name

class SiameseKpOfDataset(Dataset):
    """
    Wrapper to produce pairs for joint CrossEntropy + Contrastive training.
    Returns:
      (x1, x2), (lab1, lab2), y
      lab1, lab2: labels for CE loss
      y: binary (0: same, 1: different) for ContrastiveLoss
    """
    def __init__(self, base_dataset: KpOfDataset):
        self.base = base_dataset
        # Build mapping label -> indices for sampling
        self.label_to_indices = {}
        for idx, sample_annotation in enumerate(self.base.annotations):
            lab = int(sample_annotation['action_id'])
            self.label_to_indices.setdefault(lab, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        kp1, flow1, lab1, _ = self.base[index]
        lab1_int = int(lab1.item())

        # obtain another sample
        if random.random() < 0.5:
            idx2 = random.choice(self.label_to_indices[lab1_int])
            y = 0.0
        else:
            neg_label = random.choice([l for l in self.labels if l != lab1_int])
            idx2 = random.choice(self.label_to_indices[neg_label])
            y = 1.0

        kp2, flow2, lab2, _ = self.base[idx2]
        y = torch.tensor(y, dtype=torch.float32)
        
        return (kp1, kp2), (flow1, flow2), (lab1, lab2), y