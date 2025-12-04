import argparse
import cv2
import dataclasses
import numpy as np
import random
from .cli_args import DataArguments, ModelArguments
import os

import torch
from torch.utils.data._utils.collate import default_collate

from .cli_args import DEFAULT_SKELETON_LIST

def get_video_info(video_path:str) -> cv2.VideoCapture:
    '''
    Purpose: Read a video file and return its properties.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        dict: A dictionary containing the video's properties such as fps, width, and height.
    '''
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {"fps": fps, "frame_width": frame_width, "frame_height":frame_height, "frame_count": frame_count}

def load_from_wandb(cls, config:dict):
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config.items() if k in field_names}
    return cls(**filtered)

def make_tree_from_skeleton_list(skeleton_list = DEFAULT_SKELETON_LIST, num_joints: int | None = None) -> torch.Tensor:
    """
    只回傳 parents (V,), root=-1
      - 每個 child 僅保留第一個出現的 parent
      - 以 Union-Find 避免形成 cycle
    """
    if num_joints is None:
        num_joints = max(max(p, c) for p, c in skeleton_list) + 1
    V = num_joints

    parents = torch.full((V,), -1, dtype=torch.long)

    # Union-Find
    uf = list(range(V))
    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra
            return True
        return False

    for p, c in skeleton_list:
        if parents[c] != -1:
            continue          # 保留第一個 parent
        if find(p) == find(c):
            continue          # 會造成環 -> 跳過
        parents[c] = p
        union(p, c)

    return parents  # (V,)

def compute_bones(
    keypoints: torch.Tensor,   # (B, 3 or 2, T, V): (x, y, [conf])
) -> torch.Tensor:
    """
    Returns:
      bones: (B, 2, T, V)
    """
    B, C, T, V = keypoints.shape
    parents = make_tree_from_skeleton_list(num_joints=V)
    
    xy   = keypoints[:, 0:2]                       # (B,2,T,V)
    conf = keypoints[:, 2:3] if C >= 3 else None   # (B,1,T,V) or None

    device = keypoints.device
    parents = parents.to(device)
    valid = (parents >= 0)                         # (V,)
    parents_idx = torch.where(valid, parents, torch.zeros_like(parents))

    xy_par = xy.index_select(dim=3, index=parents_idx)    # (B,2,T,V)
    bones  = xy - xy_par                                   # (B,2,T,V)
    bones  = torch.where((~valid).view(1,1,1,V), torch.zeros_like(bones), bones)

    conf_par = conf.index_select(dim=3, index=parents_idx)  # (B,1,T,V)
    valid_conf = (conf > 0) & (conf_par > 0) & valid.view(1,1,1,V)
    bones = bones * valid_conf.to(bones.dtype)

    return bones

def kp_diff_stats(keypoints: torch.Tensor):
    """
    Args:
        keypoints: (B, 3, T, J) – normalized (x, y, conf)
    Returns:
        dxy: (B, 2, T, J)
    """
    # 只取 x,y；形狀 (B, 2, T, J)
    xy = keypoints[:, :2, ...]
    # 逐幀差分：Δx = x_t - x_{t-1}, Δy 同理；在 t=0 補 0
    dxy = xy[:, :, 1:, :] - xy[:, :, :-1, :]                     # (B, 2, T-1, J)
    zero = torch.zeros_like(dxy[:, :, :1, :])                    # (B, 2, 1,   J)
    dxy = torch.cat([zero, dxy], dim=2)                          # (B, 2, T,   J)
    return dxy

def custom_collate(batch):
    first_item = batch[0]
    is_siamese = (len(first_item) == 5 and 
                  isinstance(first_item[0], tuple) and len(first_item[0]) == 2 and # (kp1,kp2)
                  isinstance(first_item[1], tuple) and len(first_item[1]) == 2 and # (flow1,flow2)
                  isinstance(first_item[2], tuple) and len(first_item[2]) == 2 and # (rgb1, rgb2)
                  isinstance(first_item[3], tuple) and len(first_item[3]) == 2)   # (lab1,lab2)
    # SiameseKpOfDataset
    if is_siamese:
        kp1_list = [item[0][0] for item in batch] # might have None
        kp2_list = [item[0][1] for item in batch]
        flow1_list = [item[1][0] for item in batch] # might have None
        flow2_list = [item[1][1] for item in batch]
        rgb1_list = [item[2][0] for item in batch] # might have None
        rgb2_list = [item[2][1] for item in batch]
        lab1_list = [item[3][0] for item in batch]
        lab2_list = [item[3][1] for item in batch]
        y_list = [item[4] for item in batch]

        collated_lab1 = default_collate(lab1_list)
        collated_lab2 = default_collate(lab2_list)
        collated_y = default_collate(y_list)

        # special case for optical flow
        collated_kp1 = default_collate(kp1_list) if (kp1_list and kp1_list[0] is not None) else None
        collated_kp2 = default_collate(kp2_list) if (kp2_list and kp2_list[0] is not None) else None
        collated_flow1 = default_collate(flow1_list) if (flow1_list and flow1_list[0] is not None) else None
        collated_flow2 = default_collate(flow2_list) if (flow2_list and flow2_list[0] is not None) else None
        collated_rgb1 = default_collate(rgb1_list) if (rgb1_list and rgb1_list[0] is not None) else None
        collated_rgb2 = default_collate(rgb2_list) if (rgb2_list and rgb2_list[0] is not None) else None
        
        return (collated_kp1, collated_kp2), \
               (collated_flow1, collated_flow2), \
               (collated_rgb1, collated_rgb2), \
               (collated_lab1, collated_lab2), \
               collated_y
    # KpOfDataset
    else: 
        kp_list = [item[0] for item in batch]
        flow_list = [item[1] for item in batch]
        rgb_list = [item[2] for item in batch] 
        lab_list = [item[3] for item in batch]
        name_list = [item[4] for item in batch]
        
        collated_lab = default_collate(lab_list)
        name_lab = default_collate(name_list)
        
        # special case for optical flow
        collated_kp = default_collate(kp_list) if (kp_list and kp_list[0] is not None) else None
        collated_flow = default_collate(flow_list) if (flow_list and flow_list[0] is not None) else None
        collated_rgb = default_collate(rgb_list) if (rgb_list and rgb_list[0] is not None) else None
        return collated_kp, collated_flow, collated_rgb, collated_lab, name_lab

def set_comparison_config(data_args:DataArguments, dataset_name:str):
    COMPARISON_DATASETS = {"BaboonLand":"BaboonLand/charades", "KABR":"KABR/KABR_files", "LoTE":"LoTE"}
    if dataset_name not in COMPARISON_DATASETS.keys():
        raise ValueError("You have to send the correct dataset name, or o.w. you have to set the dataset.")
    sub_path = COMPARISON_DATASETS[dataset_name]
    data_args.data_dir = "data/others"
    if dataset_name != "LoTE":
        data_args.seg_dir_name = os.path.join("raw", sub_path, "dataset/video")
        data_args.window_size = 65
        data_args.num_samples = 16
        data_args.min_kp_rate = 0
    else:
        data_args.seg_dir_name = os.path.join("raw", sub_path)
        data_args.actions = ["Aggregation", "CircumanalGlandSigning", "Defecating", 
                             "Exploratory", "Foraging", "Jumping", "Mounting", "Playing", 
                             "Smelling", "Urinating", "Walking", "Amusing", "Climbing", 
                             "DrinkWater", "Feeding", "Grooming", "Miscellaneous", 
                             "Parental", "Resting", "Trotting", "UrineSigning"]
    data_args.stabilized_dir_name = os.path.join(dataset_name, "stabilized")
    data_args.feature_extract_dir_name = os.path.join(dataset_name, "feature_extracted")
    data_args.trainsplit_dir_name = os.path.join(dataset_name, "train_split")
    
    return data_args

def set_comparison_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--for_comparison", action="store_true", help="Whether compared to other dataset")
    parser.add_argument("--dataset_name", choices=["BaboonLand", "KABR", "LoTE"])
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False