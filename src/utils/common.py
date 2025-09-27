import cv2
import dataclasses
import numpy as np
import random
from .cli_args import DataArguments, ModelArguments
import os

import torch
from torch.utils.data._utils.collate import default_collate

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

def saving_self_training_best_gcn_weights_path(data_params:DataArguments, model_params:ModelArguments):
    path = os.path.join(model_params.pretrained_weights_root_dir_name, 
                                    model_params.self_training_weights_dir_name,
                                    str(data_params.fold_num),
                                    model_params.gcn_weights_dir_name, 
                                    model_params.gcn_weights_file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


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
    is_siamese = (len(first_item) == 4 and 
                  isinstance(first_item[0], tuple) and len(first_item[0]) == 2 and # (kp1,kp2)
                  isinstance(first_item[1], tuple) and len(first_item[1]) == 2 and # (flow1,flow2)
                  isinstance(first_item[2], tuple) and len(first_item[2]) == 2)   # (lab1,lab2)
    # SiameseKpOfDataset
    if is_siamese:
        kp1_list = [item[0][0] for item in batch] # might have None
        kp2_list = [item[0][1] for item in batch]
        flow1_list = [item[1][0] for item in batch] # might have None
        flow2_list = [item[1][1] for item in batch]
        lab1_list = [item[2][0] for item in batch]
        lab2_list = [item[2][1] for item in batch]
        y_list = [item[3] for item in batch]

        collated_lab1 = default_collate(lab1_list)
        collated_lab2 = default_collate(lab2_list)
        collated_y = default_collate(y_list)

        # special case for optical flow
        collated_kp1 = default_collate(kp1_list) if (kp1_list and kp1_list[0] is not None) else None
        collated_kp2 = default_collate(kp2_list) if (kp2_list and kp2_list[0] is not None) else None
        collated_flow1 = default_collate(flow1_list) if (flow1_list and flow1_list[0] is not None) else None
        collated_flow2 = default_collate(flow2_list) if (flow2_list and flow2_list[0] is not None) else None
        
        return (collated_kp1, collated_kp2), \
               (collated_flow1, collated_flow2), \
               (collated_lab1, collated_lab2), \
               collated_y
    # KpOfDataset
    else: 
        kp_list = [item[0] for item in batch]
        flow_list = [item[1] for item in batch] 
        lab_list = [item[2] for item in batch]
        name_list = [item[3] for item in batch]
        
        collated_lab = default_collate(lab_list)
        name_lab = default_collate(name_list)
        
        # special case for optical flow
        collated_kp = default_collate(kp_list) if (kp_list and kp_list[0] is not None) else None
        collated_flow = default_collate(flow_list) if (flow_list and flow_list[0] is not None) else None
        return collated_kp, collated_flow, collated_lab, name_lab

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False