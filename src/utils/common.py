import cv2
import dataclasses
from .cli_args import DataArguments, ModelArguments
import os

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
