import logging
import numpy as np
import os
import cv2
from datetime import datetime
import time
from tqdm import tqdm
from vidgear.gears import VideoGear
import argparse

from ..utils.logging_utils import setup_logger
from ..utils.cli_args import DataArguments
from ..utils.common import get_video_info

def adjust_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, help="Adjusted data root dir path")
    parser.add_argument("--stabilized_dir_name", default=None, help="Adjusted stabilized data dir path")
    parser.add_argument("--seg_dir_name", default=None, help="Adjusted segemneted dir path")
    return parser.parse_args()
    
def video_stabilization(video_path:str, crop_percentage:float) -> np.ndarray:
    '''
    Purpose: Conduct video stabilization on a given video file.
    Args:
        video_path(str): Video path for stabilization
        crop_percentage(float): Crop the central part to delete the black border
    Returns:
        np.array: The total frames for pose estimation
    '''

    # read the properties of the video
    vid_info = get_video_info(video_path)
    frame_width, frame_height, fps = vid_info["frame_width"], vid_info["frame_height"], vid_info["fps"]
    
    # start inference
    t0 = time.time()
    stream = VideoGear(source=video_path, stabilize=True, framerate=fps).start()
    delta_time = time.time() - t0
    
    
    # calculate cropped boundary
    new_width = int(frame_width * crop_percentage)
    new_height = int(frame_height * crop_percentage)
    x_start = (frame_width - new_width) // 2
    y_start = (frame_height - new_height) // 2
    
    # start video stabilization
    stablized_video = []
    while True:
        frame = stream.read()
        if frame is None:
            break
        
        # crop and resize
        cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
        cropped_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        stablized_video.append(cropped_frame)
    stream.stop()
    exec_fps = len(stablized_video) / delta_time
    
    return {"Stabilized": np.array(stablized_video), "stat":{"exec_fps": exec_fps}, "vid_info":vid_info}

def main():
    
    # get parameters
    data_args = DataArguments()
    
    adjusted_args = adjust_args()
    if adjusted_args.data_dir is not None:
        data_args.data_dir = adjusted_args.data_dir
    if adjusted_args.stabilized_dir_name is not None:
        data_args.stabilized_dir_name = adjusted_args.stabilized_dir_name
    if adjusted_args.seg_dir_name is not None:
        data_args.seg_dir_name = adjusted_args.seg_dir_name
    
    # set logging 
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    
    # detect the segmented dir
    stabilized_dir = os.path.join(data_args.data_dir, data_args.stabilized_dir_name)
    if not os.path.exists(stabilized_dir):
        os.makedirs(stabilized_dir, exist_ok=True)
        
    seg_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    seg_video_paths = [os.path.join(seg_dir, ele) for ele in os.listdir(seg_dir) if ele.endswith('.mp4')]
    
    # record input
    logger.info(f"Input Directory: {seg_dir}")
    
    # start video stabilization
    sum_exec_fps = 0
    for id_, input_path in tqdm(enumerate(seg_video_paths), total=len(seg_video_paths)):
        try:
            stabilized_result = video_stabilization(video_path=input_path, crop_percentage=data_args.stabilized_crop_percentage)
            stabilized_imgs_np = stabilized_result["Stabilized"]
            exec_fps = stabilized_result["stat"]["exec_fps"]
            sum_exec_fps += exec_fps
            
            # output stabilized videos
            vid_info = stabilized_result["vid_info"]
            output_path = os.path.join(stabilized_dir, os.path.basename(input_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, vid_info["fps"], (vid_info["frame_width"], vid_info["frame_height"]))
            for stabilized_img in stabilized_imgs_np:
                writer.write(stabilized_img)
            writer.release()
            logger.info(f"Input:'{os.path.basename(input_path)}', Execution FPS: {exec_fps:.4f}")
            
        except Exception as e:
            logger.error(f"Input:{os.path.basename(input_path)}, Error happens: {e}")
    txt = f"(Pose Estimation)Avg Execution FPS: {sum_exec_fps / len(seg_video_paths):.4f}"
    logger.info(txt)
    
if __name__ == "__main__":
    main()