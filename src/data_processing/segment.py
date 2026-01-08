import argparse
import cv2
import os
import pandas as pd
import re
from tqdm import tqdm
from ..utils.logging_utils import setup_logger
from ..utils.cli_args import DataArguments

def time_str_to_seconds(tstr):
    parts = list(map(int, tstr.split(':')))
    if len(parts) == 2:
        # MM:SS
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # H:MM:SS
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {tstr}")

def main(args):
    
    # set logging
    logger = setup_logger(file_path=__file__)
    
    # detect the download dir
    split_dir = os.path.join(args.data_dir, args.seg_dir_name)
    if not os.path.exists(split_dir):
        args.operation = "reload"
        os.makedirs(split_dir, exist_ok=True)
    
    # read the original video
    original_videos_dir = os.path.join(args.data_dir, args.raw_dir_name)
    
    # video name format: {video_id}_{Action Name}_{This Action id}
    start_num = 0
    if args.operation == "extend":
        start_num = max([int(d.split('_')[0]) for d in os.listdir(split_dir)])
    
    # read the metadata
    local_df = pd.read_csv(os.path.join(args.data_dir, args.metadata_name))
    end_num = local_df.shape[0]
    
    # start split
    pattern = r'\("([^"]+)",\s*([\d:]+-[\d:]+)\)'
    for index, row in tqdm(local_df.iloc[start_num:end_num].iterrows(), total=end_num - start_num):
        try:
            actions = re.findall(pattern, row["Actions"])
            
            # read original video
            video_path = os.path.join(original_videos_dir, f'{index:04d}.mp4')
            if not os.path.exists(video_path):
                continue
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # set this count for split name
            temp_count = dict(zip(args.actions, [0 for _ in range(len(args.actions))]))
            for action_name, time_ in actions:
                
                # obtain the start, end timestamp
                start_time_str, end_time_str = time_.split('-')
                start_sec, end_sec = time_str_to_seconds(start_time_str), time_str_to_seconds(end_time_str)
                start_frame = int(start_sec * fps)
                end_frame = int((end_sec + 1) * fps) # +1 to include the last frame
                
                # set the start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # write video clips
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = os.path.join(split_dir, f"{index:04d}_{action_name}_{str(temp_count[action_name])}.mp4") # {video_id}_{Action Name}_{This Action id}
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                temp_count[action_name] += 1
            cap.release()
        except Exception as e:
            print(f"Error processing video in id {index:04d}: {e}")
            logger.error(f"Error processing video in id {index:04d}: {e}")
    

if __name__ == "__main__":
    args = DataArguments()
    main(args)



