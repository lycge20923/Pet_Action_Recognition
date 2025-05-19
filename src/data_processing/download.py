import argparse
import os
import pandas as pd
from pytubefix import YouTube 
from pytubefix.cli import on_progress
from tqdm import tqdm
import time, random
import shutil
from ..utils.logging_utils import setup_logger
from ..utils.cli_args import DataArguments

def download_mp4(url:str, output_path:str) -> None:
    temp_download_path = os.path.join(os.getcwd(), "temp")
    yt = YouTube(url, on_progress_callback = on_progress, use_oauth=True)
    ys = yt.streams.get_highest_resolution()
    download_path = ys.download(temp_download_path)
    shutil.move(download_path, output_path)
    os.rmdir(temp_download_path)

if __name__ =='__main__':
    args = DataArguments()
    
    # make the directory for original video
    dataset_dir = os.path.join(args.data_dir, args.raw_dir_name) 
    os.makedirs(dataset_dir, exist_ok=True)

    # set logging
    logger = setup_logger(file_path=__file__)
    
    # read excel from drive
    drive_df = pd.read_csv(args.drive_url)
    num_old_rows, num_new_rows = 0, drive_df.shape[0]

    # check and set operation 
    metadata_path = os.path.join(args.data_dir, args.metadata_name)
    if args.operation == "extend":
        
        # read the local metadata file
        if not os.path.exists(metadata_path):
            logger.warning("```metadata.csv``` was not found in the destination directory. We would then use reload operation to re-download the dataset.")
            num_old_rows = 0 
        else:
            local_df = pd.read_csv(metadata_path)
            num_old_rows = local_df.shape[0]
            if num_old_rows > num_new_rows:
                logger.warning("Some videos are deleted, We would use reload operation to re-download the dataset.")
                num_old_rows = 0

    with open(metadata_path, 'w') as f:
        drive_df.to_csv(f, index=False)
        print(f"metadata.csv file is saved in {metadata_path}")

    for index, row in tqdm(drive_df.iloc[num_old_rows:num_new_rows].iterrows(),  total=num_new_rows - num_old_rows):
        
        # mimic human activity to avoild being blocked
        delay = random.uniform(5, 25) # delay
        time.sleep(delay)
        
        species, web_url, actions, validation, fold = row["Animal"], row["Website"], row["Actions"], row["Validation"], row["Fold"]
        
        # read video
        try:
            output_path = os.path.join(dataset_dir, f'{index:04d}.mp4')
            download_mp4(url=web_url, output_path=output_path)
            
        except Exception as e:
            print(f"Error processing {species} video: {e}")
            logger.error(f"Error processing {species} video: {e}")
            continue