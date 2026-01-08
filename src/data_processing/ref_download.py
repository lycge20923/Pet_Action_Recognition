import argparse
import os
import shutil
import requests
import zipfile
import glob
from hashlib import md5
import concurrent.futures
import json
import cv2
from natsort import natsorted
import pandas as pd
from tqdm import tqdm
import subprocess

def image2video(data_dir):
    path_to_image = os.path.join(data_dir, "dataset", "image")
    path_to_video = os.path.join(data_dir, "dataset", "video")
    annotation_train = os.path.join(data_dir, "annotation/train.csv")
    annotation_val = os.path.join(data_dir, "annotation/val.csv")
    classes_json = os.path.join(data_dir, "annotation/classes.json")
    visual = False
    fps = 29.97  # 你原先的 fps

    if not os.path.exists(path_to_video):
        os.makedirs(path_to_video)

    with open(classes_json, "r") as file:
        label2number = json.load(file)

    number2label = {value: key for key, value in label2number.items()}

    # 讀標註
    df_train = pd.read_csv(annotation_train, sep=" ")
    df_val = pd.read_csv(annotation_val, sep=" ")
    df = pd.concat([df_train, df_val], axis=0)

    # 建立 main -> [segments] 對應，如 ZG0001 -> [ZG0001.1, ZG0001.2, ...]
    folders = natsorted(os.listdir(path_to_image))
    hierarchy = {}
    for folder in folders:
        main = folder.split(".")[0]
        hierarchy.setdefault(main, []).append(folder)

    # 逐個 main 的每個 segment 產出影片
    for main_name, segments in tqdm(hierarchy.items(), total=len(hierarchy.keys())):
        for segment in natsorted(segments):
            segment_dir = os.path.join(path_to_image, segment)
            image_files = natsorted(os.listdir(segment_dir))
            if len(image_files) == 0:
                continue

            # 從第一張圖自動抓影格尺寸（避免硬編 400x300 尺寸不合）
            first_img = cv2.imread(os.path.join(segment_dir, image_files[0]))
            if first_img is None:
                continue
            h, w = first_img.shape[:2]

            # 輸出檔名：ZG0001_1.mp4（取 segment 的小數點後作為索引）
            try:
                seg_idx = segment.split(".")[1]
            except IndexError:
                seg_idx = "0"
            out_path = os.path.join(path_to_video, f"{main_name}_{seg_idx}.mp4")

            vw = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                fps,
                (w, h)
            )

            # 建 frame_id -> label 映射（這段沿用你的做法）
            mapping = {}
            sub_df = df[df.original_vido_id == segment]  # 注意欄位名用你原本的
            for _, row in sub_df.iterrows():
                mapping[row["frame_id"]] = number2label[row["labels"]]

            # 寫影格
            for j, fname in enumerate(image_files):
                img_path = os.path.join(segment_dir, fname)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                if visual:
                    color = (0, 0, 0)
                    label = mapping.get(j + 1, "")  # 沒標到就給空字串
                    thickness_in = 1
                    size = 0.7
                    label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)
                    copied = image.copy()
                    cv2.rectangle(image, (10, 10), (20 + label_length[0][0], 40), (255, 255, 255), -1)
                    cv2.putText(image, label, (16, 31),
                                cv2.FONT_HERSHEY_SIMPLEX, size, tuple([c - 50 for c in color]),
                                thickness_in, cv2.LINE_AA)
                    image = cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)

                vw.write(image)

            vw.release()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["CBVD", "LoTE", "KABR", "BaboonLand"])
    parser.add_argument("--root_dir", default="data/others/raw")
    return parser.parse_args()

def cbvd(download_path):
    import kagglehub
    src = kagglehub.dataset_download("fandaoerji/cbvd-5cow-behavior-video-dataset")
    if os.path.abspath(src) != os.path.abspath(download_path):
        os.makedirs(download_path, exist_ok=True)
        shutil.move(src, download_path)
    print("Saved to:", download_path)

def lote(download_path):
    
    print("First, manually download to `./Action.zip` in https://drive.google.com/file/d/1jedfvdtfzQ9NHFULkISooAqTvCpTO-0s/view")
    
    TEMP_PATH = "Action.zip"
    
    temp_download_dir = os.path.join(os.path.dirname(download_path), "temp_LoTE")
    with zipfile.ZipFile(TEMP_PATH, "r") as zip_ref:
        zip_ref.extractall(temp_download_dir)
        
    # os.remove(TEMP_PATH)
    print("Saved to temp dir:", temp_download_dir)
    
    action_names = os.listdir(temp_download_dir)
    source_id = 0
    for action_name in action_names:
        action_folder = os.path.join(temp_download_dir, action_name)
        print(f"Processing action folder: {action_name}")
        for fname in sorted(os.listdir(action_folder)):
            src_path = os.path.join(action_folder, fname)
            clip_idx = 0
            new_filename = f"{source_id}_{action_name}_{clip_idx}.mp4"
            dst_path = os.path.join(download_path, new_filename)
            cmd = [
                "ffmpeg",
                "-y",
                "-i", src_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                dst_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"[ERROR] Failed converting {src_path}")
                print(result.stderr.decode("utf-8", errors="ignore"))
            else:
                print(f"[OK] {src_path} -> {dst_path}")
            source_id += 1
    os.rmdir(temp_download_dir)

def kabr(download_path):
    from datasets import load_dataset
    ds = load_dataset("imageomics/KABR")
    ds.save_to_disk(download_path)
    
    base_url = "https://huggingface.co/datasets/imageomics/KABR/resolve/main/KABR"

    """
    To extend the dataset, add additional animals and parts ranges to the list and dictionary below.
    """

    animals = ["giraffes", "zebras_grevys", "zebras_plains"]

    animal_parts_range = {
        "giraffes": ("aa", "ad"),
        "zebras_grevys": ("aa", "am"),
        "zebras_plains": ("aa", "al"),
    }

    dataset_prefix = "dataset/image/"

    # Define the static files that are not dependent on the animals list
    static_files = [
        "README.txt",
        "annotation/classes.json",
        "annotation/distribution.xlsx",
        "annotation/train.csv",
        "annotation/val.csv",
        "configs/I3D.yaml",
        "configs/SLOWFAST.yaml",
        "configs/X3D.yaml",
        "dataset/image2video.py",
        "dataset/image2visual.py",
    ]

    def generate_part_files(animal, start, end):
        start_a, start_b = ord(start[0]), ord(start[1])
        end_a, end_b = ord(end[0]), ord(end[1])
        return [
            f"{dataset_prefix}{animal}_part_{chr(a)}{chr(b)}"
            for a in range(start_a, end_a + 1)
            for b in range(start_b, end_b + 1)
        ]

    # Generate the part files for each animal
    part_files = [
        part
        for animal, (start, end) in animal_parts_range.items()
        for part in generate_part_files(animal, start, end)
    ]

    archive_md5_files = [f"{dataset_prefix}{animal}_md5.txt" for animal in animals]

    files = static_files + archive_md5_files + part_files

    def progress_bar(iteration, total, message, bar_length=50):
        progress = (iteration / total)
        bar = '=' * int(round(progress * bar_length) - 1)
        spaces = ' ' * (bar_length - len(bar))
        message = f'{message:<100}'
        print(f'[{bar + spaces}] {int(progress * 100)}% {message}', end='\r', flush=True)

        if iteration == total:
            print()

    # Directory to save files
    save_dir = os.path.join(download_path, "KABR_files")

    # Loop through each relative file path

    print(f"Downloading the Kenyan Animal Behavior Recognition (KABR) dataset ...")

    total = len(files)
    for i, file_path in enumerate(files):
        # Construct the full URL
        save_path = os.path.join(save_dir, file_path)
        
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping download.")
            continue 
        
        full_url = f"{base_url}/{file_path}"
        
        # Create the necessary directories based on the file path
        os.makedirs(os.path.join(save_dir, os.path.dirname(file_path)), exist_ok=True)
        
        # Download the file and save it with the preserved file path
        response = requests.get(full_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        progress_bar(i+1, total, f"downloaded: {save_path}")
        
    print("Download of repository contents completed.")

    print(f"Concatenating split files into a full archive for {animals} ...")

    def concatenate_files(animal):
        print(f"Concatenating files for {animal} ...")
        part_files_pattern = f"{save_dir}/dataset/image/{animal}_part_*"
        part_files = sorted(glob.glob(part_files_pattern))
        if part_files:
            with open(f"{save_dir}/dataset/image/{animal}.zip", 'wb') as f_out:
                for f_name in part_files:
                    with open(f_name, 'rb') as f_in:
                        # Read and write in chunks
                        CHUNK_SIZE = 8*1024*1024 # 8MB
                        for chunk in iter(lambda: f_in.read(CHUNK_SIZE), b""):
                            f_out.write(chunk) 
                    # Delete part files as they are concatenated              
                    os.remove(f_name)
            print(f"Archive for {animal} concatenated.")
        else:
            print(f"No part files found for {animal}.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(concatenate_files, animals)

    def compute_md5(file_path):
        hasher = md5()
        with open(file_path, 'rb') as f:
            CHUNK_SIZE = 8*1024*1024 # 8MB
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def verify_and_extract(animal):
        print(f"Confirming data integrity for {animal}.zip ...")
        zip_md5 = compute_md5(f"{save_dir}/dataset/image/{animal}.zip")
        
        with open(f"{save_dir}/dataset/image/{animal}_md5.txt", 'r') as file:
            expected_md5 = file.read().strip().split()[0]
        
        if zip_md5 == expected_md5:
            print(f"MD5 sum for {animal}.zip is correct.")

            print(f"Extracting {animal}.zip ...")
            with zipfile.ZipFile(f"{save_dir}/dataset/image/{animal}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{save_dir}/dataset/image/")
            print(f"{animal}.zip extracted.")
            print(f"Cleaning up for {animal} ...")
            os.remove(f"{save_dir}/dataset/image/{animal}.zip")
            os.remove(f"{save_dir}/dataset/image/{animal}_md5.txt")
        else:
            print(f"MD5 sum for {animal}.zip is incorrect. Expected: {expected_md5}, but got: {zip_md5}.")
            print("There may be data corruption. Please try to download and reconstruct the data again or reach out to the corresponding authors for assistance.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(verify_and_extract, animals)

    print("Download script finished.")
    
    image2video(os.path.join(download_path, "KABR_files"))

def baboonland(download_path):
    print("First, manually download to `./charades.zip` in https://drive.google.com/drive/folders/1nH0SlmERRoYZFhXyCn_D2dz-NPwvYKzq")

    TEMP_PATH = "charades.zip"
    with zipfile.ZipFile(TEMP_PATH, "r") as zip_ref:
        zip_ref.extractall(download_path)
        
    print("Saved to:", download_path)
    
    image2video(os.path.join(download_path, "charades"))

def main():
    args = parser()
    download_path = os.path.join(args.root_dir, args.dataset_name)
    os.makedirs(download_path, exist_ok=True)

    if args.dataset_name == "CBVD":
        cbvd(download_path)
    elif args.dataset_name == "LoTE":
        lote(download_path)
    elif args.dataset_name == "KABR":
        kabr(download_path)
    elif args.dataset_name == "BaboonLand":
        baboonland(download_path)

if __name__ == "__main__":
    main()