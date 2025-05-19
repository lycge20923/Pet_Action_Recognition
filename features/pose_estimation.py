import argparse
import os
import cv2
from huggingface_hub import hf_hub_download
import shutil
import numpy as np 
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import time

from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.visualization import joints_dict

class PoseEstimationModel():
    def __init__(self, model_dir:str="models/pe", model_type:str="h", yolo_type:str="s", yolo_size:int=320, pretrained_dataset:str="ap10k", **kwargs):
        self.model_dir = model_dir
        self.model_type = model_type
        self.yolo_type = yolo_type
        self.yolo_size = yolo_size
        self.pretrained_dataset = pretrained_dataset
        self._check_and_load_model()
    
    def _check_and_load_model(self, ext:str=".pth", ext_yolo:str=".pt"):
        model_path = os.path.join(self.model_dir, f'vitpose-{self.model_type}-{self.pretrained_dataset}') + ext
        yolo_path = os.path.join(self.model_dir, f'yolov8{self.yolo_type}') + ext_yolo
        
        # check
        if (not os.path.exists(model_path)) or (not os.path.exists(yolo_path)):
            os.makedirs(self.model_dir, exist_ok=True)
            # load 
            import shutil
            SOURCE, SOURCE_REPO = "torch", 'JunkyByte/easy_ViTPose'
            FILENAME = os.path.join(SOURCE, f'{self.pretrained_dataset}/vitpose-' + self.model_type + f'-{self.pretrained_dataset}') + ext
            FILENAME_YOLO = 'yolov8/yolov8' + self.yolo_type + ext_yolo
            _model_path = hf_hub_download(repo_id=SOURCE_REPO, filename=FILENAME)
            _yolo_path = hf_hub_download(repo_id=SOURCE_REPO, filename=FILENAME_YOLO)
            shutil.copy2(_model_path, model_path)
            shutil.copy2(_yolo_path, yolo_path)
        
        # declare model
        self.model = VitInference(model_path, yolo_path, self.model_type,
                            None, self.pretrained_dataset,
                            self.yolo_size, is_video=True,
                            single_pose=None,
                            yolo_step=1)
    
    def predict(self, input_path:str, plot_path:str=None, plot_threshold:float=0.5):
        
        # prepare for the data
        is_video = False if input_path.endswith('.jpg') or input_path.endswith('.png') else True
        if is_video:
            reader = VideoReader(input_path, 0)
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            output_size = frame.shape[:2][::-1]
            cap.release()
            
            # set if the plot_path is not None
            if plot_path is not None:
                out_writer = cv2.VideoWriter(plot_path, cv2.VideoWriter_fourcc(*'MJPG'),  # More efficient codec
                                            fps, output_size)  
        else:
            total_frames = 1
            reader = [np.array(Image.open(input_path).convert('RGB').rotate(0))]
        
        
        # start inference
        total_time = 0
        keypoints, bboxes_list = [], []
        predict_count = 0
        for (ith, img) in tqdm(enumerate(reader), total=total_frames):
            t0 = time.time()
            
            frame_keypoints, bboxes = self.model.inference(img)
            if len(frame_keypoints) > 0:
                predict_count += 1
            keypoints.append(frame_keypoints)
            bboxes_list.append(bboxes)
            
            delta_time = time.time() - t0
            total_time += delta_time
            
            if plot_path is not None:
                img = self.model.draw(False, False, plot_threshold)[..., ::-1]
                if is_video:
                    out_writer.write(img)
                else:
                    cv2.imwrite(plot_path, img)
        if plot_path is not None:
            if is_video:
                out_writer.release()
        exec_fps = total_frames / total_time
        avg_predict_success = predict_count / (ith + 1)
        
        # final
        return {"keypoints": keypoints, "bboxes": bboxes_list, "stat":{"exec_fps": exec_fps, "keypoint_detection_rate": avg_predict_success}}


import cv2
import numpy as np
from typing import Union, List, Dict

def draw_keypoints_and_bboxes(
    image: np.ndarray,
    keypoints: Union[List[np.ndarray], Dict[any, np.ndarray]],
    bboxes: np.ndarray,
    kp_thresh: float = 0.3
) -> np.ndarray:
    
    vis = image.copy()
    num = len(bboxes)

    # generate colors
    colors = []
    for i in range(num):
        hue = int(i * 180 / max(num,1))
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

    if isinstance(keypoints, dict):
        items = list(keypoints.items())  # [(id, array), ...]
    else:
        items = list(enumerate(keypoints))  # [(idx, array), ...]

    # draw by pairs 
    for idx, item in enumerate(items):
        if idx >= num:
            break  # 多出的 keypoints 沒對應 bbox 就跳過
        # 取得同組的 keypoint array
        kpts_array = item[1]
        box = bboxes[idx]
        color = colors[idx]

        # draw bboxes
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=color, thickness=2)

        # draw points
        for x, y, conf in kpts_array:
            if conf >= kp_thresh:
                cv2.circle(vis, (int(y), int(x)), radius=4, color=color, thickness=-1)

    return vis

if __name__ == "__main__":
    
    # limitation: only for image 
    import sys
    input_path = sys.argv[1]
    output_dir = "output"

    # load the file 
    if os.path.isfile(input_path):
        input_paths = [input_path]
    elif os.path.isdir(input_path):
        input_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) \
            if os.path.isfile(os.path.join(input_path, f)) and not (f.endswith(".txt") or f.endswith(".json"))]
    
    # for output
    time_ = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(output_dir, f"pose_estimation_simple_trial_{os.path.basename(input_path)}_{time_}")
    os.makedirs(output_dir, exist_ok=True)
        
    # record
    with open(os.path.join(output_dir, "record.txt"), 'a+') as f:
        f.write(f"Input Directory: {input_path}" + "\n")
    
    pe_model = PoseEstimationModel()
    
    sum_exec_fps, sum_keypoint_detection_rate =0, 0
    for input_path in input_paths:
        result = pe_model.predict(input_path, plot_path=os.path.join(output_dir, os.path.basename(input_path)))
        keypoints, bboxes = result["keypoints"][0], result["bboxes"][0]
        
        # validation
        orig = cv2.imread(input_path)
        vis = draw_keypoints_and_bboxes(orig, keypoints, bboxes, kp_thresh=0.3)
        vis_name = f"vis_{os.path.basename(input_path)}"
        vis_path = os.path.join(output_dir, vis_name)
        cv2.imwrite(vis_path, vis)
        
        exec_fps = result["stat"]["exec_fps"]
        keypoint_detection_rate = result["stat"]["keypoint_detection_rate"]
        txt = f"Input:{os.path.basename(input_path)}, Execution FPS: {exec_fps}, Keypoints Detection Rate: {keypoint_detection_rate}"
        print(txt)
        # record stat info
        with open(os.path.join(output_dir, "record.txt"), 'a+') as f:
            f.write(txt + "\n")

        # calculate the sum 
        sum_exec_fps += exec_fps
        sum_keypoint_detection_rate += keypoint_detection_rate
        
    # record stat info
    with open(os.path.join(output_dir, "record.txt"), 'a+') as f:
        txt = f"Avg Execution FPS: {sum_exec_fps / len(input_paths):.2f}, Avg Keypoints Detection Rate: {sum_keypoint_detection_rate / len(input_paths):.2f}"
        f.write(txt + "\n")