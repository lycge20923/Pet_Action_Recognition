'''
This would provide end to end prediction
Input: a video
Output: actions 
'''
import argparse
import cv2
from dataclasses import asdict
import logging
import numpy as np
import time
from datetime import datetime
import os
import torch
import torch.nn.functional as F
import json
import yaml

from src.utils.logging_utils import setup_logger
from src.utils.cli_args import DataArguments, ModelArguments, OutputArguments, PoseEstimationArguments, OpticalFlowArguments, TrainingArguments
from src.utils.common import get_video_info, load_from_wandb
from src.data_processing.stabilization import video_stabilization
from src.data_processing.feature_extraction import normalize_keypoints, crop_and_save_video
from src.models.model import ActionRecognitionModel

from features.pose_estimation import PoseEstimationModel
from features.optical_flow import OpticalFlowModel

def _parse_args():
    parser = argparse.ArgumentParser(description='Prediction for Action Recognition')
    parser.add_argument('--input_path', required=True, help="The video path")
    parser.add_argument('--checkpoint_dir', required=True, help="The directory storing the pretrained weight")
    parser.add_argument('--checkpoint_name', default="best.pth", help="The .pth name")
    return parser.parse_args()

def main():
    
    # --- set logger ---
    logger = setup_logger(file_path=__file__,level=logging.INFO)
    
    # --- get args ---
    args = _parse_args()
    save_adjusted_args_name = TrainingArguments().save_adjusted_args_name
    adjusted_params_path = os.path.join(args.checkpoint_dir, save_adjusted_args_name)
    if os.path.exists(adjusted_params_path):
        with open(adjusted_params_path, 'r') as f:
            adjusted_params = yaml.safe_load(f)
    
        data_params = DataArguments()
        model_params = load_from_wandb(ModelArguments, adjusted_params)
        output_params = OutputArguments()
    else:
        logger.warning("There is no 'args_adjusted.yaml' in the checkpoint dircetory.") 
        logger.warning("You have to adjust predict.py to manually pass in the correct parameters.")
    
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    
    # --- set output folder ---
    timestampe = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_params.output_dir, f"predict_{timestampe}")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- input file name ---
    file_name = os.path.basename(args.input_path)
    base, extension = file_name.split('.')
    
    # --- calculate time and frames----
    video_info = get_video_info(args.input_path)
    fps, frame_width, frame_height, frame_count = \
        video_info["fps"], video_info["frame_width"], video_info["frame_height"], video_info["frame_count"]
    logger.info(f"The input video:{args.input_path}. \n * FPS: {fps} \n * Frame width: {frame_width} \n * Frame height: {frame_height} \n * Frame count: {frame_count} \n Duration: {frame_count/fps:.2f}(s)")
    
    # --- video stabilization(depend on the config) ---
    if not data_params.skip_stabilization:
        start_time= time.time()
        stabilized_result = video_stabilization(args.input_path, data_params.stabilized_crop_percentage)
        logger.info(f"Time for stabilizing video: {time.time() - start_time} seconds") 
        
        # write stabilized videos 
        stabilized_np = stabilized_result["Stabilized"]
        output_stabilized_path = os.path.join(output_dir, f"{base}_stabilized.{extension}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_stabilized_path, fourcc, fps, (frame_width, frame_height))
        for stabilized_img in stabilized_np:
            writer.write(stabilized_img)
        writer.release()
        args.input_path = output_stabilized_path
        
    # --- feature extraction ---
    # initialize model
    pe_model = PoseEstimationModel(**asdict(PoseEstimationArguments()))
    output_plot_path = os.path.join(output_dir, f"{base}_plot.{extension}")
    
    # extract keypoints
    start_time= time.time()
    result_pe = pe_model.predict(input_path=args.input_path, plot_path=output_plot_path, plot_threshold=data_params.plot_pe_threshold)
    logger.info(f"Time for extracting pose estimation: {time.time() - start_time} seconds") 
    
    keypoints_all_frames, bboxes_all_frames = result_pe["keypoints"], result_pe["bboxes"]
    bbox_pe_annotation, bbox_annotation = [], []
    for keypoints_, bboxes_ in zip(keypoints_all_frames, bboxes_all_frames):
        if isinstance(keypoints_, dict):
            keypoints = list(keypoints_.items())  # [(id, array), ...]
        else:
            keypoints = list(enumerate(keypoints_))  # [(idx, array), ...]
        
        # ambiguous if detecting more than one animal 
        if len(keypoints) != 1:
            keypoints = []
            bboxes = []
            
        else:
            keypoints = [[float(ele[1]), float(ele[0]), float(ele[2])] for ele in keypoints[0][1]]
            bboxes = bboxes_.tolist()[0]
        bbox_annotation.append(bboxes)
        normalized_keypoints = normalize_keypoints(keypoints, bboxes, num_nodes=data_params.num_nodes)
        bbox_pe_annotation.append(normalized_keypoints)
    keypoints_arr = np.array(bbox_pe_annotation)
    np.save(os.path.join(output_dir, f"{base}_keypoints.npy"), keypoints_arr)
    
    # extract optical flow
    of_params = OpticalFlowArguments()
    optical_model = OpticalFlowModel(**asdict(of_params))
    output_crop_path = os.path.join(output_dir, f"{base}_crop.{extension}")
    crop_and_save_video(
        input_path=args.input_path,
        bbox_annotation=bbox_annotation,
        output_path=output_crop_path,
        resize_scale=of_params.input_model_size,
        do_crop=data_params.do_crop
    )
    start_time= time.time()
    result_of = optical_model.predict(input_path=output_crop_path)
    logger.info(f"Time for extracting optical flow: {time.time() - start_time} seconds") 
    optical_flow_all_frames = result_of["optical_flows"]
    flows = np.stack(optical_flow_all_frames, axis=0)
    flows_arr = flows.astype(np.float32)
    np.save(os.path.join(output_dir, f"{base}_optical_flows.npy"), flows_arr)
    
    # --- sampling --- 
    samples = []
    sample_dir = os.path.join(output_dir, "samples_npz")
    os.makedirs(sample_dir, exist_ok=True)
    window_size = data_params.window_size
    n_windows = keypoints_arr.shape[0] // window_size 
    for index, w in enumerate(range(n_windows)):
        idxs = np.linspace(w * window_size, (w + 1) * window_size - 1, data_params.num_samples, dtype=int)
        kp_s = keypoints_arr[idxs].astype(np.float32)
        of_s = flows_arr[idxs].astype(np.float32)
        fn = os.path.join(sample_dir, f"{index:04d}.npz")
        np.savez_compressed(fn, keypoints=kp_s, optical_flows=of_s)
        samples.append(fn)
    
    # ---prediction ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coord_path = os.path.join(model_params.pretrained_weight_dir, model_params.stgcn_coords_file_name)
    coords = np.load(coord_path)
    model = ActionRecognitionModel(model_params, data_params, coords).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint {checkpoint_path}")
    window_results = []
    actions = data_params.actions
    with torch.no_grad():
        for fn in samples:
            data = np.load(fn)
            kps = torch.from_numpy(data["keypoints"]).permute(2,0,1).unsqueeze(0).to(device)  # (1,3,T,V)
            fl  = torch.from_numpy(data["optical_flows"]).squeeze(axis=1).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1,2,T,H,W)
            
            feat, logits = model(kps, fl)
            prob = F.softmax(logits, dim=1)[0]
            pred = int(prob.argmax().cpu())
            window_results.append(pred)
            logger.info(f"{os.path.basename(fn)} → class {pred} (p={prob[pred]:.3f})")
    result = {
        "video": base,
        "window_predictions": [actions[i] for i in window_results]
    }
    out_fp = os.path.join(output_dir, f"{base}_predict.json")
    with open(out_fp, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-window predictions to {out_fp}")
    
    
if __name__ == "__main__":
    main()