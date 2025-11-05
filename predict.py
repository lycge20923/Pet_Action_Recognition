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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from src.utils.logging_utils import setup_logger
from src.utils.cli_args import DataArguments, ModelArguments, ResultsArguments, PoseEstimationArguments, OpticalFlowArguments, TrainingArguments
from src.utils.common import get_video_info, load_from_wandb
from src.data_processing.stabilization import video_stabilization
from src.data_processing.feature_extraction import normalize_keypoints, crop_and_save_video
from src.models.model import ActionRecognitionModel

from features.pose_estimation import PoseEstimationModel
from features.optical_flow import OpticalFlowModel

def parse_args():
    parser = argparse.ArgumentParser(description='Prediction for Action Recognition')
    parser.add_argument('--input_path', required=True, help="The video path")
    
    parser.add_argument('--joints_stream_checkpoint_dir', default=None, help="Skeleton model(joints) checkpoint dir")
    parser.add_argument("--local_flow_stream_checkpoint_dir", default=None, help="Skeleton model(flow) checkpoint dir")
    parser.add_argument("--diff_stream_checkpoint_dir", default=None, help="Skeleton model(diff) checkpoint dir")
    parser.add_argument('--i3dgcn_stream_checkpoint_dir', default=None, help="I3D_GCN checkpoint dir")
    parser.add_argument('--I3D_checkpoint_dir', default=None, help="I3D model checkpoint_dir")
    
    return parser.parse_args()

def load_weights(dir_name:str):
    train_params = TrainingArguments()
    data_params = DataArguments()
    save_adjusted_args_name = train_params.save_adjusted_args_name
    adjusted_params_path = os.path.join(dir_name, save_adjusted_args_name)
    with open(adjusted_params_path, 'r') as f:
        adjusted_params = yaml.safe_load(f)
    
    model_params = load_from_wandb(ModelArguments, adjusted_params)
    
    checkpoint_names = [f for f in os.listdir(dir_name) if f.endswith(".pth")]
    checkpoint_name = checkpoint_names[0] if len(checkpoint_names) == 1 else "best.pth"
    checkpoint_path = os.path.join(dir_name, checkpoint_name)
    coords = None
    if model_params.gcn_model_name == "stgcn":
        coord_path = os.path.join(model_params.pretrained_weights_root_dir_name, model_params.gcn_weights_dir_name, model_params.stgcn_coords_file_name)
        coords = np.load(coord_path)
    model = ActionRecognitionModel(model_params, data_params, coords=coords).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Loading model from '{dir_name}' weights...")
    logger.info(f"Missing keys: {info.missing_keys}")
    logger.info(f"Unexpected keys:{info.unexpected_keys}")
    
    logger.info(f"Loading whole pretrained weights from {checkpoint_path}")
    return model

def set_models(model_dirs:list):
    models = []
    for model_dir in model_dirs:
        if model_dir is not None:
            model = load_weights(dir_name=model_dir)
            models.append(model)
    return models

def predict():
    args = parse_args()
    
    global device, logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    input_path = args.input_path
    
    # --- set logger ---
    logger = setup_logger(file_path=__file__,level=logging.INFO)
    
    data_params = DataArguments()
    output_params = ResultsArguments()
    
    # --- set output folder ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_params.output_dir, f"predict_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- input file name ---
    file_name = os.path.basename(input_path)
    base, extension = file_name.split('.')
    
    # --- calculate time and frames----
    video_info = get_video_info(input_path)
    fps, frame_width, frame_height, frame_count = \
        video_info["fps"], video_info["frame_width"], video_info["frame_height"], video_info["frame_count"]
    logger.info(f"The input video:{input_path}. \n * FPS: {fps} \n * Frame width: {frame_width} \n * Frame height: {frame_height} \n * Frame count: {frame_count} \n Duration: {frame_count/fps:.2f}(s)")
    
    # --- video stabilization(depend on the config) ---
    if not data_params.skip_stabilization:
        start_time= time.time()
        stabilized_result = video_stabilization(input_path, data_params.stabilized_crop_percentage)
        logger.info(f"Time for stabilizing video: {time.time() - start_time} seconds") 
        
        # write stabilized videos 
        stabilized_np = stabilized_result["Stabilized"]
        output_stabilized_path = os.path.join(output_dir, f"{base}_stabilized.{extension}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_stabilized_path, fourcc, fps, (frame_width, frame_height))
        for stabilized_img in stabilized_np:
            writer.write(stabilized_img)
        writer.release()
        input_path = output_stabilized_path
        
    # --- feature extraction ---
    # initialize model
    pe_model = PoseEstimationModel(**asdict(PoseEstimationArguments()))
    output_pe_path = os.path.join(output_dir, f"{base}_pose_estimation.{extension}")
    
    # extract keypoints
    start_time= time.time()
    result_pe = pe_model.predict(input_path=input_path, plot_path=output_pe_path, plot_threshold=data_params.plot_pe_threshold)
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
        input_path=input_path,
        bbox_annotation=bbox_annotation,
        output_path=output_crop_path,
        resize_scale=of_params.input_model_size,
        do_crop=data_params.do_crop
    )
    
    output_of_path = os.path.join(output_dir, f"{base}_optical_flow.{extension}")
    start_time= time.time()
    result_of = optical_model.predict(input_path=output_crop_path, output_path=output_of_path)
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
    sample_last_ids = [] # for visualizing
    for index, w in enumerate(range(n_windows)):
        idxs = np.linspace(w * window_size, (w + 1) * window_size - 1, data_params.num_samples, dtype=int)
        sample_last_ids.append(idxs[-1])
        kp_s = keypoints_arr[idxs].astype(np.float32)
        of_s = flows_arr[idxs].astype(np.float32)
        fn = os.path.join(sample_dir, f"{index:04d}.npz")
        np.savez_compressed(fn, keypoints=kp_s, optical_flows=of_s)
        samples.append(fn)
    
    # ---set main models---
    model_dirs = [args.joints_stream_checkpoint_dir, args.local_flow_stream_checkpoint_dir, args.diff_stream_checkpoint_dir, args.I3D_checkpoint_dir, args.i3dgcn_stream_checkpoint_dir]
    models = set_models(model_dirs)
    
    # ---prediction ---
    window_results = []
    all_probs = []
    actions = data_params.actions
    with torch.no_grad():
        for fn in samples:
            data = np.load(fn)
            kps = torch.from_numpy(data["keypoints"]).permute(2,0,1).unsqueeze(0).to(device)  # (1,3,T,V)
            fl  = torch.from_numpy(data["optical_flows"]).squeeze(axis=1).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1,2,T,H,W)
            
            total_logits = 0
            for model in models:
                model.eval()
                _, logits, _ = model(kps, fl, _)
                total_logits += logits
            
            prob = F.softmax(total_logits, dim=1)[0]
            pred = int(prob.argmax().cpu())
            window_results.append(pred)
            all_probs.append(prob.detach().cpu().numpy())
            logger.info(f"{os.path.basename(fn)} → class {pred} (p={prob[pred]:.3f})")

    # visualize prediction
    predictions = [actions[i] for i in window_results]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(output_dir, f"{base}_pred_vis.mp4"), fourcc, fps, (frame_width, frame_height))
    sample_id = 0
    
    # try to detect gt
    try:
        gt_label = file_name.split("_")[1]
    except:
        gt_label = None
        print("It is unclear about the ground-truth, so you have to check by yourself")
    
    interval = data_params.window_size // data_params.num_samples
    for idx_, (frame, keypoints_, bbox_) in enumerate(zip(result_pe["pred_frames"], keypoints_all_frames, bbox_annotation)):
        if isinstance(keypoints_, dict):
            keypoints_ = list(keypoints_.items())  # [(id, array), ...]
        else:
            keypoints_ = list(enumerate(keypoints_))  # [(idx, array), ...]
        
        pred_label = predictions[sample_id]
        
        if gt_label is not None:
            color = (0, 255, 0) if gt_label == pred_label else (0, 0, 255)
        else:
            color = (255, 0, 0)
        
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(pred_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=2)
        x_pos = frame_width - text_w - 10
        y_pos = 20 + text_h
        try:
            cv2.putText(
                frame,
                pred_label,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        except:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            cv2.putText(
                frame,
                pred_label,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        if len(keypoints_) == 1:
            keypoints = keypoints_[0][1]
            keypoints = [[round(ele[1]), round(ele[0])] for ele in keypoints]
        
        if bbox_ != []:
            x_min, y_min, x_max, y_max = bbox_
            x_min = max(1, min(x_min, frame_width - 1))
            x_max = max(1, min(x_max, frame_width - 1))
            y_min = max(1, min(y_min, frame_height - 1))
            y_max = max(1, min(y_max, frame_height - 1))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        writer.write(frame)
        if idx_ == sample_last_ids[sample_id]:
            sample_id += 1
            if sample_id >= len(sample_last_ids):
                break
    writer.release()
                        
    
    result = {
        "video": base,
        "window_predictions": predictions
    }
    print(f"The prediction is: {', '.join(predictions)}")
    out_fp = os.path.join(output_dir, f"{base}_predict.json")
    with open(out_fp, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-window predictions to {out_fp}")
    
    # plot 
    # --- plot (依 y 值是否低於 0.5 決定顏色) ---
    probs_arr = np.stack(all_probs, axis=0)           # [n_windows, n_classes]
    preds = np.array(window_results, dtype=int)       # [n_windows]
    actions = data_params.actions
    n, _ = probs_arr.shape

    gt_idx = None
    if gt_label is not None:
        gt_idx = actions.index(gt_label)

    win = data_params.window_size
    xs_center = (np.arange(n) * win + win / 2.0) / fps

    if gt_idx is not None:
        y = probs_arr[:, gt_idx]
        y_label = f"P(GT={actions[gt_idx]})"
    else:
        y = probs_arr.max(axis=1)
        y_label = "Top-1 probability"

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{base} - {y_label} over time")

    threshold = 0.5
    for i in range(n - 1):
        x1, x2 = xs_center[i], xs_center[i + 1]
        y1, y2 = y[i], y[i + 1]

        # 是否跨越閾值 (0.5)
        if (y1 - threshold) * (y2 - threshold) < 0:
            t = (threshold - y1) / (y2 - y1)
            xc = x1 + t * (x2 - x1)
            yc = threshold
            color1 = 'green' if y1 >= threshold else 'red'
            ax.plot([x1, xc], [y1, yc], color=color1, linewidth=3, solid_capstyle='round')
            color2 = 'green' if y2 >= threshold else 'red'
            ax.plot([xc, x2], [yc, y2], color=color2, linewidth=3, solid_capstyle='round')
        else:
            color = 'green' if (y1 >= threshold and y2 >= threshold) else 'red'
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, solid_capstyle='round')

    # # 畫中心點
    # ax.scatter(xs_center, y, s=30, c=['green' if v >= threshold else 'red' for v in y], zorder=3)

    ax.set_xlim(xs_center[0], xs_center[-1])
    plt.tight_layout()
    curve_path = os.path.join(output_dir, f"{base}_gt_confidence_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved GT confidence curve to {curve_path}")
    
    
if __name__ == "__main__":
    predict()