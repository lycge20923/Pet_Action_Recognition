from dataclasses import asdict
import os
import time
from datetime import datetime
import logging
import json
import numpy as np
import cv2
import argparse
import pandas as pd

from ..utils.cli_args import DataArguments, PoseEstimationArguments, OpticalFlowArguments, ResultsArguments
from features.pose_estimation import PoseEstimationModel
from features.optical_flow import OpticalFlowModel
from ..utils.logging_utils import setup_logger
from ..utils.common import get_video_info, set_comparison_config, set_comparison_config_args

def find_video_info(video_df, video_id_str):
    
    result = video_df[video_df["original_vido_id"] == video_id_str]
    if result.empty:
        print(f"找不到 original_vido_id = {video_id_str}")
    return result.iloc[0]

def aggregate_video_level_labels(annotation_dir):
    train_path, val_path = os.path.join(annotation_dir, "train.csv"), os.path.join(annotation_dir, "val.csv")
    
    train_df = pd.read_csv(train_path, sep=None, engine="python")
    train_df["split"] = "train"

    val_df = pd.read_csv(val_path, sep=None, engine="python")
    val_df["split"] = "val"

    # 合併
    df = pd.concat([train_df, val_df], ignore_index=True)

    # 將 original_vido_id 裡的 '.' 改成 '_'
    df["original_vido_id"] = df["original_vido_id"].str.replace(".", "_", regex=False)

    # 取每個影片最常出現的 label
    def most_frequent_label(s):
        counts = s.value_counts()
        maxc = counts.max()
        return sorted(counts[counts == maxc].index)[0]

    # 以 original_vido_id, video_id 分組後取最常見的 label
    video_level = (
        df.groupby(["original_vido_id", "video_id", "split"], as_index=False)
          .agg(labels=("labels", most_frequent_label))
    )

    return video_level

def normalize_keypoints(keypoints, bboxes, num_nodes):
    """
    Normalize each keypoint in a frame relative to its bounding box.
    If either keypoints or bboxes is empty, return a single [0,0,0].
    """
    if not keypoints or not bboxes or len(bboxes) != 4:
        return [[0.0, 0.0, 0.0] for _ in range(num_nodes)]
    x1, y1, x2, y2 = bboxes
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return [[0.0, 0.0, 0.0] for _ in range(num_nodes)]
    normalized = []
    for x, y, c in keypoints:
        nx = (x - x1) / width
        ny = (y - y1) / height
        normalized.append([nx, ny, c])
    return normalized


def crop_and_save_video(input_path: str,
                        bbox_annotation: list,
                        output_path: str,
                        resize_scale: tuple,
                        do_crop:bool=True):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, resize_scale)

    prev_bbox = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # first: get crop info
        # if we conduct cropping(do_crop==True), we need the cropping info. 
        # o.w.(do_crop==False), the whole image would be the bbox 
        if do_crop: 
            bbox = bbox_annotation[frame_idx]
            if bbox is None or len(bbox) == 0:
                bbox = prev_bbox
            else:
                prev_bbox = bbox
        else:
            h, w = frame.shape[:2]
            bbox = [0, 0, w, h]
        
        # second: crop 
        if not bbox:
            crop = np.zeros((resize_scale[1], resize_scale[0], 3), dtype=np.uint8)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
        
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            resized = np.zeros((resize_scale[1], resize_scale[0], 3), dtype=np.uint8)
        else:
            # scaling and cropping
            scale = min(resize_scale[0] / cw, resize_scale[1] / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            resized = cv2.resize(crop, (new_w, new_h))

            pad_w = resize_scale[0] - new_w
            pad_h = resize_scale[1] - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            resized = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

        writer.write(resized)
        frame_idx += 1

    cap.release()
    writer.release()

def process_single_video(input_path, id_, data_args, comparison_args, of_args,
                         pe_model, optical_model, logger,
                         video_df=None, plot_output_dir=None,
                         time_=None, np_save_dir=None):
    """
    將單支影片的整個流程（pose + optical flow + 存檔）包成一個函式。
    若執行失敗，直接 raise exception，外層 try-except 處理 fallback。
    """
    video_name = os.path.basename(input_path)

    # --- annotation 解析 ---
    if comparison_args.for_comparison:
        result_annotation = find_video_info(video_df, video_name.split(".")[0])
        source_video_id, action_id = result_annotation["original_vido_id"], int(result_annotation["labels"])
        annotation = {
            "id": id_,
            "action_id": action_id,
            "video_name": video_name,
            "source_video_id": source_video_id,
            "split": result_annotation["split"]
        }
    else:
        split_ = video_name.split("_")
        source_video_id, action_id = int(split_[0]), data_args.actions.index(split_[1])
        annotation = {
            "id": id_,
            "action_id": action_id,
            "video_name": video_name,
            "source_video_id": source_video_id
        }

    # --- pose estimation ---
    if data_args.plot_pe:
        result_pe = pe_model.predict(
            input_path=input_path,
            plot_path=os.path.join(plot_output_dir, os.path.basename(input_path)),
            plot_threshold=data_args.plot_pe_threshold
        )
    else:
        result_pe = pe_model.predict(input_path=input_path)

    keypoints_all_frames, bboxes_all_frames = result_pe["keypoints"], result_pe["bboxes"]
    bbox_pe_annotation, bbox_annotation = [], []
    detect_status = set()

    for keypoints_, bboxes_ in zip(keypoints_all_frames, bboxes_all_frames):
        if isinstance(keypoints_, dict):
            keypoints = list(keypoints_.items())
        else:
            keypoints = list(enumerate(keypoints_))

        if len(keypoints) == 0 or (len(keypoints) > 1 and not comparison_args.for_comparison):
            detect_status.add("nothing")
            keypoints, bboxes = [], []
        elif len(keypoints) > 1 and comparison_args.for_comparison:
            detect_status.add("multiple")
            vid_info = get_video_info(input_path)
            h, w = vid_info["frame_height"], vid_info["frame_width"]
            cx, cy = w * 0.5, h * 0.5

            b = np.asarray(bboxes_, dtype=float)
            if b.ndim == 1:
                b = b[None, :]
            centers_x = (b[:, 0] + b[:, 2]) * 0.5
            centers_y = (b[:, 1] + b[:, 3]) * 0.5
            dists = (centers_x - cx) ** 2 + (centers_y - cy) ** 2
            i_center = int(np.argmin(dists))
            bbox_center = b[i_center]
            x1, y1, x2, y2 = bbox_center
            kp_arrays = [arr for _, arr in keypoints]

            def count_in_bbox(kp_arr):
                if kp_arr is None or len(kp_arr) == 0:
                    return -1
                kpx, kpy = kp_arr[:, 0], kp_arr[:, 1]
                inside = (kpx >= x1) & (kpx <= x2) & (kpy >= y1) & (kpy <= y2)
                return int(inside.sum())

            counts = [count_in_bbox(kp) for kp in kp_arrays]
            idx_best_kp = int(np.argmax(counts))
            kps = kp_arrays[idx_best_kp]
            keypoints = [[float(x), float(y), float(c)] for x, y, c in kps]
            bboxes = bbox_center.tolist()
        else:
            detect_status.add("one")
            keypoints = [[float(ele[1]), float(ele[0]), float(ele[2])] for ele in keypoints[0][1]]
            bboxes = bboxes_.tolist()[0]

        bbox_annotation.append(bboxes)
        normalized_keypoints = normalize_keypoints(keypoints, bboxes, num_nodes=data_args.num_nodes)
        bbox_pe_annotation.append(normalized_keypoints)

    # --- 裁切 + 光流 ---
    tmp_file_path = f"temp_{time_}.mp4"
    crop_and_save_video(
        input_path=input_path,
        bbox_annotation=bbox_annotation,
        output_path=tmp_file_path,
        resize_scale=of_args.input_model_size,
        do_crop=data_args.do_crop
    )

    result_of = optical_model.predict(input_path=tmp_file_path)
    os.remove(tmp_file_path)

    optical_flow_all_frames = result_of["optical_flows"]
    flows = np.stack(optical_flow_all_frames, axis=0)
    flows_f32 = flows.astype(np.float32)

    keypoints_arr = np.array(bbox_pe_annotation)
    output_path = os.path.join(np_save_dir, f"{id_:04d}.npz")
    np.savez_compressed(output_path,
                        keypoints=keypoints_arr,
                        optical_flows=flows_f32)

    exec_fps_pe = result_pe["stat"]["exec_fps"]
    keypoint_detection_rate = result_pe["stat"]["keypoint_detection_rate"]
    exec_fps_of = result_of["stat"]["exec_fps"]

    annotation["feature_file_path"] = output_path
    annotation["keypoint_detection_rate"] = keypoint_detection_rate

    txt = (f"Input:{os.path.basename(input_path)}, "
           f"(Pose Estimation)Execution FPS: {exec_fps_pe:.4f}, "
           f"(Pose Estimation)Keypoints Detection Rate: {keypoint_detection_rate:.4f}, "
           f"(Optical Flow)Execution FPS:{exec_fps_of:.4f}")
    txt += " Detect status: " + ', '.join(detect_status)
    logger.info(txt)

    return annotation, exec_fps_pe, keypoint_detection_rate, exec_fps_of

def main():
    # get parameters
    data_args = DataArguments()
    
    comparison_args = set_comparison_config_args()
    if comparison_args.for_comparison:
        data_args = set_comparison_config(data_args, comparison_args.dataset_name)
    
    pe_args, of_args = PoseEstimationArguments(), OpticalFlowArguments()
    pe_model = PoseEstimationModel(**asdict(pe_args))
    optical_model = OpticalFlowModel(**asdict(of_args))
    
    # check whether it use videos with or without video stabilization
    stabilized_dir = os.path.join(data_args.data_dir, data_args.stabilized_dir_name)
    seg_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    if (not data_args.skip_stabilization) and os.path.exists(stabilized_dir):
        videos_dir = stabilized_dir
    else:
        # use direct videos after segmented
        videos_dir = seg_dir
    
    # seg_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    video_paths = [os.path.join(videos_dir, ele) for ele in os.listdir(videos_dir) if ele.endswith('.mp4')]
    
    # get logger
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    
    # create for plot path
    time_ = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    if data_args.plot_pe:
        output_dir = ResultsArguments().output_dir
        plot_output_dir = os.path.join(output_dir, f"{data_args.feature_extract_dir_name}_{time_}")
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # record input
    logger.info(f"Input Directory: {videos_dir}")
    
    # set for annotation
    feature_extraction_dir = os.path.join(data_args.data_dir, data_args.feature_extract_dir_name)
    np_save_dir = os.path.join(feature_extraction_dir, "npz")
    os.makedirs(np_save_dir, exist_ok=True)
    
    # for each sample, it would be 
    # {"id":id, 
    # "action_id": action_id, 
    # "video_name": video_name,
    # "source_video_id": source video id in Youtube,
    # "data": [{"keypoints": keypoint at t0, "bboxes": bboxes at t0, ""}, {"keypoints": keypoint at t1, "bboxes": bboxes at t1}]...,
    # "keypoint_detection_rate": keypoint_detection_rate}
    annotations = [] 
    
    # start inference
    sum_exec_fps_pe, sum_keypoint_detection_rate = 0, 0
    sum_exec_fps_of = 0
    
    if comparison_args.for_comparison:
        video_path = os.path.join(data_args.data_dir, data_args.seg_dir_name)
        annotation_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "annotation")
        video_df = aggregate_video_level_labels(annotation_dir)
        
    for id_, input_path in enumerate(video_paths):
        try:
            annotation, exec_fps_pe, det_rate, exec_fps_of = process_single_video(
                input_path, id_, data_args, comparison_args, of_args,
                pe_model, optical_model, logger,
                video_df=video_df if comparison_args.for_comparison else None,
                plot_output_dir=plot_output_dir if data_args.plot_pe else None,
                time_=time_, np_save_dir=np_save_dir
            )
        except Exception as e:
            if videos_dir == stabilized_dir:
                alt_path = os.path.join(seg_dir, os.path.basename(input_path))
                try:
                    annotation, exec_fps_pe, det_rate, exec_fps_of = process_single_video(
                        alt_path, id_, data_args, comparison_args,
                        pe_model, optical_model, logger,
                        video_df=video_df if comparison_args.for_comparison else None,
                        plot_output_dir=plot_output_dir if data_args.plot_pe else None,
                        time_=time_, np_save_dir=np_save_dir
                    )
                except Exception as e2:
                    logger.error(f"Input:{os.path.basename(input_path)}, Error happens after fallback: {e2}")
                    continue
            else:
                logger.error(f"Input:{os.path.basename(input_path)}, Error happens: {e}")
                continue
        
        sum_exec_fps_pe += exec_fps_pe
        sum_exec_fps_of += exec_fps_of
        sum_keypoint_detection_rate += det_rate

        # finally, add to annotation
        annotations.append(annotation)
    txt = f"(Pose Estimation)Avg Execution FPS: {sum_exec_fps_pe / len(video_paths):.4f}, (Pose Estimation)Avg Keypoints Detection Rate: {sum_keypoint_detection_rate / len(video_paths):.4f}, (Optical Flow)Avg Execution FPS:{sum_exec_fps_of/len(video_paths):.4f} "
    logger.info(txt)
    
    # write annotation
    with open(os.path.join(feature_extraction_dir, data_args.annotation_file_name), 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    main()
    