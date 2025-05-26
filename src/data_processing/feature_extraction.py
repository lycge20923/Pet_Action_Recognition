from dataclasses import asdict
import os
import time
from datetime import datetime
import logging
import json
import numpy as np
import cv2

from ..utils.cli_args import DataArguments, PoseEstimationArguments, OpticalFlowArguments, OutputArguments
from features.pose_estimation import PoseEstimationModel
from features.optical_flow import OpticalFlowModel
from ..utils.logging_utils import setup_logger

def crop_and_save_video(input_path: str,
                        bbox_annotations: list,
                        output_path: str,
                        resize_scale: tuple):
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

        bbox = bbox_annotations[frame_idx].get("bboxes", None)
        if not bbox:
            bbox = prev_bbox
        else:
            prev_bbox = bbox

        if not bbox:
            crop_img = np.zeros((resize_scale[1], resize_scale[0], 3), dtype=np.uint8)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]

            # scaling and cropping
            ch, cw = crop.shape[:2]
            scale = min(resize_scale[0] / cw, resize_scale[1] / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            resized = cv2.resize(crop, (new_w, new_h))

            pad_w = resize_scale[0] - new_w
            pad_h = resize_scale[1] - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            crop_img = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

        writer.write(crop_img)
        frame_idx += 1

    cap.release()
    writer.release()

def main():
    # get parameters
    data_args = DataArguments()
    pe_args, of_args = PoseEstimationArguments(), OpticalFlowArguments()
    pe_model = PoseEstimationModel(**asdict(pe_args))
    optical_model = OpticalFlowModel(**asdict(of_args))
    
    # check whether it use videos with or without video stabilization
    stabilized_dir = os.path.join(data_args.data_dir, data_args.stabilized_dir_name)
    if (not data_args.skip_stabilization) and os.path.exists(stabilized_dir):
        videos_dir = stabilized_dir
    else:
        # use direct videos after segmented
        videos_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    # seg_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    video_paths = [os.path.join(videos_dir, ele) for ele in os.listdir(videos_dir) if ele.endswith('.mp4')]
    
    # get logger
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    
    # create for plot path
    if data_args.plot_pe:
        output_dir = OutputArguments().output_dir
        time_ = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
        plot_output_dir = os.path.join(output_dir, f"{data_args.feature_extract_dir_name}_{time_}")
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # record input
    logger.info(f"Input Directory: {videos_dir}")
    
    # set for annotation
    pe_dir = os.path.join(data_args.data_dir, data_args.feature_extract_dir_name)
    of_dir = os.path.join(pe_dir, data_args.optical_flow_sub_dir_name)
    os.makedirs(of_dir, exist_ok=True)
    
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
    for id_, input_path in enumerate(video_paths):
        try:
            video_name = os.path.basename(input_path)
            split_ = video_name.split("_")
            source_video_id, action_id = int(split_[0]),  data_args.actions.index(split_[1])
            annotation = {"id": id_, 
                        "action_id":action_id, 
                        "video_name": video_name, 
                        "source_video_id": source_video_id}
            
            # --- pose estimation ---
            if data_args.plot_pe:
                result_pe = pe_model.predict(input_path=input_path, 
                                        plot_path=os.path.join(plot_output_dir, os.path.basename(input_path)),
                                        plot_threshold=data_args.plot_pe_threshold)
            else:
                result_pe = pe_model.predict(input_path=input_path)
            
            # write pose estimation 
            keypoints_all_frames, bboxes_all_frames = result_pe["keypoints"], result_pe["bboxes"]
            bbox_pe_annotation = []
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
                bbox_pe_annotation.append({"keypoints": keypoints, "bboxes": bboxes})
            
            # --- bbox crop, resize, and store ---
            # output
            tmp_file_path = "temp.mp4"
            crop_and_save_video(
                input_path=input_path,
                bbox_annotations=bbox_pe_annotation,
                output_path=tmp_file_path,
                resize_scale=of_args.input_model_size
            )
                        
            # --- optical flow ---
            result_of = optical_model.predict(input_path=tmp_file_path)
            os.remove(tmp_file_path)
            
            # write optical flow
            optical_flow_all_frames = result_of["optical_flows"]
            flows = np.stack(optical_flow_all_frames, axis=0)  
            min_val = flows.min()
            max_val = flows.max()
            scale   = max_val - min_val + 1e-8
            flows_uint8 = ((flows - min_val) / scale * 255.0).astype(np.uint8)
            np.savez_compressed(os.path.join(of_dir, f"{id_:04d}.npz"),
                    flows=flows_uint8,
                    min_val=min_val,
                    max_val=max_val)
            
            exec_fps_pe = result_pe["stat"]["exec_fps"]
            keypoint_detection_rate = result_pe["stat"]["keypoint_detection_rate"]
            
            exec_fps_of = result_of["stat"]["exec_fps"]
            
            # write annotation
            annotation["data"] = bbox_pe_annotation
            annotation["keypoint_detection_rate"] = keypoint_detection_rate
            
            txt = f"Input:{os.path.basename(input_path)}, (Pose Estimation)Execution FPS: {exec_fps_pe:.4f}, (Pose Estimation)Keypoints Detection Rate: {keypoint_detection_rate:.4f}, (Optical Flow)Execution FPS:{exec_fps_of:.4f}"
            logger.info(txt)
            sum_exec_fps_pe += exec_fps_pe
            sum_exec_fps_of += exec_fps_of
            sum_keypoint_detection_rate += keypoint_detection_rate

            # finally, add to annotation
            annotations.append(annotation)
        except Exception as e:
            logger.error(f"Input:{os.path.basename(input_path)}, Error happens: {e}")
        
    txt = f"(Pose Estimation)Avg Execution FPS: {sum_exec_fps_pe / len(video_paths):.4f}, (Pose Estimation)Avg Keypoints Detection Rate: {sum_keypoint_detection_rate / len(video_paths):.4f}, (Optical Flow)Avg Execution FPS:{sum_exec_fps_of/len(video_paths):.4f} "
    logger.info(txt)
    
    # write annotation
    with open(os.path.join(pe_dir, data_args.annotation_file_name), 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    main()
    