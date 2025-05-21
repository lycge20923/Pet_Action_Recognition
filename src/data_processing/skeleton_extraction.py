from dataclasses import asdict
import os
import time
from datetime import datetime
import logging
import json

from ..utils.cli_args import DataArguments, PoseEstimationArguments, OutputArguments
from features.pose_estimation import PoseEstimationModel
from ..utils.logging_utils import setup_logger

def main():
    # get parameters
    data_args = DataArguments()
    pe_args = PoseEstimationArguments()
    pe_model = PoseEstimationModel(**asdict(pe_args))
    seg_dir = os.path.join(data_args.data_dir, data_args.seg_dir_name)
    seg_video_paths = [os.path.join(seg_dir, ele) for ele in os.listdir(seg_dir) if ele.endswith('.mp4')]
    
    # get logger
    logger = setup_logger(file_path=__file__, level=logging.INFO)
    
    # create for plot path
    if data_args.plot_pe:
        output_dir = OutputArguments().output_dir
        time_ = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
        plot_output_dir = os.path.join(output_dir, f"{data_args.pe_dir_name}_{time_}")
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # record input
    logger.info(f"Input Directory: {seg_dir}")
    
    # set for annotation
    pe_dir = os.path.join(data_args.data_dir, data_args.pe_dir_name)
    os.makedirs(pe_dir, exist_ok=True)
    
    # for each sample, it would be 
    # {"id":id, 
    # "action_id": action_id, 
    # "video_name": video_name,
    # "source_video_id": source video id in Youtube,
    # "data": [{"keypoints": keypoint at t0, "bboxes": bboxes at t0}, {"keypoints": keypoint at t1, "bboxes": bboxes at t1}]...,
    # "keypoint_detection_rate": keypoint_detection_rate}
    annotations = [] 
    
    # start inference
    sum_exec_fps, sum_keypoint_detection_rate =0, 0
    for id_, input_path in enumerate(seg_video_paths):
        try:
            video_name = os.path.basename(input_path)
            split_ = video_name.split("_")
            source_video_id, action_id = int(split_[0]),  data_args.actions.index(split_[1])
            annotation = {"id": id_, 
                        "action_id":action_id, 
                        "video_name": video_name, 
                        "source_video_id": source_video_id}
            
            
            if data_args.plot_pe:
                result = pe_model.predict(input_path=input_path, 
                                        plot_path=os.path.join(plot_output_dir, os.path.basename(input_path)),
                                        plot_threshold=data_args.plot_pe_threshold)
            else:
                result = pe_model.predict(input_path=input_path)
            
            # write to annotation
            keypoints_all_frames, bboxes_all_frames = result["keypoints"], result["bboxes"]
            data_annotation = []
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
                    
                data_annotation.append({"keypoints": keypoints, "bboxes": bboxes})
            
            exec_fps = result["stat"]["exec_fps"]
            keypoint_detection_rate = result["stat"]["keypoint_detection_rate"]
            
            # write annotation
            annotation["data"] = data_annotation
            annotation["keypoint_detection_rate"] = keypoint_detection_rate
            
            txt = f"Input:{os.path.basename(input_path)}, Execution FPS: {exec_fps:.4f}, Keypoints Detection Rate: {keypoint_detection_rate}"
            logger.info(txt)
            sum_exec_fps += exec_fps
            sum_keypoint_detection_rate += keypoint_detection_rate

            # finally, add to annotation
            annotations.append(annotation)
        except Exception as e:
            logger.error(f"Input:{os.path.basename(input_path)}, Error happens: {e}")
        
    txt = f"Avg Execution FPS: {sum_exec_fps / len(seg_video_paths):.2f}, Avg Keypoints Detection Rate: {sum_keypoint_detection_rate / len(seg_video_paths):.2f}"
    logger.info(txt)
    
    # write annotation
    with open(os.path.join(pe_dir, data_args.annotation_file_name), 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    main()
    