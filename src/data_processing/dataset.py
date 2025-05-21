import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils.cli_args import DataArguments, ModelArguments, AugmentationArguments

# Assume these are correctly defined and imported
# from ..utils.cli_args import DataArguments, ModelArguments
# For standalone running, let's define dummy versions:
# from dataclasses import dataclass, field
# from typing import List, Literal

# @dataclass
# class DataArguments:
#     data_dir: str = "data"
#     trainsplit_dir_name: str = "train_val_splits" # Example
#     num_samples: int = 300 # T (frames)
#     # Add other necessary fields like yolo_pose_model_path, main_data_dir if used in _generate_keypoints
#     # For _generate_keypoints to work, these need to be actual paths
#     yolo_pose_model_path: str = "path/to/yolo.pt" # Placeholder
#     main_data_dir: str = "path/to/main_dataset"  # Placeholder
#     annotation_file_name: str = "annotations.json" # Example for _generate_keypoints

# @dataclass
# class ModelArguments:
#     num_nodes: int = 17 # num_joints
#     num_coords: int = 3 # x, y, confidence

# (AugmentationParams defined above)


class JointsDataset(Dataset):
    def __init__(self,
                 d_params: DataArguments,
                 aug_params: AugmentationArguments,
                 model_params: ModelArguments,
                 istrain: bool = True,
                 fold_num: int = 0):
        super().__init__()
        self.d_params = d_params
        self.aug_params = aug_params
        self.model_params = model_params

        # For _generate_keypoints, these paths need to be valid
        # Ensure they are properly set through d_params or model_params
        # self.yolo_pose_model_path = getattr(d_params, 'yolo_pose_model_path', 'path/to/yolo.pt') # Example
        # self.main_data_dir = getattr(d_params, 'main_data_dir', 'path/to/main_dataset') # Example

        self.datatype = "train" if istrain else "val"
        # Adjust trainsplit_dir_name if it's not in your d_params
        trainsplit_dir_name = getattr(d_params, 'trainsplit_dir_name', 'train_split')
        self.annotation_path = os.path.join(self.d_params.data_dir, trainsplit_dir_name, f"annotation_fold{str(fold_num)}_{self.datatype}_windows.json")
        
        self.T = self.d_params.num_samples
        self.num_joints = self.d_params.num_nodes
        self.num_coords = self.d_params.num_coords # Should be 3 (x,y,conf)
        
        self._load_annotation() # This might call _generate_keypoints

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load initial data from annotations
        # Assuming 'data' contains a list of frames, each frame is a list of keypoints [x,y,c]
        # And 'action_id' is the label
        keypoints_data_list = self.annotations[index]["data"] # This should be (T, num_joints, 3)
        label = torch.tensor(self.annotations[index]["action_id"], dtype=torch.long)

        data_np = np.array(keypoints_data_list, dtype=np.float32)

        # Validate shape
        expected_shape = (self.T, self.num_joints, self.num_coords)
        if data_np.shape != expected_shape:
            # Attempt to reshape or pad/truncate if necessary, or raise an error
            # This part is crucial and depends on how your data is structured if not uniform
            print(f"Warning: Sample {index} has shape {data_np.shape}, expected {expected_shape}. Attempting to fix or skip.")
            # Example: if T is wrong, pad or truncate (simple padding with zeros)
            if data_np.shape[0] < self.T:
                padding = np.zeros((self.T - data_np.shape[0], self.num_joints, self.num_coords), dtype=np.float32)
                data_np = np.concatenate((data_np, padding), axis=0)
            elif data_np.shape[0] > self.T:
                data_np = data_np[:self.T, :, :]
            
            if data_np.shape[1] != self.num_joints or data_np.shape[2] != self.num_coords:
                 raise ValueError(f"Corrected sample {index} shape {data_np.shape} still incorrect. Expected ({self.T}, {self.num_joints}, {self.num_coords}). Data: {keypoints_data_list}")

        # Data augmentation
        if self.aug_params.augment and self.datatype == "train": # Usually only augment training data
            # Geometric augmentations operate on a copy to modify x,y coordinates
            # They need access to confidence for proper center calculation and masking
            
            current_data_np = data_np.copy()

            if random.random() < 0.5 and self.aug_params.rot_max > 0:
                angle = random.uniform(-self.aug_params.rot_max, self.aug_params.rot_max)
                current_data_np = self._rotate(current_data_np, angle)

            if random.random() < 0.5 and self.aug_params.scale_min < self.aug_params.scale_max:
                scale = random.uniform(self.aug_params.scale_min, self.aug_params.scale_max)
                current_data_np = self._scale(current_data_np, scale)

            if random.random() < 0.5 and self.aug_params.trans_max > 0:
                current_data_np = self._translate(current_data_np, trans_max=self.aug_params.trans_max)

            if random.random() < 0.5 and self.aug_params.noise_std > 0:
                current_data_np = self._add_noise(current_data_np, self.aug_params.noise_std)

            if random.random() < 0.3 and self.aug_params.shear_max > 0:
                shear_factor = random.uniform(-self.aug_params.shear_max, self.aug_params.shear_max)
                sx = shear_factor if random.random() < 0.5 else 0.0
                sy = shear_factor if random.random() < 0.5 else 0.0
                current_data_np = self._shear(current_data_np, sx, sy)
            
            data_np = current_data_np # Assign back the geometrically augmented data

            # Temporal and dropout augmentations operate on the (potentially geometrically augmented) data_np
            if self.aug_params.temporal_jitter_prob > 0:
                data_np = self._temporal_jitter(data_np, self.aug_params.temporal_jitter_prob)

            if self.aug_params.joint_drop_prob > 0:
                data_np = self._joint_dropout(data_np, self.aug_params.joint_drop_prob)

            if self.aug_params.frame_drop_prob > 0:
                data_np = self._frame_dropout(data_np, self.aug_params.frame_drop_prob)

        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        # Permute to (C, T, V) -> Channels (coords), Time (frames), Vertices (joints)
        data_tensor = data_tensor.permute(2, 0, 1).contiguous()

        return data_tensor, label

    def _get_center(self, keypoints_with_conf: np.ndarray) -> np.ndarray:
        """
        Calculates the center of valid keypoints based on confidence.
        keypoints_with_conf shape: (..., num_joints, 3) where last dim is [x, y, conf]
        """
        points_xy = keypoints_with_conf[..., :2]
        confidences = keypoints_with_conf[..., 2]
        
        # Flatten to (N, 2) for points and (N,) for confidences
        flat_points_xy = points_xy.reshape(-1, 2)
        flat_confidences = confidences.reshape(-1)

        valid_mask = flat_confidences > self.aug_params.valid_kpt_confidence_thresh
        valid_points = flat_points_xy[valid_mask]

        if valid_points.shape[0] > 0:
            center = np.mean(valid_points, axis=0)
        else:
            # Fallback center if no valid points.
            # For normalized data [0,1], (0.5, 0.5) might be a sensible default center.
            # Or, if normalization is around (0,0), then (0,0) is fine.
            # This depends on your normalization scheme. Assuming [0,1] for now.
            center = np.array([0.5, 0.5], dtype=np.float32) if np.any(points_xy > 0) else np.array([0.0, 0.0], dtype=np.float32)
        return center

    def _common_preprocess_for_augmentation(self, data_slice_with_conf: np.ndarray):
        """
        Common preprocessing for geometric augmentations, uses confidence for valid points.
        data_slice_with_conf: (T, num_joints, 3) or (num_joints, 3) for a single frame.
        Returns center, original_xy_shape, flat_xy_coords, valid_mask_flat.
        """
        points_xy_slice = data_slice_with_conf[..., :2] # Shape (T, num_joints, 2) or (num_joints, 2)
        original_xy_shape = points_xy_slice.shape
        
        center = self._get_center(data_slice_with_conf) # Center is calculated based on confidences

        flat_xy_coords = points_xy_slice.reshape(-1, 2) # Shape (T*num_joints, 2)
        
        confidences_flat = data_slice_with_conf[..., 2].reshape(-1) # Shape (T*num_joints,)
        valid_mask_flat = confidences_flat > self.aug_params.valid_kpt_confidence_thresh
        
        return center, original_xy_shape, flat_xy_coords, valid_mask_flat

    def _apply_transform_to_valid_points(self, flat_xy_coords, transformed_valid_points, valid_mask_flat, original_xy_shape):
        """Helper to apply transformations back only to valid points."""
        output_flat_xy = flat_xy_coords.copy()
        if np.any(valid_mask_flat): # Check if there are any valid points
            output_flat_xy[valid_mask_flat] = transformed_valid_points
        return output_flat_xy.reshape(original_xy_shape)

    def _rotate(self, data_slice_with_conf: np.ndarray, angle: float) -> np.ndarray:
        center, original_xy_shape, flat_xy, valid_mask = self._common_preprocess_for_augmentation(data_slice_with_conf)
        
        output_data_slice = data_slice_with_conf.copy()
        points_xy_to_transform = flat_xy[valid_mask] # Only get valid points for transformation

        if points_xy_to_transform.shape[0] == 0: # No valid points to rotate
            return output_data_slice

        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        
        translated_points = points_xy_to_transform - center
        rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
        final_rotated_points = rotated_translated_points + center
        
        # Place rotated points back into the structure
        new_points_xy_flat = flat_xy.copy()
        new_points_xy_flat[valid_mask] = final_rotated_points
        
        output_data_slice[..., :2] = new_points_xy_flat.reshape(original_xy_shape)
        return output_data_slice

    def _scale(self, data_slice_with_conf: np.ndarray, scale_factor: float) -> np.ndarray:
        center, original_xy_shape, flat_xy, valid_mask = self._common_preprocess_for_augmentation(data_slice_with_conf)

        output_data_slice = data_slice_with_conf.copy()
        points_xy_to_transform = flat_xy[valid_mask]

        if points_xy_to_transform.shape[0] == 0:
            return output_data_slice
            
        scaled_points = (points_xy_to_transform - center) * scale_factor + center
        
        new_points_xy_flat = flat_xy.copy()
        new_points_xy_flat[valid_mask] = scaled_points
        
        output_data_slice[..., :2] = new_points_xy_flat.reshape(original_xy_shape)
        return output_data_slice

    def _translate(self, data_slice_with_conf: np.ndarray, trans_max: float) -> np.ndarray:
        center, original_xy_shape, flat_xy, valid_mask = self._common_preprocess_for_augmentation(data_slice_with_conf)

        output_data_slice = data_slice_with_conf.copy()
        points_xy_to_transform = flat_xy[valid_mask]
        
        if points_xy_to_transform.shape[0] == 0:
            return output_data_slice

        # Calculate bbox_wh based on *valid* points only
        bbox_wh = np.ptp(points_xy_to_transform, axis=0) if points_xy_to_transform.shape[0] > 1 else np.array([1.0, 1.0])
        # Handle case where bbox_wh might be zero if all valid points are identical
        bbox_wh[bbox_wh < 1e-6] = 1.0 # Avoid division by zero or tiny translations
        
        offset_x = random.uniform(-trans_max, trans_max) * bbox_wh[0]
        offset_y = random.uniform(-trans_max, trans_max) * bbox_wh[1]
        
        translated_points = points_xy_to_transform + np.array([offset_x, offset_y], dtype=np.float32)
        
        new_points_xy_flat = flat_xy.copy()
        new_points_xy_flat[valid_mask] = translated_points
        
        output_data_slice[..., :2] = new_points_xy_flat.reshape(original_xy_shape)
        return output_data_slice

    def _add_noise(self, data_slice_with_conf: np.ndarray, noise_std_factor: float) -> np.ndarray:
        center, original_xy_shape, flat_xy, valid_mask = self._common_preprocess_for_augmentation(data_slice_with_conf)

        output_data_slice = data_slice_with_conf.copy()
        points_xy_to_transform = flat_xy[valid_mask]

        if points_xy_to_transform.shape[0] == 0:
            return output_data_slice

        # Use std dev of valid points as scale reference for noise
        # This makes noise relative to the current spread of *valid* normalized keypoints
        pose_std_dev = np.std(points_xy_to_transform, axis=0) if points_xy_to_transform.shape[0] > 1 else np.array([0.1, 0.1]) # Default if too few points
        noise_scale = np.mean(pose_std_dev) if np.mean(pose_std_dev) > 1e-6 else 0.1 # Avoid zero or tiny scale
        
        # Generate noise only for the x,y coordinates and for all T*num_joints points initially
        noise = np.random.normal(0, noise_std_factor * noise_scale, size=flat_xy.shape).astype(np.float32)
        
        noisy_points_flat = flat_xy.copy()
        # Apply noise only to valid keypoints
        noisy_points_flat[valid_mask] += noise[valid_mask] 
        
        output_data_slice[..., :2] = noisy_points_flat.reshape(original_xy_shape)
        return output_data_slice

    def _shear(self, data_slice_with_conf: np.ndarray, sx: float = 0.0, sy: float = 0.0) -> np.ndarray:
        center, original_xy_shape, flat_xy, valid_mask = self._common_preprocess_for_augmentation(data_slice_with_conf)

        output_data_slice = data_slice_with_conf.copy()
        points_xy_to_transform = flat_xy[valid_mask]
        
        if points_xy_to_transform.shape[0] == 0:
            return output_data_slice

        shear_matrix = np.array([[1, sx], [sy, 1]], dtype=np.float32)
        
        translated_points = points_xy_to_transform - center
        sheared_translated_points = np.dot(translated_points, shear_matrix.T)
        final_sheared_points = sheared_translated_points + center
        
        new_points_xy_flat = flat_xy.copy()
        new_points_xy_flat[valid_mask] = final_sheared_points
        
        output_data_slice[..., :2] = new_points_xy_flat.reshape(original_xy_shape)
        return output_data_slice

    # Dropout and Temporal Jitter methods remain the same as they operate on data_np
    # and their logic for zeroing out or copying frames is independent of normalization scale,
    # assuming confidence=0 marks invalid/dropped keypoints.

    def _joint_dropout(self, data_np_slice: np.ndarray, drop_prob: float) -> np.ndarray:
        output_data = data_np_slice.copy()
        if random.random() < drop_prob:
            num_total_joints = output_data.shape[1] # self.num_joints
            # Drop a random fraction of joints up to drop_prob
            num_drop = int(num_total_joints * random.uniform(0.01, drop_prob)) # Ensure at least one if prob met
            if num_drop > 0:
                drop_indices = random.sample(range(num_total_joints), num_drop)
                output_data[:, drop_indices, :] = 0 # Set x, y, confidence to 0
        return output_data

    def _frame_dropout(self, data_np_slice: np.ndarray, drop_prob: float) -> np.ndarray:
        output_data = data_np_slice.copy()
        if random.random() < drop_prob:
            num_total_frames = output_data.shape[0] # self.T
            num_drop = int(num_total_frames * random.uniform(0.01, drop_prob))
            if num_drop > 0:
                drop_indices = random.sample(range(num_total_frames), num_drop)
                output_data[drop_indices, :, :] = 0 # Set x, y, confidence to 0 for entire frames
        return output_data

    def _temporal_jitter(self, data_np_slice: np.ndarray, jitter_prob: float) -> np.ndarray:
        output_data = data_np_slice.copy()
        T = output_data.shape[0]
        if random.random() < jitter_prob:
            # Jitter a small percentage of frames
            num_jitter_frames = int(T * random.uniform(0.01, jitter_prob))
            if num_jitter_frames > 0 and T > 2:
                # Ensure indices are within bounds for replacement
                # Can only replace frames from index 1 to T-2 if we use idx-1 and idx+1
                possible_indices = list(range(1, T - 1))
                if not possible_indices: # Not enough frames to jitter (e.g. T=1 or T=2)
                    return output_data

                num_jitter_frames = min(num_jitter_frames, len(possible_indices))
                jitter_indices = random.sample(possible_indices, num_jitter_frames)

                for idx in jitter_indices:
                    if random.random() < 0.5: # Replace with previous frame
                        output_data[idx, :, :] = output_data[idx - 1, :, :]
                    else: # Replace with next frame
                        output_data[idx, :, :] = output_data[idx + 1, :, :]
        return output_data
        
    def _load_annotation(self):
        # This is a placeholder for your actual annotation loading.
        # Ensure self.yolo_pose_model_path and self.main_data_dir are set in __init__ if _generate_keypoints is called.
        # Example for yolo_pose_model_path and main_data_dir (should come from d_params or model_params):
        self.yolo_pose_model_path = getattr(self.d_params, 'yolo_pose_model_path', None)
        self.main_data_dir = getattr(self.d_params, 'main_data_dir', None)


        if not os.path.exists(self.annotation_path):
            # Check if necessary paths for generation are provided
            if not self.yolo_pose_model_path or not self.main_data_dir or \
               not os.path.exists(self.yolo_pose_model_path) or not os.path.exists(self.main_data_dir):
                raise ValueError(
                    f"Annotation file '{self.annotation_path}' not found and cannot generate: "
                    "Please provide valid 'yolo_pose_model_path' and 'main_data_dir' in DataArguments, "
                    "or ensure the annotation file exists."
                )
            print(f"Annotation file not found at {self.annotation_path}. Attempting to generate keypoints from {self.main_data_dir}...")
            self._generate_keypoints() # This will populate self.annotations
            print(f"Annotations generated and saved to {self.annotation_path}")
        else:
            print(f"Loading annotations from {self.annotation_path}...")
            try:
                with open(self.annotation_path, 'r') as f:
                    self.annotations = json.load(f)
                print(f"Loaded {len(self.annotations)} samples.")
                if not self.annotations:
                    raise ValueError(f"Annotation file {self.annotation_path} is empty or malformed after loading.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {self.annotation_path}: {e}")


    def _generate_keypoints(self):
        """
        Placeholder for your keypoint generation logic.
        This method should populate self.annotations with data.
        Each item in self.annotations should be a dictionary like:
        {"data": list_of_frames, "action_id": label}
        where list_of_frames is a list (length T) of lists (length num_joints)
        of [x, y, confidence] values.
        """
        # Ensure ultralytics is installed if this part is used
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics (YOLO) package is not installed. Please install it to generate keypoints.")

        # make dirs for the new annotation file
        os.makedirs(os.path.dirname(self.annotation_path), exist_ok=True)
        
        print(f"Initializing YOLO pose model from: {self.yolo_pose_model_path}")
        yolo_pose_model = YOLO(self.yolo_pose_model_path)
        
        self.annotations = []
        # Assuming main_data_dir and d_params.annotation_file_name are for the source annotations
        main_annotation_file = getattr(self.d_params, 'annotation_file_name', 'default_main_annotation.json')
        main_annotation_path = os.path.join(self.main_data_dir, main_annotation_file)
        
        if not os.path.exists(main_annotation_path):
            raise FileNotFoundError(f"Main annotation file for generation not found: {main_annotation_path}")

        with open(main_annotation_path, 'r') as f:
            source_annotations_all_types = json.load(f)
        
        # Ensure the key self.datatype (e.g., "train" or "val") exists in the loaded annotations
        if self.datatype not in source_annotations_all_types:
            raise KeyError(f"Data type '{self.datatype}' not found in source annotation file: {main_annotation_path}. Available keys: {list(source_annotations_all_types.keys())}")

        source_annotations_for_current_type = source_annotations_all_types[self.datatype]
        
        print(f"Processing {len(source_annotations_for_current_type)} video entries for '{self.datatype}' set.")

        for i, data_entry in enumerate(source_annotations_for_current_type):
            keypoint_sequence_for_video = []
            # Ensure keys exist in data_entry
            category = data_entry.get("category", "unknown_category")
            action = data_entry.get("action", "unknown_action")
            video_name = data_entry.get("video_name")
            action_id = data_entry.get("action_id", -1) # Use a default if not present

            if not video_name:
                print(f"Warning: video_name missing in source annotation entry {i}. Skipping.")
                continue
            if action_id == -1:
                 print(f"Warning: action_id missing in source annotation entry for {video_name}. Skipping or using default.")
                 # Decide if you want to skip or assign a default label

            video_path = os.path.join(self.main_data_dir, self.datatype, category, action, video_name)
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found at {video_path} for entry {i}. Skipping.")
                continue

            print(f"Generating keypoints for: {video_path}")
            # Process video with YOLO. Adjust batch size as needed.
            # stream=True might be more memory efficient for long videos.
            results_generator = yolo_pose_model(video_path, batch=16, stream=True, verbose=False) 
            
            frames_processed_yolo = 0
            for result in results_generator:
                frames_processed_yolo += 1
                # result.keypoints.data is a tensor.
                # It can be empty if no person is detected.
                # It can have multiple persons. For ST-GCN usually one person is focused on.
                # Assuming we take the first detected person's keypoints if multiple.
                if result.keypoints.data.numel() > 0: # Check if tensor is not empty
                    # .data gives the tensor, .cpu() moves to CPU, .tolist() converts
                    # result.keypoints.data shape is [num_persons, num_keypoints, xy_or_xyc]
                    # We need to select one person, typically the first or most confident.
                    # For simplicity, taking the first person [0]
                    person_keypoints_tensor = result.keypoints.data[0].cpu() # Shape [num_keypoints, xyc]
                    
                    # Ensure it has 3 channels (x, y, conf). If only 2 (x,y), add conf=1.0
                    if person_keypoints_tensor.shape[1] == 2: # Only x, y
                        confidences = torch.ones(person_keypoints_tensor.shape[0], 1, dtype=person_keypoints_tensor.dtype)
                        person_keypoints_tensor = torch.cat((person_keypoints_tensor, confidences), dim=1)

                    keypoints_list_for_frame = person_keypoints_tensor.tolist()
                    
                    # Optional: replace (0,0,c) keypoints with (0,0,0) if (0,0) means invalid for your raw data
                    # keypoints_list_for_frame = [
                    #     [0.0, 0.0, 0.0] if kp[0] == 0 and kp[1] == 0 else kp
                    #     for kp in keypoints_list_for_frame
                    # ]
                else:
                    # No person detected, or no keypoints. Pad with zeros.
                    keypoints_list_for_frame = [[0.0, 0.0, 0.0]] * self.num_joints
                
                keypoint_sequence_for_video.append(keypoints_list_for_frame)
            
            print(f"  YOLO processed {frames_processed_yolo} frames for {video_name}. Extracted {len(keypoint_sequence_for_video)} keypoint sets.")

            if len(keypoint_sequence_for_video) < self.d_params.min_frames_for_sample: # Add a min_frames check
                 print(f"  Skipping {video_name}: not enough frames ({len(keypoint_sequence_for_video)} / {self.d_params.min_frames_for_sample})")
                 continue

            # Uniformly sample T frames
            if len(keypoint_sequence_for_video) > self.T:
                indices = np.linspace(0, len(keypoint_sequence_for_video) - 1, num=self.T, dtype=int)
                sampled_keypoints = [keypoint_sequence_for_video[i] for i in indices]
            elif len(keypoint_sequence_for_video) < self.T: # Padding if too short
                sampled_keypoints = keypoint_sequence_for_video
                padding_needed = self.T - len(sampled_keypoints)
                pad_frame = [[0.0, 0.0, 0.0]] * self.num_joints # Pad with zero confidence keypoints
                for _ in range(padding_needed):
                    sampled_keypoints.append(pad_frame)
            else: # Exactly T frames
                sampled_keypoints = keypoint_sequence_for_video
            
            if len(sampled_keypoints) != self.T:
                print(f"Error: Sampled keypoints for {video_name} have incorrect frame count: {len(sampled_keypoints)}. Expected: {self.T}")
                continue

            self.annotations.append({"data": sampled_keypoints, "action_id": action_id, "video_name": video_name})
            print(f"  Added sample for {video_name} with {len(sampled_keypoints)} frames.")

        if not self.annotations:
            raise ValueError(f"No annotations were generated for datatype '{self.datatype}'. Check video paths and YOLO processing.")

        with open(self.annotation_path, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Saved {len(self.annotations)} generated annotations to {self.annotation_path}")