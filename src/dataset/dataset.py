import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils.cli_args import DataArguments, ModelArguments, AugmentationArguments


class KpOfDataset(Dataset):
    def __init__(self,
                 data_params: DataArguments,
                 aug_params: AugmentationArguments,
                 model_params: ModelArguments,
                 istrain: bool = True,
                 fold_num: int = 0):
        super().__init__()
        self.data_params = data_params
        self.aug_params = aug_params
        self.model_params = model_params
        self.add_optical_flow = model_params.add_optical_flow
        self.only_optical_flow = model_params.only_optical_flow

        self.datatype = "train" if istrain else "val"
        # Adjust trainsplit_dir_name if it's not in your data_params
        trainsplit_dir_name = getattr(data_params, 'trainsplit_dir_name', 'train_split')
        annotation_path = os.path.join(self.data_params.data_dir, trainsplit_dir_name, "annotation_windows_metadata.json")
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        if istrain:
            self.annotations = [sample for sample in self.annotations if sample['fold'] != fold_num]
        else:
            self.annotations = [sample for sample in self.annotations if sample['fold'] == fold_num]
        
        self.T = self.data_params.num_samples
        self.num_joints = self.data_params.num_nodes
        self.num_coords = self.data_params.num_coords # Should be 3 (x,y,conf)
        
        # self._load_annotation() # This might call _generate_keypoints

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load initial data from annotations
        # And 'action_id' is the label
        # load keypoints from the .npz window file
        feat_path = self.annotations[index]["feature_file"]
        data = np.load(feat_path)
        
        # --- load feature and label --- 
        # label
        label = torch.tensor(self.annotations[index]["action_id"], dtype=torch.long)
        
        optical_flows_tensor = None
        keypoints_tensor = None 
        
        # optical flows
        if self.only_optical_flow or self.add_optical_flow:
            optical_flows_np = data["optical_flows"].astype(np.float32)
            optical_flows_np = optical_flows_np.squeeze(axis=1)
            optical_flows_np = optical_flows_np.transpose(1, 0, 2, 3)
            optical_flows_tensor = torch.from_numpy(optical_flows_np).float()
        if self.only_optical_flow:
            return keypoints_tensor, optical_flows_tensor, label
        
        # keypoints
        keypoints_np = data["keypoints"].astype(np.float32)   # shape (T, V, 3)
        
        # Validate shape
        expected_shape = (self.T, self.num_joints, self.num_coords)
        if keypoints_np.shape != expected_shape:
            
            # This part is crucial and depends on how your data is structured if not uniform
            print(f"Warning: Sample {index} has shape {keypoints_np.shape}, expected {expected_shape}. Attempting to fix or skip.")
            # Example: if T is wrong, pad or truncate (simple padding with zeros)
            if keypoints_np.shape[0] < self.T:
                padding = np.zeros((self.T - keypoints_np.shape[0], self.num_joints, self.num_coords), dtype=np.float32)
                keypoints_np = np.concatenate((keypoints_np, padding), axis=0)
            elif keypoints_np.shape[0] > self.T:
                keypoints_np = keypoints_np[:self.T, :, :]
            
            if keypoints_np.shape[1] != self.num_joints or keypoints_np.shape[2] != self.num_coords:
                 raise ValueError(f"Corrected sample {index} shape {keypoints_np.shape} still incorrect. Expected ({self.T}, {self.num_joints}, {self.num_coords}). Data: {keypoints_np}")

        # Data augmentation
        if self.aug_params.augment and self.datatype == "train": # Usually only augment training data
            # Geometric augmentations operate on a copy to modify x,y coordinates
            # They need access to confidence for proper center calculation and masking
            
            current_keypoints_np = keypoints_np.copy()

            if random.random() < 0.5 and self.aug_params.rot_max > 0:
                angle = random.uniform(-self.aug_params.rot_max, self.aug_params.rot_max)
                current_keypoints_np = self._rotate(current_keypoints_np, angle)

            if random.random() < 0.5 and self.aug_params.scale_min < self.aug_params.scale_max:
                scale = random.uniform(self.aug_params.scale_min, self.aug_params.scale_max)
                current_keypoints_np = self._scale(current_keypoints_np, scale)

            if random.random() < 0.5 and self.aug_params.trans_max > 0:
                current_keypoints_np = self._translate(current_keypoints_np, trans_max=self.aug_params.trans_max)

            if random.random() < 0.5 and self.aug_params.noise_std > 0:
                current_keypoints_np = self._add_noise(current_keypoints_np, self.aug_params.noise_std)

            if random.random() < 0.3 and self.aug_params.shear_max > 0:
                shear_factor = random.uniform(-self.aug_params.shear_max, self.aug_params.shear_max)
                sx = shear_factor if random.random() < 0.5 else 0.0
                sy = shear_factor if random.random() < 0.5 else 0.0
                current_keypoints_np = self._shear(current_keypoints_np, sx, sy)
            
            keypoints_np = current_keypoints_np # Assign back the geometrically augmented data

            # Temporal and dropout augmentations operate on the (potentially geometrically augmented) keypoints_np
            if self.aug_params.temporal_jitter_prob > 0:
                keypoints_np = self._temporal_jitter(keypoints_np, self.aug_params.temporal_jitter_prob)

            if self.aug_params.joint_drop_prob > 0:
                keypoints_np = self._joint_dropout(keypoints_np, self.aug_params.joint_drop_prob)

            if self.aug_params.frame_drop_prob > 0:
                keypoints_np = self._frame_dropout(keypoints_np, self.aug_params.frame_drop_prob)

        keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
        # Permute to (C, T, V) -> Channels (coords), Time (frames), Vertices (joints)
        keypoints_tensor = keypoints_tensor.permute(2, 0, 1).contiguous()
        
        # print(keypoints_tensor.shape)
        # print(optical_flows_np.shape)
        
        return keypoints_tensor, optical_flows_tensor, label

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

    # Dropout and Temporal Jitter methods remain the same as they operate on keypoints_np
    # and their logic for zeroing out or copying frames is independent of normalization scale,
    # assuming confidence=0 marks invalid/dropped keypoints.

    def _joint_dropout(self, keypoints_np_slice: np.ndarray, drop_prob: float) -> np.ndarray:
        output_data = keypoints_np_slice.copy()
        if random.random() < drop_prob:
            num_total_joints = output_data.shape[1] # self.num_joints
            # Drop a random fraction of joints up to drop_prob
            num_drop = int(num_total_joints * random.uniform(0.01, drop_prob)) # Ensure at least one if prob met
            if num_drop > 0:
                drop_indices = random.sample(range(num_total_joints), num_drop)
                output_data[:, drop_indices, :] = 0 # Set x, y, confidence to 0
        return output_data

    def _frame_dropout(self, keypoints_np_slice: np.ndarray, drop_prob: float) -> np.ndarray:
        output_data = keypoints_np_slice.copy()
        if random.random() < drop_prob:
            num_total_frames = output_data.shape[0] # self.T
            num_drop = int(num_total_frames * random.uniform(0.01, drop_prob))
            if num_drop > 0:
                drop_indices = random.sample(range(num_total_frames), num_drop)
                output_data[drop_indices, :, :] = 0 # Set x, y, confidence to 0 for entire frames
        return output_data

    def _temporal_jitter(self, keypoints_np_slice: np.ndarray, jitter_prob: float) -> np.ndarray:
        output_data = keypoints_np_slice.copy()
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

class SiameseKpOfDataset(Dataset):
    """
    Wrapper to produce pairs for joint CrossEntropy + Contrastive training.
    Returns:
      (x1, x2), (lab1, lab2), y
      lab1, lab2: labels for CE loss
      y: binary (0: same, 1: different) for ContrastiveLoss
    """
    def __init__(self, base_dataset: KpOfDataset, data_args:DataArguments):
        self.base = base_dataset
        # Build mapping label -> indices for sampling
        self.label_to_indices = {}
        with open(os.path.join(data_args.data_dir, data_args.trainsplit_dir_name, "annotation_windows_metadata.json"), 'r') as f:
            self.annotations = json.load(f)
        for idx, sample_annotation in enumerate(self.base.annotations):
            lab = int(sample_annotation['action_id'])
            self.label_to_indices.setdefault(lab, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        kp1, flow1, lab1 = self.base[index]
        lab1_int = int(lab1.item())

        # obtain another sample
        if random.random() < 0.5:
            idx2 = random.choice(self.label_to_indices[lab1_int])
            y = 0.0
        else:
            neg_label = random.choice([l for l in self.labels if l != lab1_int])
            idx2 = random.choice(self.label_to_indices[neg_label])
            y = 1.0

        kp2, flow2, lab2 = self.base[idx2]
        
        y = torch.tensor(y, dtype=torch.float32)
        
        return (kp1, kp2), (flow1, flow2), (lab1, lab2), y