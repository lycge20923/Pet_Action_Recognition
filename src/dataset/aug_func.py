import numpy as np
import cv2
import random

# for skeleton dataset
def get_center(keypoints_with_conf: np.ndarray, valid_kpt_confidence_thresh: float) -> np.ndarray:
    """
    Calculate the center (mean position) of valid keypoints based on confidence.
    keypoints_with_conf shape: (..., num_joints, 3) where last dim is [x, y, conf]
    Returns:
        center: np.ndarray of shape (2,) as [cx, cy]
    """
    points_xy = keypoints_with_conf[..., :2]
    confidences = keypoints_with_conf[..., 2]
    
    # Flatten to (N, 2) for points and (N,) for confidences
    flat_points_xy = points_xy.reshape(-1, 2)
    flat_confidences = confidences.reshape(-1)

    valid_mask = flat_confidences > valid_kpt_confidence_thresh
    valid_points = flat_points_xy[valid_mask]

    if valid_points.shape[0] > 0:
        center = np.mean(valid_points, axis=0)
    else:
        # Fallback: if any point values > 0, use (0.5,0.5), else (0,0)
        center = np.array([0.5, 0.5], dtype=np.float32) if np.any(points_xy > 0) else np.array([0.0, 0.0], dtype=np.float32)
    return center

def common_preprocess_for_augmentation(data_slice_with_conf: np.ndarray, valid_kpt_confidence_thresh: float):
    """
    Preprocess keypoints for geometric transforms:
      - Compute center
      - Flatten xy coords
      - Generate valid mask by confidence
    Args:
        data_slice_with_conf: np.ndarray of shape (T, num_joints, 3) or (num_joints, 3)
        valid_kpt_confidence_thresh: float threshold
    Returns:
        center: np.ndarray of shape (2,)
        original_xy_shape: tuple, the shape of the xy array before flattening
        flat_xy_coords: np.ndarray of shape (N, 2)
        valid_mask_flat: np.ndarray of shape (N,) boolean mask
    """
    points_xy = data_slice_with_conf[..., :2] # Shape (T, num_joints, 2) or (num_joints, 2)
    original_xy_shape = points_xy.shape
    
    center = get_center(data_slice_with_conf, valid_kpt_confidence_thresh) # Center is calculated based on confidences

    flat_xy_coords = points_xy.reshape(-1, 2) # Shape (T*num_joints, 2)
    confidences_flat = data_slice_with_conf[..., 2].reshape(-1) # Shape (T*num_joints,)
    valid_mask_flat = confidences_flat > valid_kpt_confidence_thresh
    
    return center, original_xy_shape, flat_xy_coords, valid_mask_flat

def rotate(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    angle: float = 0.0,
    valid_kpt_confidence_thresh: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Rotate keypoints and optical flow positions and vector values by the same angle.
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np
    # determine center
    if keypoints_np is not None:
        center_norm = get_center(keypoints_np, valid_kpt_confidence_thresh)
    else:
        center_norm = np.array([0.5, 0.5], dtype=np.float32)
    # rotate keypoints
    if keypoints_np is not None:
        center, orig_shape, flat_xy, valid_mask = common_preprocess_for_augmentation(
            keypoints_np, valid_kpt_confidence_thresh)
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        pts = flat_xy[valid_mask] - center
        pts_rot = pts @ R.T + center
        flat_xy[valid_mask] = pts_rot
        kps_out = keypoints_np.copy()
        kps_out[..., :2] = flat_xy.reshape(orig_shape)
    # rotate flow
    if optical_flows_np is not None:
        C, T, H, W = optical_flows_np.shape
        cx = center_norm[0] * (W - 1)
        cy = center_norm[1] * (H - 1)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        flow_tmp = np.zeros_like(optical_flows_np)
        # spatial warp
        for t in range(T):
            for c in range(C):
                flow_tmp[c, t] = cv2.warpAffine(optical_flows_np[c, t], M, (W, H), flags=cv2.INTER_LINEAR)
        # rotate vector values
        u = flow_tmp[0]
        v = flow_tmp[1]
        u_rot = cos_a * u - sin_a * v
        v_rot = sin_a * u + cos_a * v
        flow_out = np.stack([u_rot, v_rot], axis=0)
    return kps_out, flow_out

def scale(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    scale_factor: float = 1.0,
    valid_kpt_confidence_thresh: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Scale keypoints and optical flow positions and vector magnitudes.
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np
    if keypoints_np is not None:
        center_norm = get_center(keypoints_np, valid_kpt_confidence_thresh)
    else:
        center_norm = np.array([0.5, 0.5], dtype=np.float32)
    # scale keypoints
    if keypoints_np is not None:
        center, orig_shape, flat_xy, valid_mask = common_preprocess_for_augmentation(
            keypoints_np, valid_kpt_confidence_thresh)
        pts = flat_xy[valid_mask]
        pts_scaled = (pts - center) * scale_factor + center
        flat_xy[valid_mask] = pts_scaled
        kps_out = keypoints_np.copy()
        kps_out[..., :2] = flat_xy.reshape(orig_shape)
    # scale flow
    if optical_flows_np is not None:
        C, T, H, W = optical_flows_np.shape
        cx = center_norm[0] * (W - 1)
        cy = center_norm[1] * (H - 1)
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale_factor)
        flow_tmp = np.zeros_like(optical_flows_np)
        for t in range(T):
            for c in range(C):
                flow_tmp[c, t] = cv2.warpAffine(optical_flows_np[c, t], M, (W, H), flags=cv2.INTER_LINEAR)
        # scale vector values
        flow_out = flow_tmp * scale_factor
    return kps_out, flow_out

def translate(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    tx: float = 0.0,
    ty: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Translate keypoints and optical flow spatially. Vectors unchanged.
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np
    if keypoints_np is not None:
        kps_out = keypoints_np.copy()
        kps_out[..., 0] += tx
        kps_out[..., 1] += ty
    if optical_flows_np is not None:
        C, T, H, W = optical_flows_np.shape
        dx = tx * (W - 1)
        dy = ty * (H - 1)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        flow_tmp = np.zeros_like(optical_flows_np)
        for t in range(T):
            for c in range(C):
                flow_tmp[c, t] = cv2.warpAffine(optical_flows_np[c, t], M, (W, H), flags=cv2.INTER_LINEAR)
        flow_out = flow_tmp
    return kps_out, flow_out

def shear(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    sx: float = 0.0,
    sy: float = 0.0,
    valid_kpt_confidence_thresh: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Shear keypoints and optical flow positions and vector values.
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np
    if keypoints_np is not None:
        center_norm = get_center(keypoints_np, valid_kpt_confidence_thresh)
        center, orig_shape, flat_xy, valid_mask = common_preprocess_for_augmentation(
            keypoints_np, valid_kpt_confidence_thresh)
        S = np.array([[1, sx], [sy, 1]], dtype=np.float32)
        pts = flat_xy[valid_mask] - center
        pts_sheared = pts @ S.T + center
        flat_xy[valid_mask] = pts_sheared
        kps_out = keypoints_np.copy()
        kps_out[..., :2] = flat_xy.reshape(orig_shape)
    else:
        center_norm = np.array([0.5,0.5], dtype=np.float32)
    if optical_flows_np is not None:
        C, T, H, W = optical_flows_np.shape
        cx = center_norm[0] * (W - 1)
        cy = center_norm[1] * (H - 1)
        M = np.array([[1, sx, -sx*cy], [sy, 1, -sy*cx]], dtype=np.float32)
        flow_tmp = np.zeros_like(optical_flows_np)
        for t in range(T):
            for c in range(C):
                flow_tmp[c, t] = cv2.warpAffine(optical_flows_np[c, t], M, (W, H), flags=cv2.INTER_LINEAR)
        # shear vector values
        u = flow_tmp[0]; v = flow_tmp[1]
        uv = np.stack([u, v], axis=-1)
        # apply shear matrix to vectors
        uv_sheared = uv @ S.T
        flow_out = np.stack([uv_sheared[...,0], uv_sheared[...,1]], axis=0)
    return kps_out, flow_out

def hflip(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Horizontally flip keypoints and/or optical flow.

    Args:
        keypoints_np: (T, V, 3) array with normalized coords in [0,1], or None
        optical_flows_np: (C, T, H, W) array, or None
    Returns:
        (flipped_keypoints_np, flipped_optical_flows_np)
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np

    # Flip keypoints
    if keypoints_np is not None:
        kps_out = keypoints_np.copy()
        # Flip x-coordinate in normalized space
        kps_out[..., 0] = 1.0 - kps_out[..., 0]
    
    # Flip optical flow
    if optical_flows_np is not None:
        # Flip width axis
        flow_out = optical_flows_np[..., ::-1].copy()
        # Invert horizontal flow component (channel 0)
        flow_out[0] = -flow_out[0]

    return kps_out, flow_out

def frame_drop(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    drop_prob: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Drop (zero out) a fraction of frames in keypoints and/or optical flow.
    The random decision to apply or not should be made outside this function.
    Args:
        keypoints_np: (T, V, 3) array or None
        optical_flows_np: (C, T, H, W) array or None
        drop_prob: fraction of frames to drop (0.0 to 1.0)
    Returns:
        (dropped_keypoints_np, dropped_optical_flows_np)
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np
    
    # Determine total frames T
    if keypoints_np is not None:
        T = keypoints_np.shape[0]
    elif optical_flows_np is not None:
        T = optical_flows_np.shape[1]
    else:
        return kps_out, flow_out

    # Number of frames to drop: between 1% and drop_prob of T
    num_drop = int(T * random.uniform(0.01, drop_prob))
    if num_drop > 0:
        drop_idxs = random.sample(range(T), num_drop)
        # Apply to keypoints
        if keypoints_np is not None:
            kps_out = keypoints_np.copy()
            kps_out[drop_idxs, ...] = 0
        # Apply to flow
        if optical_flows_np is not None:
            flow_out = optical_flows_np.copy()
            flow_out[:, drop_idxs, :, :] = 0
    return kps_out, flow_out

def temporal_jitter(
    keypoints_np: np.ndarray | None = None,
    optical_flows_np: np.ndarray | None = None,
    jitter_prob: float = 0.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Randomly jitter a fraction of frames in keypoints and/or optical flow by copying neighbor frames.
    The random decision to apply or not should be made outside this function.
    Args:
        keypoints_np: (T, V, 3) array or None
        optical_flows_np: (C, T, H, W) array or None
        jitter_prob: maximum fraction of frames to jitter (0.0 to 1.0)
    Returns:
        (jittered_keypoints_np, jittered_optical_flows_np)
    """
    kps_out = keypoints_np
    flow_out = optical_flows_np

    import random
    # Determine total frames T
    if keypoints_np is not None:
        T = keypoints_np.shape[0]
    elif optical_flows_np is not None:
        T = optical_flows_np.shape[1]
    else:
        return kps_out, flow_out

    # Need at least 3 frames to jitter
    if jitter_prob <= 0 or T <= 2:
        return kps_out, flow_out

    # Number of frames to jitter: between 1% and jitter_prob of T
    num_jit = int(T * random.uniform(0.01, jitter_prob))
    if num_jit > 0:
        possible_idxs = list(range(1, T - 1))
        num_jit = min(num_jit, len(possible_idxs))
        jit_idxs = random.sample(possible_idxs, num_jit)

        # Prepare outputs
        if keypoints_np is not None:
            kps_out = keypoints_np.copy()
        if optical_flows_np is not None:
            flow_out = optical_flows_np.copy()

        for idx in jit_idxs:
            if random.random() < 0.5:
                # copy previous frame
                if kps_out is not None:
                    kps_out[idx, ...] = kps_out[idx - 1, ...]
                if flow_out is not None:
                    flow_out[:, idx, ...] = flow_out[:, idx - 1, ...]
            else:
                # copy next frame
                if kps_out is not None:
                    kps_out[idx, ...] = kps_out[idx + 1, ...]
                if flow_out is not None:
                    flow_out[:, idx, ...] = flow_out[:, idx + 1, ...]
    return kps_out, flow_out

def add_flow_noise(
    optical_flows_np: np.ndarray,
    noise_std: float
) -> np.ndarray:
    """
    Add Gaussian noise to optical flow values.
    Args:
        optical_flows_np: (C, T, H, W) array
        noise_std: standard deviation of noise
    Returns:
        noisy optical flow of same shape
    """
    noise = np.random.normal(0, noise_std, size=optical_flows_np.shape).astype(optical_flows_np.dtype)
    return optical_flows_np + noise

def random_flow_occlusion(
    optical_flows_np: np.ndarray,
    occl_size_ratio: float
) -> np.ndarray:
    """
    Randomly occlude a block in optical flow to zero.
    Args:
        optical_flows_np: (C, T, H, W) array
        occl_size_ratio: fraction of height/width for occlusion block
    Returns:
        occluded optical flow
    """
    C, T, H, W = optical_flows_np.shape
    h = int(H * occl_size_ratio)
    w = int(W * occl_size_ratio)
    y0 = np.random.randint(0, H - h + 1)
    x0 = np.random.randint(0, W - w + 1)
    out = optical_flows_np.copy()
    out[:, :, y0:y0+h, x0:x0+w] = 0
    return out

def gaussian_blur_flow(
    optical_flows_np: np.ndarray,
    ksize: int,
    sigma: float
) -> np.ndarray:
    """
    Apply Gaussian blur to each channel and frame of optical flow.
    Args:
        optical_flows_np: (C, T, H, W) array
        ksize: kernel size (must be odd)
        sigma: Gaussian sigma
    Returns:
        blurred optical flow
    """
    C, T, H, W = optical_flows_np.shape
    out = optical_flows_np.copy()
    for t in range(T):
        for c in range(C):
            out[c, t] = cv2.GaussianBlur(out[c, t], (ksize, ksize), sigma)
    return out
