import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from ..utils.cli_args import AugmentationArguments

class Augmentation(nn.Module):
    """
    GPU-friendly augmentation wrapper that mirrors the original aug_func logic.

    Inputs (expected shapes):
      - keypoints: (B, 3, T, V) with channels (x, y, conf)  [normalized to ~[0,1]]
      - flows    : (B, 2, T, H, W) with channels (u, v)

    Notes:
      - Applies only when self.training is True; eval() → no-op.
      - Pivot for geometric ops = skeleton center computed from conf>0 joints
        (fallback to (0.5, 0.5) per frame if no valid joints).
      - All ops will be implemented with torch (grid_sample/conv2d/…).
    """

    def __init__(
        self,
        # ---- global ----
        align_corners: bool = False,
        resample_mode: str = "bilinear",
        pad_mode: str = "zeros",
        clamp_xy: bool = True,
        per_frame: bool = False,   # per-frame random params (vs per-clip)
        augment_params: AugmentationArguments = None,
    ):
        super().__init__()
        # global
        if augment_params is None:
            self.enable = None
            return
        self.enable = bool(augment_params.augment) 
        self.align_corners = bool(align_corners)
        self.resample_mode = str(resample_mode)
        self.pad_mode = str(pad_mode)
        self.clamp_xy = bool(clamp_xy)
        self.per_frame = bool(per_frame)

        # rotate
        self.rotate_max_deg = float(augment_params.rot_max)

        # scale
        self.scale_min = float(augment_params.scale_min)
        self.scale_max = float(augment_params.scale_max)

        # translate
        self.translate_x_max = float(augment_params.trans_max)
        self.translate_y_max = float(augment_params.trans_max)
        
        # horizontal flip
        self.flow_hflip_prob = float(augment_params.flow_hflip_prob)

        # shear
        self.shear_x_max = float(augment_params.shear_max)
        self.shear_y_max = float(augment_params.shear_max)

        # temporal
        self.p_temporal_jitter = float(augment_params.temporal_jitter_prob)
        self.jitter_max_offset = int(1)
        self.frame_drop_ratio = float(augment_params.frame_drop_prob)
        
        # flow noise / occlusion / blur
        self.flow_noise_std = float(augment_params.flow_noise_std)
        self.occ_min = float(0.1)
        self.occ_max = float(augment_params.flow_occl_ratio)
        self.blur_ksize = int(augment_params.flow_blur_ksize)
        self.blur_sigma = float(augment_params.flow_blur_sigma)
        
        self.valid_kpt_confidence_thresh = float(augment_params.valid_kpt_confidence_thresh)
    
    def _pick_device(self, keypoints, flows, rgb):
        if keypoints is not None:
            return keypoints.device
        if flows is not None:
            return flows.device
        if rgb is not None:
            return rgb.device
        # 都是 None（理論上 forward 已避免），退回 CPU
        return torch.device("cpu")
    
    @torch.no_grad()
    def get_center(self, keypoints):
        """
        Clip-level center over all (T,V) points with conf > threshold.
        Returns:
            cx_clip, cy_clip: (B, 1, 1)
        """
        assert keypoints.dim() == 4 and keypoints.size(1) == 3, "keypoints must be (B,3,T,V)"
        x = keypoints[:, 0]  # (B,T,V)
        y = keypoints[:, 1]
        conf = keypoints[:, 2]
        device = x.device
        dtype = x.dtype

        valid = conf > self.valid_kpt_confidence_thresh  # (B,T,V)
        valid_sum = valid.float().sum(dim=(1, 2), keepdim=True)      # (B,1,1)
        none_valid = (valid_sum == 0)

        # 聚合（clip-level）
        x_sum = torch.where(valid, x, torch.zeros_like(x)).sum(dim=(1, 2), keepdim=True)
        y_sum = torch.where(valid, y, torch.zeros_like(y)).sum(dim=(1, 2), keepdim=True)
        denom = valid_sum.clamp(min=1.0)

        cx = x_sum / denom
        cy = y_sum / denom

        # aug_func.py 的 fallback：若沒有任何 valid 點，
        # 若座標中有任一值 > 0 → 0.5，否則 → 0.0
        has_any_pos = ((x > 0) | (y > 0)).any(dim=(1, 2), keepdim=True)
        fallback_val = torch.where(has_any_pos, torch.tensor(0.5, device=device, dtype=dtype),
                                torch.tensor(0.0, device=device, dtype=dtype))
        cx = torch.where(none_valid, fallback_val, cx)
        cy = torch.where(none_valid, fallback_val, cy)
        return cx, cy

    @torch.no_grad()
    def common_preprocess_for_augmentation(
        self,
        keypoints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert keypoints.dim() == 4 and keypoints.size(1) == 3, "keypoints must be (B,3,T,V)"
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        conf = keypoints[:, 2]

        # 與原版一致：用門檻形成 valid mask
        valid = conf > self.valid_kpt_confidence_thresh

        cx, cy = self.get_center(keypoints)
        # expand to per-frame for downstream shapes
        B, _, T, _ = keypoints.shape
        cx = cx.expand(B, T, 1)
        cy = cy.expand(B, T, 1)

        return {
            "kpts": keypoints,
            "x": x, "y": y, "conf": conf,
            "valid": valid,   # 後續幾何/時間增強可共用這個 mask
            "cx": cx, "cy": cy,
        }
    
    # ---------- public API ----------
    def forward(
        self,
        keypoints: torch.Tensor,  # (B,3,T,V)
        flows: torch.Tensor,      # (B,2,T,H,W)
        rgbs: torch.Tensor,       # (B,3,T,H,W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations in the same ORDER as the original aug_func.
        (暫時只放呼叫順序；各函式會在下一步逐一實作)
        """
        if (not self.enable) or (not self.training):
            return keypoints, flows, rgbs

        k, f, r = keypoints, flows, rgbs

        # 幾何（以 skeleton center 為 pivot）
        k, f, r = self.rotate(k, f, r)
        k, f, r = self.scale(k, f, r)
        k, f, r = self.translate(k, f, r)
        k, f, r = self.shear(k, f, r)
        k, f, r = self.hflip(k, f, r)

        # 時間
        k, f, r = self.frame_drop(k, f, r)
        k, f, r = self.temporal_jitter(k, f, r)

        # 流場數值類
        f = self.gaussian_blur_flow(f)
        f = self.add_flow_noise(f)
        f = self.random_flow_occlusion(f)

        return k, f, r

    # ---------- geometric ops (skeleton-center pivot) ----------
    def rotate(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor], rgb: Optional[torch.Tensor]):
        """
        - None-safe：
            * 兩者皆 None → 原樣返回
            * 只有 flows → 以影像中心 (0.5,0.5) 為 pivot，對 flow 影像 + flow 向量旋轉
            * 只有 keypoints → 以 skeleton center 為 pivot，僅旋轉 keypoints
            * 兩者皆在 → 如常
        """
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.5:
            return keypoints, flows, rgb
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # === 準備 B,T 與 pivot ===
        if keypoints is not None:
            assert keypoints.dim() == 4 and keypoints.size(1) == 3, "keypoints must be (B,3,T,V)"
            B, _, T, V = keypoints.shape
            cx, cy = self.get_center(keypoints)            # (B,1,1)
            cx = cx.expand(B, T, 1).contiguous()                # (B,T,1)
            cy = cy.expand(B, T, 1).contiguous()                # (B,T,1)
        else:
            if flows is not None:
                B, _, T, _, _ = flows.shape
            else:
                B, _, T, _, _ = rgb.shape
            cx = cy = None
            

        # === 抽角度（per-frame/per-clip）===
        if self.per_frame:
            angles_deg = (torch.rand(B, T, device=device) * 2 * self.rotate_max_deg) - self.rotate_max_deg
        else:
            a = (torch.rand(B, 1, device=device) * 2 * self.rotate_max_deg) - self.rotate_max_deg
            angles_deg = a.expand(B, T).contiguous()
        angles_rad = angles_deg * (torch.pi / 180.0)
        cos_t = torch.cos(angles_rad)  # (B,T)
        sin_t = torch.sin(angles_rad)

        # === flows 分支 ===
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2, "flows must be (B,2,T,H,W)"
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, f"(B,T) mismatch between kpts({B},{T}) and flows({Bf},{Tf})"
            dtype_img = flows.dtype

            # pivot：若無 kpts，採用影像中心 (0.5,0.5)
            if cx is None:
                cx = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)
                cy = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)

            # 轉為 [-1,1] 座標，組仿射（保持 pivot 不動）
            cx_n = (cx * 2) - 1
            cy_n = (cy * 2) - 1
            cos_t3, sin_t3 = cos_t[..., None], sin_t[..., None]
            tx = cx_n * (1 - cos_t3) + cy_n * (sin_t3)
            ty = cy_n * (1 - cos_t3) - cx_n * (sin_t3)

            theta = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta[..., 0, 0] =  cos_t
            theta[..., 0, 1] = -sin_t
            theta[..., 1, 0] =  sin_t
            theta[..., 1, 1] =  cos_t
            theta[..., 0, 2] =  tx.squeeze(-1)
            theta[..., 1, 2] =  ty.squeeze(-1)
            theta = theta.reshape(B*T, 2, 3)

            flows_bt = flows.permute(0,2,1,3,4).contiguous().reshape(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img_rot = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # 旋轉 flow 向量值
            u = flows_img_rot[:, 0:1]
            v = flows_img_rot[:, 1:2]
            u_r =  u * cos_t.reshape(B*T,1,1,1) - v * sin_t.reshape(B*T,1,1,1)
            v_r =  u * sin_t.reshape(B*T,1,1,1) + v * cos_t.reshape(B*T,1,1,1)
            flows = torch.cat([u_r, v_r], dim=1).reshape(B, T, 2, H, W).permute(0,2,1,3,4).contiguous()

        # === keypoints 分支 ===
        if keypoints is not None:
            dtype_kpt = keypoints.dtype
            x = keypoints[:, 0]  # (B,T,V)
            y = keypoints[:, 1]
            conf = keypoints[:, 2]
            cx_k = cx.expand(B, T, 1)
            cy_k = cy.expand(B, T, 1)
            cos_t3, sin_t3 = cos_t[..., None], sin_t[..., None]
            x0, y0 = x - cx_k, y - cy_k
            x_rot = x0 * cos_t3 - y0 * sin_t3 + cx_k
            y_rot = x0 * sin_t3 + y0 * cos_t3 + cy_k
            if self.clamp_xy:
                x_rot = x_rot.clamp(0,1)
                y_rot = y_rot.clamp(0,1)
            valid = conf > self.valid_kpt_confidence_thresh
            out = keypoints.clone()
            out[:, 0] = torch.where(valid, x_rot, x)
            out[:, 1] = torch.where(valid, y_rot, y)
            out[:, 2] = conf
            keypoints = out
        
        # === rgb 分支 ===
        if rgb is not None:
            assert rgb.dim() == 5, "rgb should be (B,C,T,H,W)"
            Br, Cr, Tr, Hr, Wr = rgb.shape
            assert B == Br and T == Tr, f"(B,T) mismatch between reference({B},{T}) and rgb({Br},{Tr})"
            
            if cx is None:
                cx = torch.full((B, 1, 1), 0.5, device=device, dtype=rgb.dtype)
                cy = torch.full((B, 1, 1), 0.5, device=device, dtype=rgb.dtype)

            # 用與 flows/Keypoints 相同的 pivot 與角度建 theta_r
            cx_n_r = (cx * 2) - 1
            cy_n_r = (cy * 2) - 1
            cos_t3, sin_t3 = cos_t[..., None], sin_t[..., None]
            t_x = cx_n_r * (1 - cos_t3) + cy_n_r * (sin_t3)
            t_y = cy_n_r * (1 - cos_t3) - cx_n_r * (sin_t3)

            theta_r = torch.zeros(B, T, 2, 3, device=device, dtype=rgb.dtype)
            theta_r[..., 0, 0] =  cos_t
            theta_r[..., 0, 1] = -sin_t
            theta_r[..., 1, 0] =  sin_t
            theta_r[..., 1, 1] =  cos_t
            theta_r[..., 0, 2] =  t_x.squeeze(-1)
            theta_r[..., 1, 2] =  t_y.squeeze(-1)
            theta_r = theta_r.reshape(B*T, 2, 3)

            rgb_bt = rgb.permute(0,2,1,3,4).contiguous().reshape(B*T, Cr, Hr, Wr)
            grid_r = F.affine_grid(theta_r, size=(B*T, Cr, Hr, Wr), align_corners=self.align_corners)
            rgb_img_rot = F.grid_sample(
                rgb_bt, grid_r,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            rgb = rgb_img_rot.reshape(B, T, Cr, Hr, Wr).permute(0,2,1,3,4).contiguous()

        return keypoints, flows, rgb
    

    def scale(self, keypoints, flows, rgb):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.5:
            return keypoints, flows, rgb
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # 取 B,T
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        elif flows is not None:
            B, _, T, _, _ = flows.shape
        else:
            B, _, T, _, _ = rgb.shape

        # 取每幀/每段 s ∈ [scale_min, scale_max]
        s_min = float(self.scale_min)
        s_max = float(self.scale_max)
        if self.per_frame:
            s = torch.empty(B, T, device=device).uniform_(s_min, s_max)  # (B,T)
        else:
            s = torch.empty(B, 1, device=device).uniform_(s_min, s_max).expand(B, T)
        
        # pivot：有 kps 用骨架中心；否則用影像中心 0.5,0.5
        if keypoints is not None:
            cx, cy = self.get_center(keypoints)
            cx = cx.expand(B, T, 1).contiguous()
            cy = cy.expand(B, T, 1).contiguous()
        else:
            dtype_img = flows.dtype if flows is not None else rgb.dtype
            cx = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)
            cy = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)

        # --- flows 分支：影像 warp + 向量值域縮放 ---
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, "scale: (B,T) mismatch between keypoints and flows"
            dtype_img = flows.dtype

            # 到 [-1,1] 座標並保持 pivot：t = (I - S) p
            cx_n = (cx * 2) - 1
            cy_n = (cy * 2) - 1
            s3 = s[..., None]  # (B,T,1)
            tx = cx_n * (1.0 - s3)
            ty = cy_n * (1.0 - s3)

            theta = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta[..., 0, 0] = s
            theta[..., 1, 1] = s
            theta[..., 0, 2] = tx.squeeze(-1)
            theta[..., 1, 2] = ty.squeeze(-1)
            theta = theta.reshape(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # flow 向量值域同步縮放（u,v 乘 s）
            s_flat = s.reshape(B*T, 1, 1, 1)
            flows = (flows_img * s_flat).reshape(B, T, 2, H, W).permute(0,2,1,3,4).contiguous()

        # --- keypoints 分支：以 pivot 為中心縮放 ---
        if keypoints is not None:
            dtype_kpt = keypoints.dtype
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            conf = keypoints[:, 2]
            cx_k = cx.expand(B, T, 1)
            cy_k = cy.expand(B, T, 1)
            s3 = s[..., None]
            x_s = (x - cx_k) * s3 + cx_k
            y_s = (y - cy_k) * s3 + cy_k
            if self.clamp_xy:
                x_s = x_s.clamp(0,1)
                y_s = y_s.clamp(0,1)
            valid = conf > self.valid_kpt_confidence_thresh
            out = keypoints.clone()
            out[:, 0] = torch.where(valid, x_s, x)
            out[:, 1] = torch.where(valid, y_s, y)
            out[:, 2] = conf
            keypoints = out
        
        # --- RGB 分支：影像 warp（不縮放像素值，只做幾何） ---
        if rgb is not None:
            assert rgb.dim() == 5, "rgb should be (B,C,T,H,W)"
            Br, Cr, Tr, Hr, Wr = rgb.shape
            assert B == Br and T == Tr, "scale: (B,T) mismatch between reference and rgb"

            # 以同一 pivot 與 s 建立對應 RGB 的 theta/grid
            cx_n_r = (cx * 2) - 1
            cy_n_r = (cy * 2) - 1
            s3_r = s[..., None]
            tx_r = cx_n_r * (1.0 - s3_r)
            ty_r = cy_n_r * (1.0 - s3_r)

            theta_r = torch.zeros(B, T, 2, 3, device=device, dtype=rgb.dtype)
            theta_r[..., 0, 0] = s
            theta_r[..., 1, 1] = s
            theta_r[..., 0, 2] = tx_r.squeeze(-1)
            theta_r[..., 1, 2] = ty_r.squeeze(-1)
            theta_r = theta_r.reshape(B*T, 2, 3)

            rgb_bt = rgb.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, Cr, Hr, Wr)
            grid_r = F.affine_grid(theta_r, size=(B*T, Cr, Hr, Wr), align_corners=self.align_corners)
            rgb_img = F.grid_sample(
                rgb_bt, grid_r,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            rgb = rgb_img.reshape(B, T, Cr, Hr, Wr).permute(0, 2, 1, 3, 4).contiguous()

        return keypoints, flows, rgb


    def translate(self, keypoints, flows, rgb):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.5:
            return keypoints, flows, rgb
        
        # gating removed to mirror aug_func.py (external control)
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # 取 B,T
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        elif flows is not None:
            B, _, T, _, _ = flows.shape
        else:
            B, _, T, _, _ = rgb.shape

        # 取 (dx,dy) ∈ [-trans_max, +trans_max]
        tx_max = float(self.translate_x_max)
        ty_max = float(self.translate_y_max)
        if self.per_frame:
            dx = (torch.rand(B, T, device=device) * 2 * tx_max) - tx_max  # (B,T)
            dy = (torch.rand(B, T, device=device) * 2 * ty_max) - ty_max
        else:
            dx = ((torch.rand(B, 1, device=device) * 2 * tx_max) - tx_max).expand(B, T)
            dy = ((torch.rand(B, 1, device=device) * 2 * ty_max) - ty_max).expand(B, T)

        # --- flows 分支：影像平移；向量值不變 ---
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, "translate: (B,T) mismatch between keypoints and flows"
            dtype_img = flows.dtype

            # grid_sample 的平移量以 [-1,1] 記，所以把 [0..1] 的 dx,dy 轉 2*dx, 2*dy
            tnx = (dx * 2.0).unsqueeze(-1)    # (B,T,1)
            tny = (dy * 2.0).unsqueeze(-1)

            theta = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta[..., 0, 0] = 1.0
            theta[..., 1, 1] = 1.0
            theta[..., 0, 2] = tnx.squeeze(-1)
            theta[..., 1, 2] = tny.squeeze(-1)
            theta = theta.reshape(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            flows = flows_img.reshape(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()
            # 注意：flow 向量值不變（不需改 u,v）

        # --- keypoints 分支：x += dx, y += dy ---
        if keypoints is not None:
            prep = self.common_preprocess_for_augmentation(keypoints)
            x, y, conf = prep["x"], prep["y"], prep["conf"]
            dx3 = dx[..., None]   # (B,T,1)
            dy3 = dy[..., None]
            x_t = x + dx3
            y_t = y + dy3
            if self.clamp_xy:
                x_t = x_t.clamp(0, 1)
                y_t = y_t.clamp(0, 1)
            valid = conf > self.valid_kpt_confidence_thresh
            out = keypoints.clone()
            out[:, 0] = torch.where(valid, x_t, x)
            out[:, 1] = torch.where(valid, y_t, y)
            out[:, 2] = conf
            keypoints = out
        
        if rgb is not None:
            assert rgb.dim() == 5, "rgb should be (B,C,T,H,W)"
            Br, Cr, Tr, Hr, Wr = rgb.shape
            assert B == Br and T == Tr, "translate: (B,T) mismatch between reference and rgb"
            dtype_img = rgb.dtype

            # grid_sample 的平移量以 [-1,1] 記，所以把 [0..1] 的 dx,dy 轉 2*dx, 2*dy
            tnx = (dx * 2.0).unsqueeze(-1)    # (B,T,1)
            tny = (dy * 2.0).unsqueeze(-1)

            theta_r = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta_r[..., 0, 0] = 1.0
            theta_r[..., 1, 1] = 1.0
            theta_r[..., 0, 2] = tnx.squeeze(-1)
            theta_r[..., 1, 2] = tny.squeeze(-1)
            theta_r = theta_r.reshape(B*T, 2, 3)

            rgb_bt = rgb.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, Cr, Hr, Wr)  # (B*T,Cr,H,W)
            grid_r = F.affine_grid(theta_r, size=(B*T, Cr, Hr, Wr), align_corners=self.align_corners)
            rgb_img = F.grid_sample(
                rgb_bt, grid_r,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            rgb = rgb_img.reshape(B, T, Cr, Hr, Wr).permute(0, 2, 1, 3, 4).contiguous()

        return keypoints, flows, rgb

    def shear(self, keypoints, flows, rgb):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.2:
            return keypoints, flows, rgb
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # 取 T 與 B
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        elif flows is not None:
            B, _, T, _, _ = flows.shape
        else:
            B, _, T, _, _ = rgb.shape

        shx_max = float(self.shear_x_max)
        shy_max = float(self.shear_y_max)
        if self.per_frame:
            shx = (torch.rand(B, T, device=device) * 2 * shx_max) - shx_max
            shy = (torch.rand(B, T, device=device) * 2 * shy_max) - shy_max
        else:
            shx = ((torch.rand(B, 1, device=device) * 2 * shx_max) - shx_max).expand(B, T)
            shy = ((torch.rand(B, 1, device=device) * 2 * shy_max) - shy_max).expand(B, T)

        # pivot：有 kps 用骨架中心；否則用影像中心 0.5,0.5
        if keypoints is not None:
            cx, cy = self.get_center(keypoints)
            cx = cx.expand(B, T, 1).contiguous()
            cy = cy.expand(B, T, 1).contiguous()
        else:
            dtype_img = flows.dtype if flows is not None else rgb.dtype
            cx = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)
            cy = torch.full((B, 1, 1), 0.5, device=device, dtype=dtype_img)

        # --- flows 分支：影像 warp（保持 pivot），並對 (u,v) 施加同一剪切 ---
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, "shear: (B,T) mismatch between keypoints and flows"
            dtype_img = flows.dtype
            # 轉到 [-1,1]，A = [[1, shx],[shy, 1]]；t = (I - A)p = [-shx*cy, -shy*cx]
            cx_n = (cx * 2) - 1
            cy_n = (cy * 2) - 1
            shx3 = shx[..., None]
            shy3 = shy[..., None]
            tx = -shx3 * cy_n
            ty = -shy3 * cx_n

            theta = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta[..., 0, 0] = 1.0
            theta[..., 0, 1] = shx
            theta[..., 1, 0] = shy
            theta[..., 1, 1] = 1.0
            theta[..., 0, 2] = tx.squeeze(-1)
            theta[..., 1, 2] = ty.squeeze(-1)
            theta = theta.reshape(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # flow 向量剪切（與影像同一 A）
            u = flows_img[:, 0:1]
            v = flows_img[:, 1:2]
            u_s = u + shx.reshape(B*T,1,1,1) * v
            v_s = v + shy.reshape(B*T,1,1,1) * u
            flows = torch.cat([u_s, v_s], dim=1).reshape(B, T, 2, H, W).permute(0,2,1,3,4).contiguous()

        # --- keypoints 分支 ---
        if keypoints is not None:
            dtype_kpt = keypoints.dtype
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            conf = keypoints[:, 2]
            cx_k = cx.expand(B, T, 1)
            cy_k = cy.expand(B, T, 1)
            shx3 = shx[..., None]
            shy3 = shy[..., None]
            x_s = (x - cx_k) + shx3 * (y - cy_k) + cx_k
            y_s = (y - cy_k) + shy3 * (x - cx_k) + cy_k
            if self.clamp_xy:
                x_s = x_s.clamp(0,1)
                y_s = y_s.clamp(0,1)
            valid = conf > self.valid_kpt_confidence_thresh
            out = keypoints.clone()
            out[:, 0] = torch.where(valid, x_s, x)
            out[:, 1] = torch.where(valid, y_s, y)
            out[:, 2] = conf
            keypoints = out
        
        if rgb is not None:
            assert rgb.dim() == 5, "rgb should be (B,C,T,H,W)"
            Br, Cr, Tr, Hr, Wr = rgb.shape
            assert B == Br and T == Tr, "shear: (B,T) mismatch between reference and rgb"

            # 若前面沒由 kpts/flows 設 pivot，這裡用影像中心
            if cx is None:
                cx = torch.full((B, 1, 1), 0.5, device=device, dtype=rgb.dtype)
                cy = torch.full((B, 1, 1), 0.5, device=device, dtype=rgb.dtype)

            # 轉到 [-1,1]，A = [[1, shx],[shy, 1]]；t = (I - A)p = [-shx*cy, -shy*cx]
            cx_n = (cx * 2) - 1
            cy_n = (cy * 2) - 1
            shx3 = shx[..., None]
            shy3 = shy[..., None]
            tx = -shx3 * cy_n
            ty = -shy3 * cx_n

            theta_r = torch.zeros(B, T, 2, 3, device=device, dtype=rgb.dtype)
            theta_r[..., 0, 0] = 1.0
            theta_r[..., 0, 1] = shx
            theta_r[..., 1, 0] = shy
            theta_r[..., 1, 1] = 1.0
            theta_r[..., 0, 2] = tx.squeeze(-1)
            theta_r[..., 1, 2] = ty.squeeze(-1)
            theta_r = theta_r.reshape(B*T, 2, 3)

            rgb_bt = rgb.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, Cr, Hr, Wr)
            grid_r = F.affine_grid(theta_r, size=(B*T, Cr, Hr, Wr), align_corners=self.align_corners)
            rgb_img = F.grid_sample(
                rgb_bt, grid_r,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            rgb = rgb_img.reshape(B, T, Cr, Hr, Wr).permute(0, 2, 1, 3, 4).contiguous()

        return keypoints, flows, rgb

    def hflip(self, keypoints, flows, rgb):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.5:
            return keypoints, flows, rgb
        
        # gating removed to mirror aug_func.py (external control)
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # flows：影像左右翻；u 取負
        if flows is not None:
            out = flows.flip(dims=[4]).clone()  # flip width
            out[:, 0] = -out[:, 0]              # negate u
            flows = out

        # keypoints：x -> 1 - x
        if keypoints is not None:
            out = keypoints.clone()
            out[:, 0] = 1.0 - out[:, 0]
            if self.clamp_xy:
                out[:, 0] = out[:, 0].clamp(0, 1)
            keypoints = out
        
        # rgb
        if rgb is not None:
            rgb = torch.flip(rgb, dims=[-1])

        return keypoints, flows, rgb

    # ---------- temporal ops ----------
    def temporal_jitter(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor], rgb: Optional[torch.Tensor]):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.3:
            return keypoints, flows, rgb
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # Take T and B
        if keypoints is not None:
            B, _, T, V = keypoints.shape
        elif flows is not None:
            B, _, T, H, W = flows.shape
        else:
            B, _, T, H, W = rgb.shape
            
        if T <= 2 or self.p_temporal_jitter <= 0:
            return keypoints, flows, rgb

        base = torch.arange(T, device=device, dtype=torch.long)
        final_idx_all = []

        for _ in range(B):
            # 每個樣本各自抽 p ~ Uniform(0.01, p_temporal_jitter)，num_jit = int(T*p)，允許為 0
            low = 0.01
            high = float(self.p_temporal_jitter)
            if high < low:
                high = low
            p = float(torch.empty(()).uniform_(low, high).item())
            num_jit = int(T * p)

            if num_jit <= 0:
                final_idx_all.append(base)
                continue

            # 僅從中間幀 [1..T-2] 選；每個被選索引獨立決定 ±1
            middle = torch.arange(1, T - 1, device=device)
            if middle.numel() == 0:
                final_idx_all.append(base)
                continue

            k = min(num_jit, middle.numel())
            idx = middle[torch.randperm(middle.numel(), device=device)[:k]]
            dirs = torch.where(
                torch.rand(k, device=device) > 0.5,
                torch.ones(k, dtype=torch.long, device=device),
                -torch.ones(k, dtype=torch.long, device=device)
            )
            new_idx = base.clone()
            new_idx[idx] = (idx + dirs).clamp(0, T - 1)
            final_idx_all.append(new_idx)

        final_idx = torch.stack(final_idx_all, dim=0)  # (B,T)

        if keypoints is not None:
            _, Ck, _, Vk = keypoints.shape
            k_idx = final_idx[:, None, :, None].expand(B, Ck, T, Vk)
            keypoints = torch.gather(keypoints, dim=2, index=k_idx).contiguous()
        if flows is not None:
            _, Cf, _, H, W = flows.shape
            f_idx = final_idx[:, None, :, None, None].expand(B, Cf, T, H, W)
            flows = torch.gather(flows, dim=2, index=f_idx).contiguous()
        if rgb is not None:
            _, Cr, _, Hr, Wr = rgb.shape
            r_idx = final_idx[:, None, :, None, None].expand(B, Cr, T, Hr, Wr)
            rgb = torch.gather(rgb, dim=2, index=r_idx).contiguous()

        return keypoints, flows, rgb

    def frame_drop(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor], rgb:Optional[torch.Tensor]):
        device = self._pick_device(keypoints, flows, rgb)
        if torch.rand((), device=device) >= 0.3:
            return keypoints, flows, rgb
        if keypoints is None and flows is None and rgb is None:
            return None, None, None

        # 取 T 與 B
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        elif flows is not None:
            B, _, T, _, _ = flows.shape
        else:
            B, _, T, _, _ = rgb.shape
        
        if T <= 1 or self.frame_drop_ratio <= 0:
            return keypoints, flows, rgb

        # 每個樣本各自抽 p ~ Uniform(0.01, frame_drop_ratio)，num_drop = int(T*p)，允許為 0
        low = 0.01
        high = float(self.frame_drop_ratio)
        if high < low:
            high = low
        num_drop_per_b = [
            int((T * float(torch.empty(()).uniform_(low, high).item())))
            for _ in range(B)
        ]

        if keypoints is not None:
            keypoints = keypoints.clone()
        if flows is not None:
            flows = flows.clone()
        if rgb is not None:
            rgb = rgb.clone()

        for b in range(B):
            k = num_drop_per_b[b]
            if k <= 0:
                continue
            idx = torch.randperm(T, device=device)[:k]
            if keypoints is not None:
                keypoints[b, :, idx, :] = 0
            if flows is not None:
                flows[b, :, idx, :, :] = 0
            if rgb is not None:
                rgb[b, :, idx, :, :] = 0
        return keypoints, flows, rgb

    # ---------- flow numeric ops ----------
    def add_flow_noise(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Add i.i.d. Gaussian noise N(0, sigma^2) to flow values (u,v).
        Controlled by p_aug; no change to shape or dtype.
        """
        if flows is None or torch.rand((), device=flows.device) >= float(self.flow_noise_std):
            return flows
        if self.flow_noise_std <= 0:
            return flows

        noise = torch.randn_like(flows) * float(self.flow_noise_std)
        return flows + noise

    def random_flow_occlusion(self, flows: Optional[torch.Tensor]):
        if flows is None:
            return None
        device = flows.device
        if torch.rand((), device=device) >= 0.1:
            return flows

        B, C, T, H, W = flows.shape  # C=2
        # 將 occ_max 視為遮擋尺寸比例（對齊 aug_func.py 的 occl_size_ratio）
        occl_size_ratio = float(self.occ_max)
        if occl_size_ratio <= 0:
            return flows

        h = max(1, int(H * occl_size_ratio))
        w = max(1, int(W * occl_size_ratio))

        out = flows.clone()
        for b in range(B):
            # 每個樣本抽一個位置；同一塊遮擋套用到該樣本的所有幀（與 aug_func.py 一致）
            y0 = torch.randint(low=0, high=max(1, H - h + 1), size=(1,), device=device).item()
            x0 = torch.randint(low=0, high=max(1, W - w + 1), size=(1,), device=device).item()
            out[b, :, :, y0:y0 + h, x0:x0 + w] = 0
        return out


    def gaussian_blur_flow(self, flows):
        if flows is None:
            return None
        device = flows.device
        if torch.rand((), device=device) >= 0.3:
            return flows

        k = int(self.blur_ksize)
        if k <= 1:
            return flows
        if k % 2 == 0:
            k += 1  # 強制奇數大小

        sigma = float(self.blur_sigma)
        if sigma <= 0:
            # 常見近似：OpenCV 的預設估計（可微調）
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8

        # 產生 2D 高斯核
        coords = torch.arange(k, device=device) - (k // 2)
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = (g / g.sum()).unsqueeze(1)            # (k,1)
        kernel2d = (g @ g.t())                    # (k,k)
        kernel2d = kernel2d / kernel2d.sum()

        B, C, T, H, W = flows.shape               # C=2
        flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().reshape(B*T, C, H, W)  # (B*T,2,H,W)

        # 準備 depthwise（groups=2）卷積權重
        weight = kernel2d.reshape(1, 1, k, k).repeat(C, 1, 1, 1)  # (2,1,k,k)
        padding = k // 2
        '''
        out = F.conv2d(flows_bt, weight, bias=None, stride=1, padding=padding, groups=C)
        '''
        flows_bt = F.pad(flows_bt, (padding, padding, padding, padding), mode='reflect')
        out = F.conv2d(flows_bt, weight, padding=0, groups=C)

        flows = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return flows
