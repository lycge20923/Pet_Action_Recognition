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
        per_frame: bool = True,   # per-frame random params (vs per-clip)
        augment_params: AugmentationArguments = None,

        # probability
        p_aug: float = 0.5,
    ):
        super().__init__()
        # global
        self.enable = bool(augment_params.augment)
        self.align_corners = bool(align_corners)
        self.resample_mode = str(resample_mode)
        self.pad_mode = str(pad_mode)
        self.clamp_xy = bool(clamp_xy)
        self.per_frame = bool(per_frame)

        # rotate
        self.p_aug = float(p_aug)
        self.rotate_max_deg = float(augment_params.rot_max)

        # scale
        self.scale_min = float(augment_params.scale_min)
        self.scale_max = float(augment_params.scale_max)

        # translate
        self.translate_x_max = float(augment_params.trans_max)
        self.translate_y_max = float(augment_params.trans_max)

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
        self.occ_max = float(augment_params.flow_occl_prob)
        self.blur_ksize = int(augment_params.flow_blur_ksize)
        self.blur_sigma = float(augment_params.flow_blur_sigma)
        
        self.valid_kpt_confidence_thresh = float(augment_params.valid_kpt_confidence_thresh)
    
    def _pick_device(self, keypoints, flows):
        if keypoints is not None:
            return keypoints.device
        if flows is not None:
            return flows.device
        # 都是 None（理論上 forward 已避免），退回 CPU
        return torch.device("cpu")

    def _coin(self, device):
        # True 表示要做 augmentation；False 表示跳過
        return torch.rand((), device=device) >= 0.5

    @torch.no_grad()
    def get_center(self, keypoints):
        """
        Args:
            keypoints: (B, 3, T, V) – normalized (x, y, conf)
        Returns:
            cx, cy: (B, T, 1) – center of keypoints
        """
        mask = keypoints[:, 2] > 0  # conf > 0
        x = torch.where(mask, keypoints[:, 0], torch.zeros_like(keypoints[:, 0]))
        y = torch.where(mask, keypoints[:, 1], torch.zeros_like(keypoints[:, 1]))
        valid = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)

        cx = (x.sum(dim=-1, keepdim=True) / valid)
        cy = (y.sum(dim=-1, keepdim=True) / valid)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations in the same ORDER as the original aug_func.
        (暫時只放呼叫順序；各函式會在下一步逐一實作)
        """
        if (not self.enable) or (not self.training):
            return keypoints, flows

        k, f = keypoints, flows

        # 幾何（以 skeleton center 為 pivot）
        k, f = self.rotate(k, f)
        k, f = self.scale(k, f)
        k, f = self.translate(k, f)
        k, f = self.shear(k, f)
        k, f = self.hflip(k, f)

        # 時間
        k, f = self.temporal_jitter(k, f)   
        k, f = self.frame_drop(k, f)

        # 流場數值類
        f = self.add_flow_noise(f)
        f = self.random_flow_occlusion(f)
        f = self.gaussian_blur_flow(f)

        return k, f

    # ---------- geometric ops (skeleton-center pivot) ----------
    def rotate(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor]):
        """
        目標：
        - 一進來先丟硬幣：<0.5 直接 return 原樣
        - None-safe：
            * 兩者皆 None → 原樣返回
            * 只有 flows → 以影像中心 (0.5,0.5) 為 pivot，對 flow 影像 + flow 向量旋轉
            * 只有 keypoints → 以 skeleton center 為 pivot，僅旋轉 keypoints
            * 兩者皆在 → 如你現有邏輯
        """
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # === 準備 B,T, 以及 pivot (cx,cy) ===
        # pivot 來源：
        #   - 有 keypoints：用 get_center(keypoints)
        #   - 否則：用影像中心 0.5,0.5
        if keypoints is not None:
            assert keypoints.dim() == 4 and keypoints.size(1) == 3, "keypoints must be (B,3,T,V)"
            Bk, _, Tk, Vk = keypoints.shape
            cx, cy = self.get_center(keypoints)  # (B,T,1),(B,T,1)
            B, T = Bk, Tk
        else:
            cx = cy = None
            B, T = flows.shape[0], flows.shape[2]  # flows 不會是 None（上面已過濾）

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
                cx = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)
                cy = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)

            # to [-1,1]
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
            theta = theta.view(B*T, 2, 3)

            flows_bt = flows.permute(0,2,1,3,4).contiguous().view(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img_rot = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # 旋轉 flow 向量值
            u = flows_img_rot[:, 0:1]
            v = flows_img_rot[:, 1:2]
            R = torch.stack([
                torch.stack([cos_t.reshape(B*T), -sin_t.reshape(B*T)], dim=-1),
                torch.stack([sin_t.reshape(B*T),  cos_t.reshape(B*T)], dim=-1)
            ], dim=-2)  # (B*T,2,2)
            uv = torch.cat([u, v], dim=1).view(B*T, 2, H*W)
            uv_rot = torch.bmm(R, uv).view(B*T, 2, H, W)
            flows = uv_rot.view(B, T, 2, H, W).permute(0,2,1,3,4).contiguous()

        # === keypoints 分支 ===
        if keypoints is not None:
            dtype_kpt = keypoints.dtype
            prep = self.common_preprocess_for_augmentation(keypoints)
            x, y, conf = prep["x"], prep["y"], prep["conf"]
            cx_k, cy_k = prep["cx"], prep["cy"]
            cos_t3, sin_t3 = cos_t[..., None], sin_t[..., None]
            x0, y0 = x - cx_k, y - cy_k
            x_rot = x0 * cos_t3 - y0 * sin_t3 + cx_k
            y_rot = x0 * sin_t3 + y0 * cos_t3 + cy_k
            if self.clamp_xy:
                x_rot = x_rot.clamp(0,1)
                y_rot = y_rot.clamp(0,1)
            out = keypoints.clone()
            out[:,0], out[:,1], out[:,2] = x_rot.to(dtype_kpt), y_rot.to(dtype_kpt), conf
            keypoints = out

        return keypoints, flows
    

    def scale(self, keypoints, flows):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # 取 B,T
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        else:
            B, _, T, _, _ = flows.shape

        # 取每幀/每段的 scale 因子 s ∈ [scale_min, scale_max]
        s_min = float(self.scale_min)
        s_max = float(self.scale_max)
        if self.per_frame:
            s = torch.empty(B, T, device=device).uniform_(s_min, s_max)  # (B,T)
        else:
            s = torch.empty(B, 1, device=device).uniform_(s_min, s_max).expand(B, T)  # (B,T)

        # --- flows 分支：影像 warp + 向量值域縮放 ---
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, "scale: (B,T) mismatch between keypoints and flows"
            dtype_img = flows.dtype

            # pivot：有 kps 用骨架中心；否則用影像中心 0.5,0.5
            if keypoints is not None:
                cx, cy = self.get_center(keypoints)  # (B,T,1)
            else:
                cx = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)
                cy = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)

            # 轉到 [-1,1] 座標求平移項（保持 pivot 不動）：t = (I - S) * p
            cx_n = (cx * 2) - 1
            cy_n = (cy * 2) - 1
            s3 = s[..., None]  # (B,T,1)
            tx = cx_n * (1.0 - s3)         # (B,T,1)
            ty = cy_n * (1.0 - s3)

            theta = torch.zeros(B, T, 2, 3, device=device, dtype=dtype_img)
            theta[..., 0, 0] = s           # a11 = s
            theta[..., 1, 1] = s           # a22 = s
            theta[..., 0, 2] = tx.squeeze(-1)
            theta[..., 1, 2] = ty.squeeze(-1)
            theta = theta.view(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().view(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # flow 向量值域同步縮放（u,v 乘 s）
            s_flat = s.view(B*T, 1, 1, 1)
            flows_img = flows_img * s_flat

            flows = flows_img.view(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # --- keypoints 分支：以 pivot 縮放 (x,y) ---
        if keypoints is not None:
            prep = self.common_preprocess_for_augmentation(keypoints)
            x, y, conf = prep["x"], prep["y"], prep["conf"]     # (B,T,V)
            cx, cy = prep["cx"], prep["cy"]                     # (B,T,1)
            s3 = s[..., None]                                   # (B,T,1)

            x_s = (x - cx) * s3 + cx
            y_s = (y - cy) * s3 + cy
            if self.clamp_xy:
                x_s = x_s.clamp(0, 1)
                y_s = y_s.clamp(0, 1)

            out = keypoints.clone()
            out[:, 0], out[:, 1], out[:, 2] = x_s, y_s, conf
            keypoints = out

        return keypoints, flows

    def translate(self, keypoints, flows):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # 取 B,T
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        else:
            B, _, T, _, _ = flows.shape

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
            theta = theta.view(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().view(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )
            flows = flows_img.view(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()
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
            out = keypoints.clone()
            out[:, 0], out[:, 1], out[:, 2] = x_t, y_t, conf
            keypoints = out

        return keypoints, flows

    def shear(self, keypoints, flows):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # 取 B,T
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        else:
            B, _, T, _, _ = flows.shape

        # 取剪切係數 shx, shy ∈ [-shear_max, +shear_max]
        shx_max = float(self.shear_x_max)
        shy_max = float(self.shear_y_max)
        if self.per_frame:
            shx = (torch.rand(B, T, device=device) * 2 * shx_max) - shx_max  # (B,T)
            shy = (torch.rand(B, T, device=device) * 2 * shy_max) - shy_max
        else:
            shx = ((torch.rand(B, 1, device=device) * 2 * shx_max) - shx_max).expand(B, T)
            shy = ((torch.rand(B, 1, device=device) * 2 * shy_max) - shy_max).expand(B, T)

        # --- flows 分支：影像 warp（保持 pivot），並對 (u,v) 施加同一剪切 ---
        if flows is not None:
            assert flows.dim() == 5 and flows.size(1) == 2
            Bf, _, Tf, H, W = flows.shape
            assert B == Bf and T == Tf, "shear: (B,T) mismatch between keypoints and flows"
            dtype_img = flows.dtype

            # pivot：有 kps 用骨架中心；否則用影像中心 0.5,0.5
            if keypoints is not None:
                cx, cy = self.get_center(keypoints)  # (B,T,1)
            else:
                cx = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)
                cy = torch.full((B, T, 1), 0.5, device=device, dtype=dtype_img)

            cx_n = (cx * 2) - 1    # [-1,1]
            cy_n = (cy * 2) - 1

            # A = [[1, shx], [shy, 1]],  t = (I - A) p = [-shx*cy, -shy*cx]
            shx3 = shx[..., None]  # (B,T,1)
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
            theta = theta.view(B*T, 2, 3)

            flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().view(B*T, 2, H, W)
            grid = F.affine_grid(theta, size=(B*T, 2, H, W), align_corners=self.align_corners)
            flows_img = F.grid_sample(
                flows_bt, grid,
                mode=self.resample_mode, padding_mode=self.pad_mode,
                align_corners=self.align_corners
            )  # (B*T,2,H,W)

            # flow 向量值域： [u'; v'] = [[1, shx], [shy, 1]] @ [u; v]
            u = flows_img[:, 0:1]  # (B*T,1,H,W)
            v = flows_img[:, 1:2]
            shx_f = shx.view(B*T, 1, 1, 1)
            shy_f = shy.view(B*T, 1, 1, 1)
            u_new = u + shx_f * v
            v_new = shy_f * u + v
            flows_img = torch.cat([u_new, v_new], dim=1)
            flows = flows_img.view(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # --- keypoints 分支：相對 pivot 做剪切 ---
        if keypoints is not None:
            prep = self.common_preprocess_for_augmentation(keypoints)
            x, y, conf = prep["x"], prep["y"], prep["conf"]   # (B,T,V)
            cx, cy = prep["cx"], prep["cy"]                   # (B,T,1)
            shx3 = shx[..., None]                             # (B,T,1)
            shy3 = shy[..., None]

            dx = x - cx
            dy = y - cy
            x_s = dx + shx3 * dy + cx
            y_s = shy3 * dx + dy + cy
            if self.clamp_xy:
                x_s = x_s.clamp(0, 1)
                y_s = y_s.clamp(0, 1)

            out = keypoints.clone()
            out[:, 0], out[:, 1], out[:, 2] = x_s, y_s, conf
            keypoints = out

        return keypoints, flows


    def hflip(self, keypoints, flows):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

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

        return keypoints, flows


    # ---------- temporal ops ----------
    def temporal_jitter(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor]):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # 取 T 與 B
        if keypoints is not None:
            B, _, T, V = keypoints.shape
        else:
            B, _, T, H, W = flows.shape
        if T <= 2 or self.p_temporal_jitter <= 0:
            return keypoints, flows

        # 逐 batch/逐 frame 的 Bernoulli mask（沿用你現有設計）
        dtype_i = torch.long
        jitter_mask = torch.rand(B, T, device=device) < self.p_temporal_jitter  # (B,T)

        # 避免挑到首末（對應你的「避免越界→clamp」邏輯，可保留）
        base = torch.arange(T, device=device)[None, :].expand(B, T)  # (B,T)
        k = int(self.jitter_max_offset)
        offsets = torch.randint(low=-k, high=k + 1, size=(B, T), device=device, dtype=dtype_i)

        # 若 mask=True 但 offset=0，強制 ±1
        zero_o = (offsets == 0) & jitter_mask
        if zero_o.any():
            alt = torch.where(torch.rand_like(offsets.float()) > 0.5, torch.ones_like(offsets), -torch.ones_like(offsets))
            offsets = torch.where(zero_o, alt, offsets)

        new_idx = (base + offsets).clamp_(0, T - 1)
        final_idx = torch.where(jitter_mask, new_idx, base)  # (B,T)

        if keypoints is not None:
            _, Ck, _, Vk = keypoints.shape  # Ck=3
            k_idx = final_idx[:, None, :, None].expand(B, Ck, T, Vk)
            keypoints = torch.gather(keypoints, dim=2, index=k_idx).contiguous()
        if flows is not None:
            _, Cf, _, H, W = flows.shape  # Cf=2
            f_idx = final_idx[:, None, :, None, None].expand(B, Cf, T, H, W)
            flows = torch.gather(flows, dim=2, index=f_idx).contiguous()

        return keypoints, flows

    def frame_drop(self, keypoints: Optional[torch.Tensor], flows: Optional[torch.Tensor]):
        device = self._pick_device(keypoints, flows)
        if not self._coin(device):
            return keypoints, flows
        if keypoints is None and flows is None:
            return None, None

        # 取 T 與 B
        if keypoints is not None:
            B, _, T, _ = keypoints.shape
        else:
            B, _, T, _, _ = flows.shape
        if T <= 1 or self.frame_drop_ratio <= 0:
            return keypoints, flows

        num_drop = max(1, int(round(self.frame_drop_ratio * T)))
        idx = torch.randperm(T, device=device)[:num_drop]  # (num_drop,)

        if keypoints is not None:
            keypoints = keypoints.clone()
            keypoints[:, :, idx, :] = 0
        if flows is not None:
            flows = flows.clone()
            flows[:, :, idx, :, :] = 0
        return keypoints, flows

    # ---------- flow numeric ops ----------
    def add_flow_noise(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Add i.i.d. Gaussian noise N(0, sigma^2) to flow values (u,v).
        Controlled by p_aug; no change to shape or dtype.
        """
        if flows is None or torch.rand((), device=flows.device) >= float(self.p_aug):
            return flows
        if self.flow_noise_std <= 0:
            return flows

        noise = torch.randn_like(flows) * float(self.flow_noise_std)
        return flows + noise

    def random_flow_occlusion(self, flows: Optional[torch.Tensor]):
        if flows is None:
            return None
        device = flows.device
        if not self._coin(device):
            return flows
        if self.occ_max <= 0:
            return flows

        B, _, T, H, W = flows.shape
        dtype_i = torch.long
        occ_min, occ_max = float(self.occ_min), float(self.occ_max)

        h_frac = torch.rand(B, T, device=device) * (occ_max - occ_min) + occ_min
        w_frac = torch.rand(B, T, device=device) * (occ_max - occ_min) + occ_min
        h_box = (h_frac * H).clamp(min=1).to(dtype_i)  # (B,T)
        w_box = (w_frac * W).clamp(min=1).to(dtype_i)

        max_y0 = (H - h_box).clamp(min=0)
        max_x0 = (W - w_box).clamp(min=0)

        y0 = torch.empty(B, T, device=device, dtype=dtype_i)
        x0 = torch.empty(B, T, device=device, dtype=dtype_i)
        for b in range(B):
            for t in range(T):
                y0[b, t] = torch.randint(0, int(max_y0[b, t].item()) + 1, (1,), device=device)
                x0[b, t] = torch.randint(0, int(max_x0[b, t].item()) + 1, (1,), device=device)

        out = flows.clone()
        for b in range(B):
            for t in range(T):
                yy = int(y0[b, t])
                xx = int(x0[b, t])
                hh = int(h_box[b, t])
                ww = int(w_box[b, t])
                out[b, :, t, yy:yy+hh, xx:xx+ww] = 0
        return out

    def gaussian_blur_flow(self, flows):
        if flows is None:
            return None
        device = flows.device
        if not self._coin(device):
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
        flows_bt = flows.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # (B*T,2,H,W)

        # 準備 depthwise（groups=2）卷積權重
        weight = kernel2d.view(1, 1, k, k).repeat(C, 1, 1, 1)  # (2,1,k,k)
        padding = k // 2
        out = F.conv2d(flows_bt, weight, bias=None, stride=1, padding=padding, groups=C)

        flows = out.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return flows
