import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .degcn import Basic_Block
from .gcn.graph import Graph
from ..utils.cli_args import ModelArguments, DataArguments

"""
KP Embedding
"""

class KP_EMBEDDING(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.blocks = nn.ModuleList([
                Basic_Block(in_channels=3, out_channels=32, A=self.A, num_frame=32, eta=4),
                Basic_Block(in_channels=32, out_channels=64, A=self.A, num_frame=32, eta=4),
                Basic_Block(in_channels=64, out_channels=128, A=self.A, num_frame=32, eta=4),
                Basic_Block(in_channels=128, out_channels=256, A=self.A, num_frame=32, eta=4),
                ]
            )
    def forward(self, keypoints: torch.Tensor):
        x = keypoints
        for block in self.blocks:
            x = block(x)
        return x # (B, E, T, V)

"""
MiniI3D
"""
def _best_grid(P: int):
    # 盡量接近方形的 gh × gw，使 gh*gw >= P（但我們後面只會用前 P 個或直接用 P）
    g = int(math.sqrt(P))
    while g > 1 and P % g != 0:
        g -= 1
    gh = g
    gw = P // g if P % g == 0 else (P + g - 1) // g
    return gh, gw

def build_2d_sincos(h: int, w: int, dim: int, device=None):
    """2D sin-cos，將 dim 一分為二給 y/x（各自 sin+cos），要求 dim % 4 == 0。"""
    assert dim % 4 == 0, "E 必須可被 4 整除以使用 2D sin-cos（例如 256）"
    device = device or "cpu"
    y = torch.arange(h, device=device).float().unsqueeze(1).repeat(1, w)  # (h,w)
    x = torch.arange(w, device=device).float().unsqueeze(0).repeat(h, 1)  # (h,w)
    wy = torch.arange(dim // 4, device=device).float() / (dim // 4)
    wx = torch.arange(dim // 4, device=device).float() / (dim // 4)
    wy = 1.0 / (10000 ** wy)  # (dim/4,)
    wx = 1.0 / (10000 ** wx)
    posy = (y[..., None] * wy).contiguous()  # (h,w,dim/4)
    posx = (x[..., None] * wx).contiguous()
    posy = torch.cat([posy.sin(), posy.cos()], dim=-1)  # (h,w,dim/2)
    posx = torch.cat([posx.sin(), posx.cos()], dim=-1)
    pe = torch.cat([posy, posx], dim=-1)  # (h,w,dim)
    return pe  # (h,w,dim)

def build_1d_sincos(t: int, dim: int, device=None):
    """1D sin-cos，dim 必須是偶數。"""
    assert dim % 2 == 0, "E 必須為偶數以使用 1D sin-cos"
    device = device or "cpu"
    pos = torch.arange(t, device=device).float()[:, None]           # (T,1)
    w   = torch.arange(dim // 2, device=device).float() / (dim // 2)
    w   = 1.0 / (10000 ** w)                                        # (dim/2,)
    out = pos * w[None, :]                                          # (T,dim/2)
    pe  = torch.cat([out.sin(), out.cos()], dim=-1)                 # (T,dim)
    return pe

# MiniI3D
class MiniI3D(nn.Module):
    """
    Input : (B, 2, T, H, W)
    Output: (B, Cb, T, H', W')   # H'≈H/4, W'≈W/4, T preserved
    """
    def __init__(self, Cb: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),

            nn.Conv3d(128, Cb, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(Cb), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)

"""
I3D Embedding
"""

class I3DEmbedding(nn.Module):
    def __init__(self,
                 add_temp_encoder: bool = True,
                 E: int = 256,
                 P: int = 16,
                 target_T: int | None = None,
                 Cb: int = 256,
                 drop: float = 0.1,
                 hidden: int = 256,
                 # Transformer 相關
                 nheads: int = 4,
                 n_layers: int = 2,
                 ffn_mult: int = 4):
        super().__init__()
        self.E = E
        self.P = P
        self.target_T = target_T

        # -------- Backbone & 專案化 --------
        self.backbone = MiniI3D(Cb=Cb)
        self.proj = nn.Conv3d(Cb, E, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm3d(E)
        self.act  = nn.ReLU(inplace=True)

        # -------- 空間格點聚合（保持原有邏輯）--------
        self.gh, self.gw = _best_grid(P)     # S ~ P
        self.pool2d = nn.AdaptiveAvgPool2d((self.gh, self.gw))

        # 注意：保持單層 cross-attn 聚合以維持輕量（資料 4k 影片）
        self.q = nn.Parameter(torch.randn(1, 1, P, E) * 0.02)  # (1,1,P,E)
        self.k_lin = nn.Linear(E, E, bias=False)
        self.v_lin = nn.Linear(E, E, bias=False)
        self.attn_drop = nn.Dropout(drop)

        self.ln_tok = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, E),
            nn.Dropout(drop),
        )

        # -------- 位置編碼（可插值）--------
        # 2D：建在 buffer，依照 (gh, gw, E)；時間的 1D 每次 forward 依 T 生成
        pe2d = build_2d_sincos(self.gh, self.gw, E)  # (gh,gw,E)
        self.register_buffer("pe2d", pe2d, persistent=False)

        # -------- 時序 Transformer（沿 T 建模；輸出維度不變）--------
        self.add_temp_encoder = add_temp_encoder
        if self.add_temp_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=E,
                nhead=nheads,
                dim_feedforward=E * ffn_mult,
                dropout=drop,
                batch_first=True,   # 讓輸入用 (N, T, E)
                norm_first=True     # Pre-LN 比較穩
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        B = flows.size(0)

        # 1) backbone -> (B,Cb,T',H',W')
        z = self.backbone(flows)

        # 2) project to E
        z = self.act(self.bn(self.proj(z)))                 # (B,E,T',H',W')

        # 3) optional temporal align
        if self.target_T is not None and z.shape[2] != self.target_T:
            z = F.interpolate(
                z, size=(self.target_T, z.shape[3], z.shape[4]),
                mode="trilinear", align_corners=False
            )                                              # (B,E,T,H',W')
        T = z.shape[2]

        # 4) make spatial tokens per frame via pooling to gh×gw
        #    並在格點 tokens 上加 2D 位置（幫助隨後的空間聚合）
        z_bt_e_hw = z.permute(0, 2, 1, 3, 4).contiguous()               # (B,T,E,H',W')
        z_bt_e_hw = z_bt_e_hw.view(B*T, z_bt_e_hw.shape[2], z_bt_e_hw.shape[3], z_bt_e_hw.shape[4])  # (B*T,E,H',W')
        z_bt_e_ghgw = self.pool2d(z_bt_e_hw)                             # (B*T,E,gh,gw)

        # 加 2D sin-cos：broadcast 到通道 E
        pe2d = self.pe2d.view(1, self.gh, self.gw, self.E).permute(0,3,1,2)   # (1,E,gh,gw)
        z_bt_e_ghgw = z_bt_e_ghgw + pe2d                                      # (B*T,E,gh,gw)

        # 展為 (B,T,S,E)
        z_tok_grid = z_bt_e_ghgw.view(B, T, self.E, self.gh*self.gw)          # (B,T,E,S)
        z_tok_grid = z_tok_grid.permute(0, 1, 3, 2).contiguous()              # (B,T,S,E), S=gh*gw

        # 5) attention pooling: P queries over S spatial tokens  (K/V + 時間PE，Q + 時間PE)
        # 時間 1D sin-cos（可插值）：每次 forward 依 T 重新生成
        pe1d = build_1d_sincos(T, self.E, device=z_tok_grid.device)    # (T,E)

        K = self.k_lin(z_tok_grid + pe1d.view(1, T, 1, self.E))        # (B,T,S,E)
        V = self.v_lin(z_tok_grid + pe1d.view(1, T, 1, self.E))        # (B,T,S,E)
        Q = self.q.expand(B, T, self.P, self.E) + pe1d.view(1, T, 1, self.E)  # (B,T,P,E)

        attn = (Q @ K.transpose(-1, -2)) / (self.E ** 0.5)             # (B,T,P,S)
        attn = self.attn_drop(attn.softmax(dim=-1))
        z_tok = attn @ V                                               # (B,T,P,E)

        # 6) token-wise LN + MLP (residual)
        z_tok = self.ln_tok(z_tok)                                     # (B,T,P,E)
        z_tok = self.mlp(z_tok) + z_tok                                # (B,T,P,E)

        # 7) 時序 Transformer（沿 T 建模；不改 P，不改 E）
        #    將每個 slot p 的序列（長度 T）獨立送入 Encoder
        x = z_tok  # (B,T,P,E)
        x = x.permute(0, 2, 1, 3).contiguous()     # (B,P,T,E)
        x = x.view(B * self.P, T, self.E)          # (B*P, T, E)

        # 加 1D 時間位置（再次加在 Transformer 輸入；保持明確的時序偏置）
        x = x + pe1d.view(1, T, self.E)
        
        if self.add_temp_encoder:
            x = self.temporal_encoder(x)               # (B*P, T, E)
        x = x.view(B, self.P, T, self.E).permute(0, 2, 1, 3).contiguous()  # (B,T,P,E)

        # 8) to (B,E,T,P)
        out = x.permute(0, 3, 1, 2).contiguous()
        return out 

"""
Final Cross Attention 
"""    

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class CrossAttnBlock(nn.Module):
    """對稱 Cross-Attn：A←B 與 B←A。輸入 shape 都是 (B*, N, E)。"""
    def __init__(self, dim, nheads=4, drop=0.1):
        super().__init__()
        self.attn_a = nn.MultiheadAttention(dim, nheads, dropout=drop, batch_first=True)
        self.attn_b = nn.MultiheadAttention(dim, nheads, dropout=drop, batch_first=True)
        self.ln_a1 = nn.LayerNorm(dim); self.ln_b1 = nn.LayerNorm(dim)
        self.ffn_a = MLP(dim, drop=drop); self.ffn_b = MLP(dim, drop=drop)
        self.ln_a2 = nn.LayerNorm(dim); self.ln_b2 = nn.LayerNorm(dim)

    def forward(self, a, b, mask_a=None, mask_b=None):
        # A ← B
        a2, _ = self.attn_a(self.ln_a1(a), self.ln_b1(b), self.ln_b1(b), key_padding_mask=mask_b)
        a = a + a2
        a = a + self.ffn_a(self.ln_a2(a))
        # B ← A
        b2, _ = self.attn_b(self.ln_b1(b), self.ln_a1(a), self.ln_a1(a), key_padding_mask=mask_a)
        b = b + b2
        b = b + self.ffn_b(self.ln_b2(b))
        return a, b

class JointSelfAttnBlock(nn.Module):
    """把兩模態 token 串起來做 self-attn 混合。輸入 (B*, N, E)。"""
    def __init__(self, dim, nheads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nheads, dropout=drop, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, drop=drop)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x, key_padding_mask=None):
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x

class ATT_GCN(nn.Module):
    """
    將骨架 GCN 與影像(I3D+slot)表徵融合：
    1) __init__ 直接初始化 KP_EMBEDDING 與 I3DEmbedding
    2) forward 接受 keypoints 與 optical flows，依序經過兩個 encoder
    3) 時間步內 cross-attn + joint self-attn，之後逐 token 的時序 Transformer
    4) 平均池化 + MLP 分類
    """
    def __init__(self,
                 num_nodes, neighbor_base, num_classes,
                 add_joint_attention=True,
                 add_temp_encoder=True,
                 E=256,
                 P=16,
                 # 融合/Transformer 參數
                 nheads=4,
                 depth_cross=1,
                 depth_joint=1,
                 depth_temporal=4,
                 drop=0.1,
                 # I3D backbone 參數（視你的 I3DEmbedding 而定）
                 i3d_add_temp_encoder=True,
                 i3d_target_T=None,
                 i3d_Cb=256,
                 i3d_hidden=256,
                 i3d_nheads=4,
                 i3d_nlayers=2,
                 i3d_ffn_mult=4,
                 i3d_drop=0.1):
        super().__init__()
        self.E = E
        self.P = P

        # 1) 初始化兩個 encoder
        graph = Graph(num_nodes=num_nodes, neighbor_base=neighbor_base)
        A = graph.A  # (3, V, V)
        self.kp_encoder = KP_EMBEDDING(A)  # 假設輸出通道最終為 E=256
        self.vid_encoder = I3DEmbedding(add_temp_encoder=i3d_add_temp_encoder,
            E=E, P=P, target_T=i3d_target_T, Cb=i3d_Cb, drop=i3d_drop,
            hidden=i3d_hidden, nheads=i3d_nheads, n_layers=i3d_nlayers, ffn_mult=i3d_ffn_mult
        )

        # 若 KP_EMBEDDING 最終通道不是 E，可用 1x1 Conv 做投影（保險）
        self.proj_kp = nn.Identity()
        self.proj_vid = nn.Identity()

        # 2) 模態嵌入（讓注意力知道誰是骨架/誰是影像 slot）
        self.mod_embed_kp  = nn.Parameter(torch.zeros(1, 1, E))
        self.mod_embed_vid = nn.Parameter(torch.zeros(1, 1, E))
        nn.init.trunc_normal_(self.mod_embed_kp, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_vid, std=0.02)

        # 3) 時間步內的跨模態注意 + 混合注意
        self.add_joint_attention =  add_joint_attention
        if self.add_joint_attention:
            self.cross_blocks = nn.ModuleList([CrossAttnBlock(E, nheads, drop) for _ in range(depth_cross)])
            self.joint_blocks = nn.ModuleList([JointSelfAttnBlock(E, nheads, drop) for _ in range(depth_joint)])

        # 4) 逐 token 的時序 Transformer
        self.add_temp_encoder = add_temp_encoder
        if self.add_temp_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=E, nhead=nheads, dim_feedforward=E*4, dropout=drop,
                batch_first=True, norm_first=True, activation="gelu"
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth_temporal)

        # 5) 分類頭
        self.head_norm = nn.LayerNorm(E)
        self.head = nn.Sequential(
            nn.Linear(E, E),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(E, num_classes)
        )

    @staticmethod
    def _interp_time(x, target_T: int):
        """
        x: (B, E, T, N)  → 對 T 做線性插值到 target_T
        """
        B, E, T, N = x.shape
        if T == target_T:
            return x
        x = x.permute(0, 3, 1, 2).contiguous().view(B*N, E, T)  # (B*N, E, T)
        x = F.interpolate(x, size=target_T, mode='linear', align_corners=False)
        x = x.view(B, N, E, target_T).permute(0, 2, 3, 1).contiguous()  # (B, E, T, N)
        return x

    def forward(self, keypoints: torch.Tensor, flows: torch.Tensor, mask_kp: torch.Tensor|None=None):
        """
        keypoints: (B, C=3, T_kp, V)
        flows:     (B, C_flow, T_vid, H, W)  （或你定義的 I3D 輸入）
        mask_kp:   (B, T_kp, V)，1=有效、0=無效（可選）
        """
        # === 先各自編碼 ===
        kp_feat  = self.kp_encoder(keypoints)   # (B, E, T_kp, V)
        vid_feat = self.vid_encoder(flows)      # (B, E, T_vid, P)

        # 如有需要可投影到同一 E
        kp_feat  = self.proj_kp(kp_feat)
        vid_feat = self.proj_vid(vid_feat)

        # === 時間對齊（若 I3DEmbedding 未對齊 target_T，這裡再保險一次）===
        B, E, T_kp, V = kp_feat.shape
        T_vid, P = vid_feat.shape[2], vid_feat.shape[3]
        if T_vid != T_kp:
            vid_feat = self._interp_time(vid_feat, T_kp)  # (B, E, T_kp, P)

        # === 時間步內 Cross-Attn + Joint Self-Attn ===
        # 攤平成 (B*T, N, E)，並加模態嵌入
        kp_bt = kp_feat.permute(0, 2, 3, 1).contiguous().view(B*T_kp, V, E)   # (B*T, V, E)
        vid_bt = vid_feat.permute(0, 2, 3, 1).contiguous().view(B*T_kp, P, E) # (B*T, P, E)
        kp_bt = kp_bt + self.mod_embed_kp
        vid_bt = vid_bt + self.mod_embed_vid

        # 準備 key_padding_mask（True=要遮）
        mask_kp_bt = None
        if mask_kp is not None:
            mask_kp_bt = (mask_kp.view(B*T_kp, V) == 0)
        mask_vid_bt = None  # slot 通常不需要 mask

        # Cross-Attn
        if self.add_joint_attention:
            for blk in self.cross_blocks:
                kp_bt, vid_bt = blk(kp_bt, vid_bt, mask_a=mask_kp_bt, mask_b=mask_vid_bt)

            # Joint Self-Attn
            x = torch.cat([kp_bt, vid_bt], dim=1)  # (B*T, V+P, E)
            joint_mask = None
            if mask_kp_bt is not None:
                zeros_vid = torch.zeros(B*T_kp, P, dtype=torch.bool, device=x.device)
                joint_mask = torch.cat([mask_kp_bt, zeros_vid], dim=1)
            for blk in self.joint_blocks:
                x = blk(x, key_padding_mask=joint_mask)  # (B*T, V+P, E)
        else:
            x = torch.cat([kp_bt, vid_bt], dim=1)

        # === 逐 token 的時序 Transformer（時間注意力）===
        x = x.view(B, T_kp, V+P, E).permute(0, 2, 1, 3).contiguous().view(B*(V+P), T_kp, E)  # (B*(V+P), T, E)
        if self.add_temp_encoder:
            x = self.temporal_encoder(x)  # (B*(V+P), T, E)
        x = x.view(B, V+P, T_kp, E)

        # === 池化 + 分類 ===
        feat = x.mean(dim=2).mean(dim=1)  # 時間平均 + token 平均 → (B, E)
        feat = self.head_norm(feat)
        logits = self.head(feat)          # (B, num_classes)
        return feat, logits
    

###################################################
##                    trial                      ##
###################################################

# simple I3D Embedding

class I3DEmbedding_Simple(nn.Module):
    def __init__(self,
                 E: int = 256,
                 P: int = 16,
                 target_T: int | None = None,
                 Cb: int = 256,
                 drop: float = 0.1,
                 hidden: int = 256,
                 # Transformer 相關
                 nheads: int = 4,
                 n_layers: int = 2,
                 ffn_mult: int = 4):
        super().__init__()
        self.E = E
        self.P = P
        self.target_T = target_T

        # -------- Backbone & 專案化 --------
        self.backbone = MiniI3D(Cb=Cb)
        self.proj = nn.Conv3d(Cb, E, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm3d(E)
        self.act  = nn.ReLU(inplace=True)

        # -------- 空間格點聚合（保持原有邏輯）--------
        self.gh, self.gw = _best_grid(P)     # S ~ P
        self.pool2d = nn.AdaptiveAvgPool2d((self.gh, self.gw))

        self.ln_tok = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, E),
            nn.Dropout(drop),
        )

        # -------- 位置編碼（可插值）--------
        # 2D：建在 buffer，依照 (gh, gw, E)；時間的 1D 每次 forward 依 T 生成
        pe2d = build_2d_sincos(self.gh, self.gw, E)  # (gh,gw,E)
        self.register_buffer("pe2d", pe2d, persistent=False)

        # -------- 時序 Transformer（沿 T 建模；輸出維度不變）--------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=E,
            nhead=nheads,
            dim_feedforward=E * ffn_mult,
            dropout=drop,
            batch_first=True,   # 讓輸入用 (N, T, E)
            norm_first=True     # Pre-LN 比較穩
        )
        self.temporal_encoder = nn.Identity()

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        B = flows.size(0)

        # 1) backbone -> (B,Cb,T',H',W')
        z = self.backbone(flows)

        # 2) project to E
        z = self.act(self.bn(self.proj(z)))                 # (B,E,T',H',W')

        # 3) optional temporal align
        if self.target_T is not None and z.shape[2] != self.target_T:
            z = F.interpolate(
                z, size=(self.target_T, z.shape[3], z.shape[4]),
                mode="trilinear", align_corners=False
            )                                              # (B,E,T,H',W')
        T = z.shape[2]

        # 4) make spatial tokens per frame via pooling to gh×gw
        #    並在格點 tokens 上加 2D 位置（幫助隨後的空間聚合）
        z_bt_e_hw = z.permute(0, 2, 1, 3, 4).contiguous()               # (B,T,E,H',W')
        z_bt_e_hw = z_bt_e_hw.view(B*T, z_bt_e_hw.shape[2], z_bt_e_hw.shape[3], z_bt_e_hw.shape[4])  # (B*T,E,H',W')
        z_bt_e_ghgw = self.pool2d(z_bt_e_hw)                             # (B*T,E,gh,gw)

        # 加 2D sin-cos：broadcast 到通道 E
        pe2d = self.pe2d.view(1, self.gh, self.gw, self.E).permute(0,3,1,2)   # (1,E,gh,gw)
        z_bt_e_ghgw = z_bt_e_ghgw + pe2d                                      # (B*T,E,gh,gw)

        # 展為 (B,T,S,E)
        z_tok_grid = z_bt_e_ghgw.view(B, T, self.E, self.gh*self.gw)          # (B,T,E,S)
        z_tok_grid = z_tok_grid.permute(0, 1, 3, 2).contiguous()              # (B,T,S,E), S=gh*gw

        # 5) no attention
        pe1d = build_1d_sincos(T, self.E, device=z_tok_grid.device)    # (T,E)

        S = z_tok_grid.shape[2]
        if S > self.P:
            z_tok = z_tok_grid[:, :, :self.P, :]  # (B,T,P,E)
        else:
            z_tok = z_tok_grid                     # (B,T,S(=P),E)

        # 6) token-wise LN + MLP (residual)
        z_tok = self.ln_tok(z_tok)                                     # (B,T,P,E)
        z_tok = self.mlp(z_tok) + z_tok                                # (B,T,P,E)

        # 7) 時序 Transformer（沿 T 建模；不改 P，不改 E）
        #    將每個 slot p 的序列（長度 T）獨立送入 Encoder
        x = z_tok  # (B,T,P,E)
        
        # 8) to (B,E,T,P)
        out = x.permute(0, 3, 1, 2).contiguous()
        return out 



# Deformable
from torchvision.ops import DeformConv2d
def apply_2d_over_time(x_3d: torch.Tensor, layer_2d: nn.Module) -> torch.Tensor:
    # (B,C,T,H,W) -> (B*T,C,H,W) -> layer_2d -> (B,C',T,H',W')
    B, C, T, H, W = x_3d.shape
    x_btchw = x_3d.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W).contiguous()
    y_btchw = layer_2d(x_btchw)  # (B*T,C',H',W')
    C2, H2, W2 = y_btchw.shape[1:]
    return y_btchw.view(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

class TVDeformConv2dPack(nn.Module):
    """
    Pack 版本：內建 offset/mask head，外部使用時就像 Conv2d 一樣。
    依據 torchvision DeformConv2d 的介面：
      - offset 通道數 = 2*k*k
      - mask   通道數 =   k*k （可選；用作 modulated deform）
    """
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.k = k
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.dcn = DeformConv2d(
            in_c, out_c, kernel_size=k, stride=self.stride, padding=self.padding,
            dilation=self.dilation, bias=bias
        )

        k2 = k * k
        # 讓 offset/mask 的空間尺寸與 DCN 輸出一致：使用同 stride/pad/dilation
        self.offset_head = nn.Conv2d(in_c, 2*k2, kernel_size=k, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation)
        self.mask_head   = nn.Conv2d(in_c,   k2,  kernel_size=k, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation)

        # 初始化：offset=0、mask≈0（sigmoid 後低於 0.5，比較保守；也可改 0 得 0.5）
        nn.init.zeros_(self.offset_head.weight); nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.mask_head.weight);   nn.init.constant_(self.mask_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 建議先用 fp32 驗證；若要 AMP，可直接在更外層開 autocast 即可
        offset = self.offset_head(x)
        mask   = self.mask_head(x).sigmoid()
        return self.dcn(x, offset, mask)
class MiniI3D_DCN2D_TV(nn.Module):
    """
    Input : (B, 2, T, H, W)
    Output: (B, Cb, T, H/4, W/4)
    兩個空間下採樣層用 Deformable 2D（stride=2），中間插一層 stride=1 的 DCN2D。
    時間維用 3×1×1（不降採樣），完全對齊你原本的形狀。
    """
    def __init__(self, Cb: int = 256, mid: int = 128, norm: str = "bn"):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        N = (lambda c: nn.GroupNorm(min(32, c), c)) if norm == "gn" else nn.BatchNorm3d

        # 時間聚合（不動 H,W）
        self.t1 = nn.Conv3d(2, 64, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.n1 = N(64)

        # 空間 DCN2D 下採樣 ×2
        self.s1 = TVDeformConv2dPack(64, 64, k=3, stride=2, padding=1)

        # 中間：時間 3×1×1 + 空間 DCN2D（stride=1）
        self.t2 = nn.Conv3d(64, mid, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.n2 = N(mid)
        self.s2 = TVDeformConv2dPack(mid, mid, k=3, stride=1, padding=1)

        # 再一次空間 DCN2D 下採樣 ×2
        self.t3 = nn.Conv3d(mid, mid, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.n3 = N(mid)
        self.s3 = TVDeformConv2dPack(mid, mid, k=3, stride=2, padding=1)

        # 最後 1×1×1 投影
        self.proj = nn.Conv3d(mid, Cb, kernel_size=1, bias=False)
        self.n4   = N(Cb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.t1(x)))             # (B,64,T,H,W)
        x = apply_2d_over_time(x, self.s1)            # (B,64,T,H/2,W/2)

        x = self.act(self.n2(self.t2(x)))             # (B,mid,T,H/2,W/2)
        x = apply_2d_over_time(x, self.s2)            # (B,mid,T,H/2,W/2)

        x = self.act(self.n3(self.t3(x)))             # (B,mid,T,H/2,W/2)
        x = apply_2d_over_time(x, self.s3)            # (B,mid,T,H/4,W/4)

        x = self.act(self.n4(self.proj(x)))           # (B,Cb,T,H/4,W/4)
        return x
