import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowAdjacency(nn.Module):
    # def __init__(self, num_nodes, patch_size=5, hidden_dim=16):
    def __init__(self, num_nodes, patch_size=5, hidden_dims=[16]):
        super().__init__()
        self.V = num_nodes
        self.ps = patch_size
        self.pad = patch_size // 2
        # self.mlp = nn.Sequential(
        #     nn.Linear(8, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 1)
        # )
        
        dims = [8] + hidden_dims + [1]
        layers = []
        for i in range(len(dims)-2):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, flow_seq, keypoints):
        # flow_seq: (N, 2, T, H, W), keypoints: (N,3, T, V)
        flow_seq = flow_seq.permute(0, 2, 1, 3, 4) # N, 2, T, H, W => N, T, 2, H, W
        keypoints = keypoints.permute(0, 2, 3, 1) # N, C, T, V => N, T, V, C
        
        N, T, _, H, W = flow_seq.shape # 32, 32, 2, 256, 256
        V = self.V
        ps, pad = self.ps, self.pad

        # 1) calculate coordinate of keypoints
        coords = keypoints[..., :2].clamp(0, 1)
        px = (coords[...,0] * (W-1)).long()  # (N, T, V)
        py = (coords[...,1] * (H-1)).long()  # (N ,T ,V)

        # 2) Flatten batch & time dim, and conduct unfold
        NT = N * T
        flow_flat = flow_seq.reshape(NT, 2, H, W)  # (N*T, 2, H, W)
        
        '''
        unfold: 
        [1 2 3                           [1 2 4 5
         4 5 6  + kernel dim = (2, 2) ->  2 3 5 6 
         7 8 9]                           4 5 7 8
                                          5 6 8 9]
        '''
        patches_all = F.unfold(
            flow_flat,
            kernel_size=(ps, ps),
            padding=pad
        )  # → (N * T, 2*ps*ps, H*W)

        # 3) extract every patch of the keypoints
        P  = ps * ps
        HW = H * W
        idx = (py * W + px).view(NT, V)            # (N * T, V)
        idx = idx.unsqueeze(1).expand(NT, 2*P, V)  # (NT,2P,V)
        patches = patches_all.gather(2, idx)       # (NT,2P,V)

        # 4) seperate u/v two channels，and calculate μ, σ
        patch_u = patches[:, :P, :]                # (NT, P, V)
        patch_v = patches[:, P:, :]                # (NT, P, V)

        mu_u    = patch_u.mean(dim=1)              # (NT, V)
        sigma_u = patch_u.std(dim=1, unbiased=False)  # (NT, V)
        mu_v    = patch_v.mean(dim=1)              # (NT, V)
        sigma_v = patch_v.std(dim=1, unbiased=False)  # (NT, V)

        # 5) concatenate feature
        feats = torch.stack([mu_u, mu_v, sigma_u, sigma_v], dim=2)  # (NT, V, 4)
        
        valid = (keypoints[..., 2].reshape(NT, V) > 0).to(feats.dtype)  # (NT, V)
        feats = feats * valid.unsqueeze(-1)  

        # 6) construct pairwise features (NT, V, V, 8)
        Fi = feats.unsqueeze(2).expand(NT, V, V, 4)
        Fj = feats.unsqueeze(1).expand(NT, V, V, 4)
        pairs = torch.cat([Fi, Fj], dim=3)  # (NT, V, V, 8)

        # 7) MLP
        scores = self.mlp(pairs.view(-1, 8)).view(NT, V, V)
        # 8) reverse to (N, V, V)
        A_flow = scores.view(N, T, V, V).mean(dim=1)  # (N, V, V)
        # A_flow = scores.view(N, T, V, V) # (N, T, V, V)

        return A_flow

class NodeAttention(nn.Module):
    """
    Node-level global self-attention: 对所有节点做自注意力计算，得到每个节点的重要性分数。
    """
    def __init__(self, in_channels, num_heads=1):
        super(NodeAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        # 多头自注意力，embed_dim=in_channels, 序列长度为节点数 V
        # batch_first=False 表示输入形状 (seq_len, batch, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T, V) -- 输入特征

        返回:
        alpha: Tensor of shape (B, V)   -- 每个节点的注意力权重
        """
        B, C, T, V = x.size()

        # 1) Temporal pooling: 对时间维度做均值池化，得到每个节点的通道特征 (B, C, V)
        x_node = x.mean(dim=2)

        # 2) 转换为 (seq_len=V, batch=B, embed_dim=C) 以供 MultiheadAttention
        #    x_node: (B, C, V) -> (V, B, C)
        x_seq = x_node.permute(2, 0, 1)

        # 3) 应用多头自注意力: Query=Key=Value=x_seq
        #    attn_output: (V, B, C), attn_weights: (B, num_heads, V, V)
        _, attn_weights = self.attn(
            x_seq, x_seq, x_seq,
            need_weights=True,
            average_attn_weights=False
        )
        
        # 4) 在头维度上取平均: (B, V, V)
        avg_weights = attn_weights.mean(dim=1)

        # 5) 计算每个节点的全局重要性: 对所有查询位置求平均，得到 (B, V)
        scores = avg_weights.mean(dim=1)

        # 6) 归一化成注意力分布
        alpha = F.softmax(scores, dim=1)
        return alpha
