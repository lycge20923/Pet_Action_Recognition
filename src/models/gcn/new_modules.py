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
        
        # to 0, 1
        eps   = 1e-6
        minv  = A_flow.amin(dim=(-2,-1), keepdim=True)  # (N,1,1)
        maxv  = A_flow.amax(dim=(-2,-1), keepdim=True)  # (N,1,1)
        # 2) 归一化到 [0,1]
        A_flow = (A_flow - minv) / (maxv - minv + eps)
        
        # print(A_flow.shape)
        # print(A_flow)

        return A_flow

class NodeAttention(nn.Module):
    """
    Node-level global self-attention: 对所有节点做自注意力计算，得到每个节点的重要性分数。
    """
    def __init__(self, in_channels, num_heads=1, temparture=0.01):
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
        self.temparture = temparture

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
        alpha = F.softmax(scores / self.temparture, dim=1)
        # alpha = scores
        return alpha

'''
class NodeAttention(nn.Module):
    """
    Node‐level attention，支持可变时间长度T。
    输入 x: (B, C, T, V)
    输出 alpha: (B, V)
    """
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 32,
                 num_heads: int = 1):
        """
        Args:
            in_channels: 输入通道数 C
            embed_dim:   GRU 和 MHA 的隐藏维度
            num_heads:   attention 的头数
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim   = embed_dim

        # 用 GRU 把 (C, T) 序列映射到 embed_dim，支持任意 T
        # 输入给 GRU 时，我们会把 batch = B*V, seq_len=T, feature=C
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )

        # MHA 接受 seq_len=V, batch=B, embed_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False,
            kdim=embed_dim,
            vdim=embed_dim
        )

        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, V)
        return alpha: (B, V)
        """
        B, C, T, V = x.shape
        # ---- 1) 用 GRU 编码时序 ----
        # 把每个 node 的 (C, T) 视为一个序列，batch 扩到 B*V
        # reshaped: (B*V, T, C)
        x_flat = x.permute(0, 3, 2, 1).reshape(B * V, T, C)
        # 转成 GRU 要求的 (seq_len=T, batch=B*V, feature=C)
        seq = x_flat.permute(1, 0, 2)  # (T, B*V, C)
        # 只取最后一个 time-step 的 hidden state: h_n (1, B*V, embed_dim)
        _, h_n = self.gru(seq)
        h = h_n[0]                     # (B*V, embed_dim)

        # reshape 回 (B, V, embed_dim)
        h_nodes = h.view(B, V, self.embed_dim)

        # ---- 2) MultiheadAttention ----
        # 转成 (seq_len=V, batch=B, embed_dim)
        h_seq = h_nodes.permute(1, 0, 2)  # (V, B, embed_dim)
        # 得到 (B, V, V) 的 attention 权重矩阵
        _, attn_weights = self.attn(
            h_seq, h_seq, h_seq,
            need_weights=True,
            average_attn_weights=True
        )  # attn_weights: (B, V, V)

        # ---- 3) 汇总每个 node 的分数 ----
        # 对所有 source node 平均 → (B, V)
        scores = attn_weights.mean(dim=2)

        # ---- 4) 温度化 softmax ----
        alpha = F.softmax(scores / (self.temperature + 1e-6), dim=1)

        return alpha
'''

'''
class NodeAttention(nn.Module):
    """
    Node‐level self‐attention，序列长度为节点数 V，embed_dim = C * T（将时间维度摺叠进特征维度）
    最终输出每个节点的注意力分数 alpha ∈ ℝ[B, V]。
    """
    def __init__(self, in_channels: int, num_frames=32, num_heads: int = 1):
        """
        Args:
            in_channels: 输入通道数 C
            num_frames:   时间步长 T
            num_heads:    多头数量
        """
        super().__init__()
        self.V = None  # 动态设置
        self.C = in_channels
        self.T = num_frames
        self.embed_dim = in_channels * num_frames
        
        # 可学习的温度参数，初始化为 1.0
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # MultiheadAttention：embed_dim=C*T，sequence length=V
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=False,            # (seq_len, batch, embed_dim)
            kdim=self.embed_dim,          # key/query dim
            vdim=self.embed_dim           # value dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (B, C, T, V)
        Returns:
            alpha: Tensor, shape (B, V)
        """
        B, C, T, V = x.shape
        
        # 1) 将时间维度摺叠进特征：(B, C, T, V) -> (B, V, C*T)
        x_node = x.permute(0, 3, 2, 1).reshape(B, V, C * T)  # (B, V, embed_dim)
        
        # 2) 转为 MultiheadAttention 要求的形状： (seq_len=V, batch=B, embed_dim)
        x_seq = x_node.permute(1, 0, 2)                      # (V, B, embed_dim)
        
        # 3) Self‐attention
        #    average_attn_weights=True -> attn_weights: (B, V, V)
        attn_output, attn_weights = self.attn(
            x_seq, x_seq, x_seq,
            need_weights=True,
            average_attn_weights=True
        )
        # attn_weights[b, i, j]: 从 node j 到 node i 的 attention 强度

        # 4) 汇总每个节点 i 的注意力得分：在所有源节点 j 上平均
        scores = attn_weights.mean(dim=2)  # (B, V)

        # 5) 通过可学习温度做 softmax
        alpha = F.softmax(scores / (self.temperature + 1e-6), dim=1)  # (B, V)
        return alpha
'''

'''
class NodeAttention(nn.Module):
    """
    Node-level global self-attention + Joint Positional Encoding
    输入 x: (B, C, T, V)
    输出 alpha: (B, V)
    """
    def __init__(self, in_channels, num_nodes=17, num_heads=1):
        super(NodeAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_nodes = num_nodes

        # 1) 可学习的节点位置编码: 每个关节点一个 C 维向量
        #    pos_emb.shape = (V, C)
        self.pos_emb = nn.Parameter(torch.zeros(num_nodes, in_channels))
        nn.init.xavier_uniform_(self.pos_emb)  

        # 2) 多头自注意力，embed_dim=in_channels, 序列长度=V
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False
        )

    def forward(self, x):
        B, C, T, V = x.shape
        # assert V == self.num_nodes, f"Expect V={self.num_nodes}, got {V}"

        # 1) 时间池化 -> (B, C, V)
        x_node = x.mean(dim=2)

        # 2) 加上节点位置编码
        #    pos_emb: (V, C) -> 转为 (C, V) 后广播到 (B, C, V)
        x_node = x_node + self.pos_emb.t().unsqueeze(0)

        # 3) 转成 (V, B, C)
        x_seq = x_node.permute(2, 0, 1)

        # 4) 全局自注意力
        #    attn_output: (V, B, C), attn_weights: (B, num_heads, V, V)
        _, attn_weights = self.attn(
            x_seq, x_seq, x_seq,
            need_weights=True,
            average_attn_weights=False
        )

        # 5) 融合多头 -> (B, V, V)
        avg_weights = attn_weights.mean(dim=1)

        # 6) 每个节点打分 -> (B, V)
        scores = avg_weights.mean(dim=1)

        # 7) 归一化为注意力分布
        alpha = F.softmax(scores, dim=1)
        return alpha
'''
'''
class NodeAttention(nn.Module):
    """
    混合局部 + 全局的节点级自注意力
    输入 x: Tensor(B, C, T, V)
    输出 alpha: Tensor(B, V)
    """
    def __init__(self, in_channels, A, num_nodes=17,  num_heads=4, mix_ratio=0.5):
        """
        in_channels: 特征维度 C
        num_nodes: 关节点数量 V
        A: 邻接矩阵，Tensor(V, V)，非零表示邻居
        num_heads: 多头数量
        mix_ratio: 局部 vs 全局注意力的融合权重 λ
        """
        super(NodeAttention, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.mix_ratio = mix_ratio

        # 1) 准备局部 attention 的 mask 矩阵 (V x V)
        #    PyTorch MHA 中 attn_mask 如果是 float，会在加权之前做 additive mask
        mask = torch.full((num_nodes, num_nodes), float('-inf'))
        # 把 A>0 的位置（邻居关系）以及对角线（self-attn）设为 0
        A = torch.from_numpy(A)
        neigh = (A > 0).float()
        mask = mask.masked_fill(neigh.bool(), 0.0)
        mask = mask.masked_fill(torch.eye(num_nodes).bool(), 0.0)
        # 注册为 buffer，随模型一起移动到 GPU/CPU
        self.register_buffer('local_attn_mask', mask)

        # 2) 定义同一个 MultiheadAttention，用于两次调用（全局/局部）
        #    保持 batch_first=False, 输入 (seq_len=V, batch=B, embed=C)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False
        )

    def forward(self, x):
        B, C, T, V = x.shape
        assert V == self.num_nodes, f"Expect V={self.num_nodes}, got {V}"

        # —— 1. 时间池化 —— (B, C, V)
        x_node = x.mean(dim=2)

        # —— 2. 重塑为 (V, B, C) —— 供 MHA 使用
        x_seq = x_node.permute(2, 0, 1)

        # —— 3. 全局 Attention —— 不带 mask
        # attn_weights_global: (B, num_heads, V, V)
        _, attn_weights_global = self.attn(
            x_seq, x_seq, x_seq,
            need_weights=True,
            average_attn_weights=False
        )

        # —— 4. 局部 Attention —— 带邻居 mask
        _, attn_weights_local = self.attn(
            x_seq, x_seq, x_seq,
            attn_mask=self.local_attn_mask,
            need_weights=True,
            average_attn_weights=False
        )

        # —— 5. 头平均 —— from (B, num_heads, V, V) to (B, V, V)
        avg_global = attn_weights_global.mean(dim=1)
        avg_local  = attn_weights_local .mean(dim=1)

        # —— 6. 局部 + 全局 融合 —— 按 mix_ratio λ 加权
        #     comb[i,j] = λ * local[i,j] + (1-λ) * global[i,j]
        comb = self.mix_ratio * avg_local + (1 - self.mix_ratio) * avg_global

        # —— 7. 每节点打分 & 归一化 —— 最终输出 (B, V)
        scores = comb.mean(dim=1)      # 在所有 query 位置上取平均
        alpha  = F.softmax(scores, dim=1)
        return alpha
'''