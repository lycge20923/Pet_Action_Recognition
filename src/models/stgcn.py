import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.cli_args import ModelArguments, DataArguments
class Graph:
    def __init__(
        self,
        num_nodes: int,
        neighbor_base: list,
        coords: np.ndarray,
        hop_size: int = 1,
        normalization_strategy: str = 'symmetric'
    ):
        """
        Graph for ST-GCN with spatial configuration partitioning.

        Args:
            num_nodes: number of joints (V)
            neighbor_base: list of skeleton edges (pairs of indices)
            coords: numpy array of shape (V, dim) giving each joint's coordinates (static template)
            hop_size: must be 1 for spatial partitioning
            normalization_strategy: 'symmetric' | 'asymmetric_out_degree' | 'none'
        """
        assert hop_size == 1, "Spatial config partitioning requires hop_size=1"
        self.num_nodes = num_nodes
        self.neighbor_base = neighbor_base
        self.hop_size = hop_size
        self.normalization = normalization_strategy

        # Build self-links and neighbor links
        self.edges = [(i, i) for i in range(num_nodes)] + [(i, j) for i, j in neighbor_base]
        # Precompute hop distance
        self.hop_dis = self._get_hop_distance()

        # --- MODIFIED: compute skeleton centroid and distances ---
        self.coords = coords
        self.center = coords.mean(axis=0)
        self.dist2center = np.linalg.norm(coords - self.center, axis=1)

        # Build spatial-config adjacency
        self.A = self._get_adjacency_spatial()

    def _get_hop_distance(self) -> np.ndarray:
        # adjacency matrix
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        # powers for hops
        mats = [np.linalg.matrix_power(A, d) for d in range(self.hop_size + 1)]
        reach = (np.stack(mats) > 0)
        hop_dis = np.full((self.num_nodes, self.num_nodes), np.inf)
        for d in range(self.hop_size, -1, -1):
            hop_dis[reach[d]] = d
        return hop_dis

    def _normalize(self, A_part: np.ndarray) -> np.ndarray:
        if self.normalization == 'none':
            return A_part
        # symmetric
        deg = A_part.sum(axis=1)
        D_inv_sqrt = np.diag([d**(-0.5) if d > 0 else 0 for d in deg])
        if self.normalization == 'symmetric':
            return D_inv_sqrt @ A_part @ D_inv_sqrt
        # asymmetric_out_degree
        D_out_inv = np.diag([1.0/d if d>0 else 0 for d in deg])
        return A_part @ D_out_inv

    def _get_adjacency_spatial(self) -> np.ndarray:
        """
        Spatial configuration partitioning (3 subsets):
        - A0: self-loops
        - A1: centripetal edges (neighbor, moving closer to center)
        - A2: centrifugal edges (neighbor, moving farther from center)
        Returns:
            A: numpy array of shape (3, V, V)
        """
        A_spatial = np.zeros((3, self.num_nodes, self.num_nodes), dtype=float)
        # A0: self-loops
        for i in range(self.num_nodes):
            A_spatial[0, i, i] = 1
        # hop=1 neighbors split by dist2center
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.hop_dis[i, j] == 1:
                    if self.dist2center[j] < self.dist2center[i]:
                        A_spatial[1, i, j] = 1  # centripetal
                    else:
                        A_spatial[2, i, j] = 1  # centrifugal
        # normalize each partition
        for k in range(3):
            A_spatial[k] = self._normalize(A_spatial[k])
        return A_spatial

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s_kernel_size: int):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * s_kernel_size,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.conv(x)
        N, KC, T, V = x.shape
        x = x.view(N, self.s_kernel_size, KC // self.s_kernel_size, T, V)
        # einsum over spatial partitions
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGC_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        t_kernel_size: int,
        A_size: tuple,
        dropout: float = 0.5,
        dilations: list = None,
        add_gate_node: bool = False
    ):
        super().__init__()
        # spatial
        self.sgc = SpatialGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            s_kernel_size=A_size[0]
        )
        self.M = nn.Parameter(torch.ones(A_size))
        self.dilations = dilations
        
        # for No dilations
        if not self.dilations:
            self.tgc = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(t_kernel_size, 1),
                    stride=(stride, 1),
                    padding=((t_kernel_size - 1) // 2, 0)
                ),
                nn.BatchNorm2d(out_channels)
            )
        # for dilations
        else:
            self.t_branches = nn.ModuleList()
            for d in dilations:
                pad = d * ((t_kernel_size - 1) // 2)
                self.t_branches.append(
                    nn.Sequential(
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(t_kernel_size, 1),
                            stride=(stride, 1),
                            padding=(pad, 0),
                            dilation=(d, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    )
                )
        
        # residual
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU()
        
        # for node gate
        self.add_gate_node = add_gate_node
        if add_gate_node:
            self.node_gate = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        

    def forward(
            self,
            x: torch.Tensor,                # 当前层的特征（[N, C_feat, T_in, V]）
            orig_coords: torch.Tensor,      # 一直不变的“Bone keypoints (x_pix,y_pix)” [N, 2, T_orig, V]
            x_flow: torch.Tensor,           # dense 光流 [N, 2, T_orig, H, W]
            A: torch.Tensor
        ) -> torch.Tensor:
        """
        x:         [N, C_feat, T_in, V]
        orig_coords: [N, 2, T_orig, V]
        x_flow:    [N, 2, T_orig, H, W]
        A:         [3, V, V]
        """
        # 1) 残差 + 空间卷积 + 时序卷积 → 得到 out: [N, C_out, T_out, V]
        res = self.residual(x)
        out = self.sgc(x, A * self.M)       # [N, C_out, T_in, V]
        if not self.dilations:
            out = self.tgc(out)             # [N, C_out, T_out, V]
        else:
            branch_outs = [b(out) for b in self.t_branches]
            out = sum(branch_outs)          # [N, C_out, T_out, V]

        N, C_out, T_out, V = out.shape
        # —— 2) 这里要用 orig_coords，而不是 x[:,0,:,:] —— 
        if x_flow is not None and self.add_gate_node:
            # 2.1 从 orig_coords 中提取关键点像素坐标 (x_pix, y_pix)：
            #      orig_coords[:,0,t,i] = x_pixel^(t)_i
            #      orig_coords[:,1,t,i] = y_pixel^(t)_i
            x_pix = orig_coords[:, 0, :, :].clone()  # [N, T_orig, V]
            y_pix = orig_coords[:, 1, :, :].clone()  # [N, T_orig, V]

            # 2.2 把像素坐标归一化到 [-1, +1]
            H, W = x_flow.size(3), x_flow.size(4)    # e.g. 256,256
            x_norm = (x_pix / (W - 1)) * 2.0 - 1.0   # [N, T_orig, V]
            y_norm = (y_pix / (H - 1)) * 2.0 - 1.0   # [N, T_orig, V]

            # 2.3 拼成 grid: [N, T_orig, V, 2]
            grid = torch.stack((x_norm, y_norm), dim=-1)  # [N, T_orig, V, 2]

            # 2.4 把 dense 光流 [N,2,T_orig,H,W] → [N*T_orig, 2, H, W]
            orig_T = x_flow.size(2)
            flow_reshape = x_flow.permute(0, 2, 1, 3, 4).contiguous()  # [N, T_orig, 2, H, W]
            flow_reshape = flow_reshape.view(N * orig_T, 2, H, W)      # [N*T_orig, 2, H, W]

            # 2.5 grid reshape → [N*T_orig, 1, V, 2]
            grid_reshape = grid.contiguous().view(N * orig_T, V, 2)  # [N*T_orig, V, 2]
            grid_reshape = grid_reshape.unsqueeze(1)                 # [N*T_orig, 1, V, 2]

            # 2.6 用 grid_sample → [N*T_orig, 2, 1, V]
            flow_sampled = F.grid_sample(
                flow_reshape,      # [N*T_orig, 2, H, W]
                grid_reshape,      # [N*T_orig, 1, V, 2]
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # → [N*T_orig, 2, 1, V]

            # 2.7 reshape 回 [N, 2, T_orig, V]
            flow_sampled = flow_sampled.view(N, orig_T, 2, 1, V)    # [N, T_orig, 2, 1, V]
            flow_sampled = flow_sampled.squeeze(3)                  # [N, T_orig, 2, V]
            flow_at_kp   = flow_sampled.permute(0, 2, 1, 3)         # [N, 2, T_orig, V]

            # 2.8 如果当前层做了 time‐downsample (T_out < T_orig)，就对 [N,2,T_orig,V] 插值到 [N,2,T_out,V]
            if flow_at_kp.size(2) != T_out:
                flow_at_kp_ds = F.interpolate(
                    flow_at_kp.view(N, 2, flow_at_kp.size(2), V),  # [N,2,T_orig,V]
                    size=(T_out, V),
                    mode='bilinear',
                    align_corners=True
                )  # → [N,2,T_out,V]
            else:
                flow_at_kp_ds = flow_at_kp  # [N,2,T_out,V]

            # 2.9 Node Gate：permute→view→MLP→reshape→unsqueeze→repeat → [N,C_out,T_out,V]
            uv_flat     = flow_at_kp_ds.permute(0, 2, 3, 1).contiguous().view(N * T_out * V, 2)  # [N*T_out*V,2]
            alpha_flat  = self.node_gate(uv_flat)                                                 # [N*T_out*V,1]
            alpha       = alpha_flat.view(N, T_out, V).unsqueeze(1)                                # [N,1,T_out,V]
            alpha_expand = alpha.repeat(1, C_out, 1, 1)                                            # [N,C_out,T_out,V]

            out = out * alpha_expand  # [N,C_out,T_out,V] × [N,C_out,T_out,V]

        # 3) 残差 + ReLU
        out = out + res
        return self.relu(out)

class ST_GCN(nn.Module):
    def __init__(
        self,
        model_params:ModelArguments,
        data_params:DataArguments,
        coords: np.ndarray,
    ):
        super().__init__()
        self.add_learnable_node = model_params.add_learnable_node
        # Build graph with spatial config partitioning
        graph = Graph(
            num_nodes=data_params.num_nodes,
            neighbor_base=model_params.neighbor_base,
            coords=coords,
            hop_size=model_params.hop_size,
            normalization_strategy='symmetric'
        )
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        
        # add learnable node
        if self.add_learnable_node:
            V_ori = A.shape[1]
            V_after = V_ori + 1
            A_after = torch.zeros((A.shape[0], V_after, V_after), dtype=A.dtype)
            A_after[:, :V_ori, :V_ori] = A
            for k in range(data_params.num_nodes): # connect all
                A_after[:, k, V_ori] = 1.0
                A_after[:, V_ori, k] = 1.0
            A_after[:, V_ori, V_ori] = 1.0
            A = A_after
            
            # define a learnable node
            self.global_token = nn.Parameter(torch.randn(data_params.num_coords))
        
        self.register_buffer('A', A)
        A_size = A.size()  # (3, V, V)
        # BN over input channels * V
        self.bn = nn.BatchNorm1d(data_params.num_coords * A_size[1])
        # STGC blocks
        self.stgc1 = STGC_block(
            data_params.num_coords,
            model_params.intermediate_channels,
            stride=1,
            t_kernel_size=model_params.t_kernel_size,
            A_size=A_size,
            dilations=model_params.dilations,
            add_gate_node=model_params.add_gate_node
        )
        self.stgc2 = STGC_block(
            model_params.intermediate_channels,
            model_params.intermediate_channels,
            stride=1,
            t_kernel_size=model_params.t_kernel_size,
            A_size=A_size,
            dilations=model_params.dilations,
            add_gate_node=model_params.add_gate_node
        )
        self.stgc3 = STGC_block(
            model_params.intermediate_channels,
            model_params.final_channels,
            stride=2,
            t_kernel_size=model_params.t_kernel_size,
            A_size=A_size,
            dilations=model_params.dilations,
            add_gate_node=model_params.add_gate_node
        )
        self.stgc4 = STGC_block(
            model_params.final_channels,
            model_params.final_channels,
            stride=1,
            t_kernel_size=model_params.t_kernel_size,
            A_size=A_size,
            dilations=model_params.dilations,
            add_gate_node=model_params.add_gate_node
        )
        # Prediction head
        self.fc = nn.Conv2d(model_params.final_channels, model_params.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, x_flow:torch.Tensor=None) -> torch.Tensor:
        N, C, T, V = x.size()
        orig_coords = x
        # learnable_node
        if self.add_learnable_node:
            V = V + 1
            glb = self.global_token.unsqueeze(0).repeat(N, 1)
            glb = glb.unsqueeze(-1).repeat(1, 1, T)
            glb = glb.unsqueeze(-1)
            x = torch.cat([x, glb], dim=-1)
        
        # BN
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # STGC blocks
        x = self.stgc1(x, orig_coords, x_flow, self.A)
        x = self.stgc2(x, orig_coords, x_flow, self.A)
        x = self.stgc3(x, orig_coords, x_flow, self.A)
        x = self.stgc4(x, orig_coords, x_flow, self.A)
        # Global pooling + fc
        feat_4d = F.avg_pool2d(x, x.size()[2:])
        feat = feat_4d.view(N, -1)
        
        if self.add_learnable_node:
            x_time_avg = x.mean(dim=2, keepdim=True)
            global_vec = x_time_avg[:, :, 0, 17]
        
        logits = self.fc(feat_4d).view(N, -1)
        
        return feat, logits
