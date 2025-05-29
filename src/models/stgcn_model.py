import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.sgc(x, A * self.M)
        
        if not self.dilations:
            out = self.tgc(out)
        else:
            branch_outs = [branch(out) for branch in self.t_branches]
            out = sum(branch_outs)
        out = out + res
        return self.relu(out)

class ST_GCN(nn.Module):
    def __init__(
        self,
        params,
        data_params,
        coords: np.ndarray,
    ):
        super().__init__()
        # Build graph with spatial config partitioning
        graph = Graph(
            num_nodes=data_params.num_nodes,
            neighbor_base=params.neighbor_base,
            coords=coords,
            hop_size=1,
            normalization_strategy='symmetric'
        )
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()  # (3, V, V)
        # BN over input channels * V
        self.bn = nn.BatchNorm1d(params.in_channels * A_size[1])
        # STGC blocks
        self.stgc1 = STGC_block(
            params.in_channels,
            params.intermediate_channels,
            stride=1,
            t_kernel_size=params.t_kernel_size,
            A_size=A_size,
            dilations=params.dilations
        )
        self.stgc2 = STGC_block(
            params.intermediate_channels,
            params.intermediate_channels,
            stride=1,
            t_kernel_size=params.t_kernel_size,
            A_size=A_size,
            dilations=params.dilations
        )
        self.stgc3 = STGC_block(
            params.intermediate_channels,
            params.final_channels,
            stride=2,
            t_kernel_size=params.t_kernel_size,
            A_size=A_size,
            dilations=params.dilations
        )
        self.stgc4 = STGC_block(
            params.final_channels,
            params.final_channels,
            stride=1,
            t_kernel_size=params.t_kernel_size,
            A_size=A_size,
            dilations=params.dilations
        )
        # Prediction head
        self.fc = nn.Conv2d(params.final_channels, params.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        # BN
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # STGC blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        # Global pooling + fc
        feat_4d = F.avg_pool2d(x, x.size()[2:])
        feat = feat_4d.view(N, -1)
        logits = self.fc(feat_4d).view(N, -1)
        return feat, logits

class STGCN_MultiHead(nn.Module):
    """
    ST-GCN backbone with dual heads:
      - proj_head: for contrastive embedding (normalized)
      - class_head: for classification logits
    """
    def __init__(
        self,
        params,
        data_params,
        coords: np.ndarray,
        embed_dim: int = 128
    ):
        super().__init__()
        # Backbone unchanged
        self.backbone = ST_GCN(
            params=params,
            data_params=data_params,
            coords=coords,
        )
        final_ch = params.final_channels
        num_classes = params.num_classes  
        # Projection head for contrastive loss
        self.proj_head  = nn.Linear(final_ch, embed_dim)
        # Classification head for cross-entropy
        self.class_head = nn.Linear(final_ch, num_classes)

    def forward(self, x: torch.Tensor):
        # feat: (N, final_channels)
        feat, _ = self.backbone(x)
        # emb: normalized embedding
        emb = F.normalize(self.proj_head(feat), dim=1)  # MODIFIED: L2-normalize
        # logits: classification
        logits = self.class_head(feat)
        return emb, logits
