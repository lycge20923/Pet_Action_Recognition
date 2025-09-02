import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowAdjacencyModule(nn.Module):
    # def __init__(self, num_nodes, patch_size=5, hidden_dim=16):
    def __init__(self, num_nodes, flow_statistic_dim, patch_size=5, hidden_dims=[16]):
        super().__init__()
        start_dim = flow_statistic_dim * 4
        self.V = num_nodes
        self.ps = patch_size
        self.pad = patch_size // 2
        
        dims = [start_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims)-2):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, flow_map):
        """
        Args:
            flow_map: (B, T, V, C)  # C: (mean_u, mean_v, ...) per joint per frame
        Returns:
            A_flow: (B, V, V)  # Adjacency Matrix
        """
        B, T, V, C = flow_map.shape
        # assert C == 4, f"flow_map last dim must be 4, got {C}"

        # 1) Build pairwise joint features (8-D)
        Fi = flow_map.unsqueeze(3).expand(B, T, V, V, C)  # (B,T,V,V,C)
        Fj = flow_map.unsqueeze(2).expand(B, T, V, V, C)  # (B,T,V,V,C)
        pair_8d = torch.cat([Fi, Fj], dim=-1)             # (B,T,V,V,2*C)

        # 2) Feed into MLP
        scores = self.mlp(pair_8d.reshape(B * T * V * V, 2*C))  # (B*T*V*V, 1)
        scores = scores.view(B, T, V, V)                      # (B,T,V,V)

        # 3) Temporal aggregation (same as your original: average over T)
        A_flow = scores.mean(dim=1)                           # (B,V,V)

        # 4) Per-sample min-max normalization to [0,1], 
        # avoiding division by zero for constant matrices
        eps  = 1e-6
        minv = A_flow.amin(dim=(-2, -1), keepdim=True)       # (B,1,1)
        maxv = A_flow.amax(dim=(-2, -1), keepdim=True)       # (B,1,1)
        denom = (maxv - minv)
        safe_denom = torch.where(denom < eps, torch.ones_like(denom), denom)
        A_flow = (A_flow - minv) / safe_denom

        return A_flow