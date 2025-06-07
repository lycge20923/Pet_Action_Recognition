import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalInteractionBlock(nn.Module):
    def __init__(self, skel_channels, flow_channels, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.temporal_align = nn.AdaptiveAvgPool2d((None, 17)) # T -> T/4, V -> 17
    def forward(self, x_skel, x_flow):
        '''
        Args:
            x_skel (torch.Tensor): ST-GCN 中間層特徵, shape [B, C_skel, T, V]
            x_flow (torch.Tensor): I3D 中間層特徵, shape [B, C_flow, T/2, H, W]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 融合後的兩個分支的特徵向量
        '''
        x_skel_residual = x_skel.clone()
        x_flow_residual = x_flow.clone()
        
        pool_kernel_size = (2, 1)
        x_skel_aligned = F.avg_pool2d(x_skel_residual, kernel_size=pool_kernel_size, stride=pool_kernel_size)
        
        
        B, C_skel, T_aligned, V = x_skel_aligned.shape
        B, C_flow, T_aligned_flow, H, W = x_flow_residual.shape
        L = H * W # I3D 的空間點數
        
        # [B, C, T, V] -> [B, T, V, C] -> [B*T, V, C]
        skel_seq = x_skel_aligned.permute(0, 2, 3, 1).contiguous().view(B * T_aligned, V, C_skel)
        # [B, C, T, H, W] -> [B, C, T, L] -> [B, T, L, C] -> [B*T, L, C]
        flow_seq = x_flow_residual.view(B, C_flow, T_aligned_flow, L).permute(0, 2, 3, 1).contiguous().view(B * T_aligned_flow, L, C_flow)
        
        # KQV
        sim_skel_flow = torch.matmul(skel_seq, flow_seq.transpose(-1, -2))
        attn_s_f = F.softmax(sim_skel_flow, dim=-1)
        skel_updated = torch.matmul(attn_s_f, flow_seq)
        
        sim_flow_skel = torch.matmul(flow_seq, skel_seq.transpose(-1, -2))
        attn_f_s = F.softmax(sim_flow_skel, dim=-1)
        flow_updated = torch.matmul(attn_f_s, skel_seq) # [B*T, L, C_skel]
        
        skel_updated = skel_updated.view(B, T_aligned, V, C_flow).permute(0, 3, 1, 2).contiguous()
        flow_updated = flow_updated.view(B, T_aligned_flow, L, C_skel).permute(0, 3, 1, 2).contiguous()
        
        skel_vec = F.adaptive_avg_pool2d(skel_updated, 1).squeeze()
        flow_vec = F.adaptive_avg_pool2d(flow_updated, 1).squeeze()
        
        return skel_vec, flow_vec
