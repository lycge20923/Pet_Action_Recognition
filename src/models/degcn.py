import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn.tools import *
from .gcn.graph import Graph
from ..utils.cli_args import ModelArguments, DataArguments

LEAKY_ALPHA = 0.1
def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_ALPHA, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A): # 3, 64, (3, 17, 17)
        super(ST_GC, self).__init__()
        
        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A) # make sure the it could be updated
        self.Nh = A.size(0) # number of heads, 3
        
        '''
        Input: (N, input_channles, T, V)
        Output: (N, output_channels*Nf, T, V)
        '''
        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1) 
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size() # 32, 3, 32, 17
        v = self.conv(x).view(N, self.Nh, -1, T, V) # N, out_channels*Nf, T, 17 -> N, Nf, output_channels, T, V) 
        weights = self.A.to(v.dtype)
        
        '''
        h: 3, v, u:17
        n: N, h:hf, c:output_channels=64, t: 32, u: 17
        function: 
            1. for each head(total 3), weights[h], times and sum v[:, h, :, :, :] 
            2. Sum all results in the dimension of head, and delete h, u
        '''
        x = torch.einsum('hvu,nhctu->nctv', weights, v) # N, output_channels, T, V = 32, 64, 32, 17
        x = self.bn(x) # same dim
        return x # for output alignment 
    

class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1, start_ch=3): # 64*a, 64*b, (3, 17, 17), 4, 17
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale
        
        '''
        in_channels would never be 3 since it only exists in first block 
        '''
        rel_channels = in_channels // 8 if in_channels != start_ch else 8 # 64*a//8 
        
        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        
    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        
        # q, k: (N, 64*a, T, V) => (N, (64*a//8)*nh, T, V) => (N, (64*a//8)*nh, V)
        # v: (N, 64*b, T, V) => (N, 64*b*nh, T, V) => (N, 4, nh, 16*b, T, V)
        q, k, v = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x).view(N, self.num_scale, self.Nh, -1, T, V) 
        
        '''
        Attention between Nodes,  q: (N, (64*a//8)*nh, V) -> (N, (64*a//8)*nh, V, 1), k: (N, (64*a//8)*nh, V) -> (N, (64*a//8)*nh, 1, V)
        q - k: broacasting, q, k map to  (N, (64*a//8)*nh, V, V)
        tanh: each element maps to (-1, 1)
        weights: (N, ) -> (N, 4, nh, , V, V)
        '''
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)    
        dtype = weights.dtype
        device = weights.device
        
        A_static = self.A.view(1,1,self.Nh,1,V,V).to(dtype).to(device)
        weights = weights * self.alpha.to(dtype) + A_static
        x = torch.einsum('ngacvu, ngactu->ngctv', weights, v).contiguous().view(N, -1, T, V)
        x = self.bn(x)
        return x
    
class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0, dilation=1, 
                 num_scale=1, num_frame=64):
        super(DeTGC, self).__init__()
        
        self.ks, self.stride, self.dilation = kernel_size, stride, dilation # 5, 1or2, 2or2
        self.T = num_frame # 8
        self.num_scale = num_scale # 4
        
        self.eta = eta # 4
        ref = (self.ks + (self.ks-1) * (self.dilation-1) - 1) // 2
        tr = torch.linspace(-ref, ref, self.eta) # sampling from -ref to ref, total eta points
        self.tr = nn.Parameter(tr) # make it trainable

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        '''
        C: 16, 32, 64
        T: 32, 32, 16or8
        '''
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype 
        
        #learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device) # 0~T sampling self.stride, each length = Tout, t0 dim = [T/self.stride]
        tr = self.tr.to(dtype) # dim = [self.eta]
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), tr.view(1, self.eta, 1) # [1, 1, T/self.stride] => [1, self.eta, T/self.stride], [1, self.eta, 1] 
        t = t0 + tr # tr would be duplicated to match the size of t0, [1, self.eta, T/self.stride]
        t = t.view(1, 1, -1, 1) # [1, self.eta, T/self.stride] -> [1, self.eta * T/self.stride, 1, 1]
        
        #indexing
        tdn = t.detach().floor() # detatch(): no gradient update, [1, self.eta * T/self.stride, 1, 1]; .floor(): 1.8 -> 1.0
        tup = tdn + 1
        index1, index2 = torch.clamp(tdn, 0, self.T-1).long(), torch.clamp(tup, 0, self.T-1).long()
        index1, index2 = index1.expand(N, C, -1, V), index2.expand(N, C, -1, V) # [1, self.eta * T/self.stride, 1, 1] => [N, C, self.eta * T/self.stride, V]
        
        #sampling
        alpha = tup - t # [1, self.eta * T/self.stride, 1, 1]
        x1, x2 = x.gather(-2, index=index1), x.gather(-2, index=index2) # extract the related x'elements in indexes in index1, index2
        x = x1 * alpha + x2 * (1 - alpha)
        x = x.view(N, C, self.eta, Tout, V)
        
        #conv
        x = self.conv_out(x).squeeze(2)
        return x


class MultiScale_TemporalModeling(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=5, stride=1, dilations=1, 
                 num_scale=1, num_frame=64):
        super(MultiScale_TemporalModeling, self).__init__()
        
        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1

        self.tcn1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, 
                  scale_channels, 
                  eta,
                  kernel_size=5, 
                  stride=stride, 
                  dilation=1, 
                  num_scale=num_scale, 
                  num_frame=num_frame)
        )
        
        self.tcn2 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, 
                  scale_channels, 
                  eta,
                  kernel_size=5, 
                  stride=stride, 
                  dilation=2, 
                  num_scale=num_scale, 
                  num_frame=num_frame)
        )
        
        self.maxpool3x1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(scale_channels) 
        )
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        x = torch.cat([self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1)
        return x
    
class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, eta, stride=1, num_frame=64, residual=True, start_ch=3):
        super(Basic_Block, self).__init__()
        
        num_scale = 4
        # scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels != start_ch else 1
        
        if in_channels == start_ch:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            self.gcn = CTR_GC(in_channels, 
                              out_channels, 
                              A, 
                              self.num_scale,
                              start_ch=start_ch)
        self.tcn = MultiScale_TemporalModeling(out_channels, 
                                               out_channels, 
                                               eta,
                                               stride=stride, 
                                               num_scale=num_scale, 
                                               num_frame=num_frame) 
        
        if in_channels != out_channels:
            self.residual1 = PointWiseTCN(in_channels, out_channels, groups=self.num_scale)
        else:
            self.residual1 = lambda x: x
            
        if not residual:
            self.residual2 = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual2 = lambda x: x
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels, stride=stride, groups=self.num_scale)
        
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())
    
    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.relu(x + self.residual1(res))
        x = self.tcn(x)
        x = x + self.residual2(res)            
        x = self.relu(x)
        
        return x

class DeGCN(nn.Module):
    def __init__(self, block_args, A, eta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Basic_Block(
                in_ch, out_ch, A, eta,
                stride=stride,
                residual=residual,
                num_frame=num_frame,
                start_ch = start_ch,
            )
            for in_ch, out_ch, stride, residual, num_frame, start_ch in block_args
        ])

    def forward(self, x):
        # x: (N*M, C, T, V)
        for block in self.blocks:
            x = block(x)
        return x

class DE_GCN(nn.Module):
    def __init__(
        self, model_params:ModelArguments, data_params:DataArguments, num_person=1, k=8, eta=4, drop_out=0,
    ):
        super(DE_GCN, self).__init__()
        self.graph = Graph(num_nodes=data_params.num_nodes, neighbor_base=model_params.neighbor_base)
        self.A = self.graph.A  # (3, V, V)
        self.k = k
        self.eta = eta
        self.num_person = num_person
        self.model_params = model_params
        self.data_params = data_params
        self.flow_block_size = model_params.flow_block_size
        self.use_frame_diff = model_params.use_frame_diff
        
        self.data_bn, self.streams, self.fc = self._init_streams(model_params.in_channels)
        
        self.add_flow_stream = model_params.add_flow_stream
        if self.add_flow_stream:
            in_channels_flow = 2 * model_params.flow_block_size * model_params.flow_block_size
            self.data_bn_flow, self.streams_flow, self.fc_flow = self._init_streams(in_channels_flow)
        
        if self.use_frame_diff:
            assert not self.add_flow_stream
            in_channels = 2
            self.data_bn_flow, self.streams_flow, self.fc_flow = self._init_streams(in_channels)
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x  
    
    def _init_streams(self, in_channels):
        data_bn = nn.BatchNorm1d(self.num_person * in_channels * self.data_params.num_nodes)
        bn_init(data_bn, 1)
        
        block_list = [
            [in_channels, self.model_params.base_channels, 1, False, self.data_params.num_samples, in_channels],
            [self.model_params.base_channels, self.model_params.base_channels, 1, True, self.data_params.num_samples, in_channels],
            [self.model_params.base_channels, self.model_params.base_channels, 1, True, self.data_params.num_samples, in_channels],
            [self.model_params.base_channels, self.model_params.base_channels, 1, True, self.data_params.num_samples, in_channels],
            [self.model_params.base_channels, self.model_params.base_channels * 2, 2, True, self.data_params.num_samples, in_channels],
            [self.model_params.base_channels * 2, self.model_params.base_channels * 2, 1, True, self.data_params.num_samples // 2, in_channels],
            [self.model_params.base_channels * 2, self.model_params.base_channels * 2, 1, True, self.data_params.num_samples // 2, in_channels],
            [self.model_params.base_channels * 2, self.model_params.base_channels * 4, 2, True, self.data_params.num_samples // 2, in_channels],
            [self.model_params.base_channels * 4, self.model_params.base_channels * 4, 1, True, self.data_params.num_samples // 4, in_channels],
            [self.model_params.base_channels * 4, self.model_params.base_channels * 4, 1, True, self.data_params.num_samples // 4, in_channels]
        ]
        blockargs = [block_list[int(i - 1)] for i in self.model_params.gcn_include_blocks]
        streams = nn.ModuleList([
            DeGCN(blockargs, self.A, self.eta)for _ in range(self.model_params.degcn_num_streams)]
        )
        
        fc = nn.ModuleList([
            nn.Linear(self.model_params.base_channels*4, self.model_params.num_classes) for _ in range(self.model_params.degcn_num_streams)
        ])
        for fc_ in fc:
            nn.init.normal_(fc_.weight, 0, math.sqrt(2. / self.model_params.num_classes))
        
        return data_bn, streams, fc
    
    
    def _kp_diff_stats(self, keypoints: torch.Tensor):
        """
        Args:
            keypoints: (B, 3, T, J) – normalized (x, y, conf)
        Returns:
            flow_map: (B, 2, T, J)
        """
        # 只取 x,y；形狀 (B, 2, T, J)
        xy = keypoints[:, :2, ...]
        # 逐幀差分：Δx = x_t - x_{t-1}, Δy 同理；在 t=0 補 0
        dxy = xy[:, :, 1:, :] - xy[:, :, :-1, :]                     # (B, 2, T-1, J)
        zero = torch.zeros_like(dxy[:, :, :1, :])                    # (B, 2, 1,   J)
        dxy = torch.cat([zero, dxy], dim=2)                          # (B, 2, T,   J)
        return dxy
    
    def local_flow_stats(self, keypoints: torch.Tensor,
                        optical_flows: torch.Tensor,
                        flow_block_size: int = 3):
        """
        Args:
            keypoints:      (B, 3, T, J) – normalized (x, y, conf), but x/y may be outside [0,1].
                            If [x,y,conf] == [0,0,0], treat as missing and skip.
            optical_flows:  (B, 2, T, H, W) – flow u/v.
            flow_block_size:     int – patch side length.

        Returns:
            features: (B, T, J, 4) – (mean_u, mean_v, std_u, std_v).
        """
        
        B, _, T, J = keypoints.shape
        _, _, _, H, W = optical_flows.shape
        
        device = optical_flows.device
        dtype  = optical_flows.dtype

        ps   = flow_block_size
        half = flow_block_size // 2  # 與你給的名稱一致
        r    = half
        P    = ps * ps

        # 1) 取出 x,y,conf 並轉成像素座標（從 normalized → pixels）
        #    若你的 x,y 原本已是像素座標，可把兩行縮放改成 x_pix = x, y_pix = y。
        x   = keypoints[:, 0, ...].to(dtype=dtype, device=device)     # (B, T, J)
        y   = keypoints[:, 1, ...].to(dtype=dtype, device=device)     # (B, T, J)
        conf= keypoints[:, 2, ...].to(dtype=dtype, device=device)     # (B, T, J)

        x_pix = x * (W - 1)
        y_pix = y * (H - 1)

        # 2) 取整數中心（模擬原本硬切）
        cx = x_pix.round()                                         # (B, T, J)
        cy = y_pix.round()                                         # (B, T, J)

        # 3) 建 ps×ps 偏移網格（像素座標）
        base_y, base_x = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij"
        )                                                          # (ps, ps)
        base = torch.stack([base_x, base_y], dim=-1)               # (ps, ps, 2)

        # 4) 組成每個關節的像素座標網格 (B,T,J,ps,ps,2)
        grid_pix = base.view(1, 1, 1, ps, ps, 2).expand(B, T, J, ps, ps, 2).clone()
        grid_pix[..., 0] = grid_pix[..., 0] + cx.unsqueeze(-1).unsqueeze(-1)  # x
        grid_pix[..., 1] = grid_pix[..., 1] + cy.unsqueeze(-1).unsqueeze(-1)  # y

        # 5) 像素座標 → [-1,1]（給 grid_sample）
        grid = grid_pix.clone()
        grid[..., 0] = 2.0 * (grid_pix[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid_pix[..., 1] / (H - 1)) - 1.0
        grid = grid.view(B * T * J, ps, ps, 2)                     # (BTJ, ps, ps, 2)

        # 6) 準備 flow，展平成 (BTJ, 2, H, W)
        #    optical_flows: (B, 2, T, H, W) → (B, T, 2, H, W) → (B*T, 2, H, W)
        flow_bt = optical_flows.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 2, H, W)
        flow_rep = flow_bt.unsqueeze(1).expand(B * T, J, 2, H, W).reshape(B * T * J, 2, H, W)

        # 7) 取樣（nearest + border 貼近原邏輯；若要亞像素平滑可改 mode='bilinear'）
        patches = F.grid_sample(
            flow_rep, grid,
            mode='nearest',
            padding_mode='border',
            align_corners=False
        )  # (BTJ, 2, ps, ps)
        
        # 7) 整理成 (B, 2*ps*ps, T, J)
        patches = patches.view(B, T, J, 2, ps, ps)          # (B,T,J,2,ps,ps)
        patches = patches.permute(0, 3, 4, 5, 1, 2).contiguous()  # (B,2,ps,ps,T,J)
        patches = patches.view(B, 2 * ps * ps, T, J)        # (B, C_flow, T, J)
        
        # 8) 遮罩無效點（conf=0 或 [0,0,0]）
        with torch.no_grad():
            x0 = (keypoints[:, 0] == 0)
            y0 = (keypoints[:, 1] == 0)
            c0 = (keypoints[:, 2] == 0)
            zero_xyc = (x0 & y0 & c0)                       # (B,T,J)
            valid = ((keypoints[:, 2] > 0) & (~zero_xyc)).float()  # (B,T,J)
        patches = patches * valid.unsqueeze(1)              # broadcast 到 C_flow

        return patches  # (B, 2*ps*ps, T, J)

    def forward(self, x, optical_flows=None):
        keypoints = x.detach().clone()
        
        # Multi-stream processing and pooling
        feats, logits_list = [], []
        
        # trial
        if self.add_flow_stream:
            y = self.local_flow_stats(keypoints, optical_flows, self.flow_block_size)
        
        if self.use_frame_diff:
            y = self._kp_diff_stats(keypoints)
        
        # stream of coords
        x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x_ = x
        for i, (stream, fc) in enumerate(zip(self.streams, self.fc)):
            x_s = stream(x_) # (N*M, C', T', V')
            c_new = x_s.size(1)
            x_pooled = x_s.reshape(N, M, c_new, -1).mean(-1).mean(1)  # (N, C')
            x_d = self.drop_out(x_pooled)
            logits = fc(x_d)                   # (N, num_class)
            feats.append(x_pooled)
            logits_list.append(logits)

        # stream of flow
        if self.add_flow_stream or self.use_frame_diff:
            y = y.unsqueeze(-1)
            N, C, T, V, M = y.size()
            y = y.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            y = self.data_bn_flow(y)
            y = y.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
            y_ = y
            for i, (stream, fc) in enumerate(zip(self.streams_flow, self.fc_flow)):
                y_s = stream(y_) # (N*M, C', T', V')
                c_new = y_s.size(1)
                y_pooled = y_s.reshape(N, M, c_new, -1).mean(-1).mean(1)  # (N, C')
                y_d = self.drop_out(y_pooled)
                logits = fc(y_d)                   # (N, num_class)
                feats.append(y_pooled)
                logits_list.append(logits)
        
        # Average across streams
        feat = torch.stack(feats, dim=0).mean(dim=0)         # (N, C')
        logits = torch.stack(logits_list, dim=0).mean(dim=0) # (N, num_class)
        return feat, logits

if __name__ == "__main__":
    pass