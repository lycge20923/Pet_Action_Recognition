import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools import *
from .graph import Graph
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
            
            
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(TemporalConv, self).__init__()
        
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0), 
                              stride=(stride, 1), 
                              dilation=(dilation, 1), 
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
    
class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
    
class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1, pad=True):
        super().__init__()
        
        self.window_size = window_size
        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2 if pad else 0
        
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, -1, V)
        x = x.transpose(2, 3).contiguous()
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                pos_list.append(j_id)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.time_len * self.joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe.to(x.dtype)[:, :, :x.size(2)]
        return x   
    
    
class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(ST_GC, self).__init__()
        
        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)
        
        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        weights = self.A.to(v.dtype)
        
        x = torch.einsum('hvu,nhctu->nctv', weights, v)
        x = self.bn(x)
        return x
    

class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1):
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale
        
        rel_channels = in_channels // 8 if in_channels != 3 else 8
        
        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x, A=None, alpha=1):
        N, C, T, V = x.size()
        res = x
        q, k, v = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x).view(N, self.num_scale, self.Nh, -1, T, V)
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)        
        weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)
        x = torch.einsum('ngacvu, ngactu->ngctv', weights, v).contiguous().view(N, -1, T, V)
        x = self.bn(x)
        return x
    
    
class DeSGC(nn.Module):
    '''
    Note: This module is not included in the open-source release due to subsequent research and development. 
    It will be made available in future updates after the completion of related studies.
    '''
    def __init__(self, in_channels, out_channels, A, k, num_scale=4, num_frame=64, num_joint=25):
        super(DeSGC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(A)
        
        self.num_scale = num_scale
        self.k = k
        self.delta = 10
        
        rel_channels = in_channels // 8 if in_channels != 3 else 8
        self.factor = rel_channels // num_scale
        
        self.pe = PositionalEncoding(in_channels, num_joint, num_frame)
        self.conv = PointWiseTCN(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.convQK = nn.Conv2d(in_channels, 2 * rel_channels * self.Nh, 1, groups=num_scale)
        self.convW = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1, 1, self.Nh, 1, 1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        v = self.relu(self.conv(x)).view(N, self.num_scale, self.Nh, -1, T, V)
        dtype, device = v.dtype, v.device
        
        # calculate score
        # ...
        
        # calculate weight
        # ...
        
        # convert to onehot
        # ...
        
        # sampling & aggregation
        # ...
        
        return x
    

class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0, dilation=1, 
                 num_scale=1, num_frame=64):
        super(DeTGC, self).__init__()
        
        self.ks, self.stride, self.dilation = kernel_size, stride, dilation
        self.T = num_frame
        self.num_scale = num_scale
        
        self.eta = eta
        ref = (self.ks + (self.ks-1) * (self.dilation-1) - 1) // 2
        tr = torch.linspace(-ref, ref, self.eta)
        self.tr = nn.Parameter(tr)

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype 
        
        #learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        tr = self.tr.to(dtype)
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), tr.view(1, self.eta, 1) 
        t = t0 + tr 
        t = t.view(1, 1, -1, 1) 
        
        #indexing
        tdn = t.detach().floor()
        tup = tdn + 1
        index1, index2 = torch.clamp(tdn, 0, self.T-1).long(), torch.clamp(tup, 0, self.T-1).long()
        index1, index2 = index1.expand(N, C, -1, V), index2.expand(N, C, -1, V)
        
        #sampling
        alpha = tup - t
        x1, x2 = x.gather(-2, index=index1), x.gather(-2, index=index2) 
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
    def __init__(self, in_channels, out_channels, A, k, eta, kernel_size=5, stride=1, dilations=2, 
                 num_frame=64, num_joint=25, residual=True):
        super(Basic_Block, self).__init__()
        
        num_scale = 4
        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1
        
        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            # self.gcn = DeSGC(in_channels, 
            #                  out_channels, 
            #                  A, 
            #                  k, 
            #                  self.num_scale, 
            #                  num_frame=num_frame, 
            #                  num_joint=num_joint)
            self.gcn = CTR_GC(in_channels, 
                              out_channels, 
                              A, 
                              self.num_scale)
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
        x = self.relu(x + self.residual2(res))
        return x


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class DeGCN(nn.Sequential):
    def __init__(self, block_args, A, k, eta):
        super(DeGCN, self).__init__()
        for i, [in_channels, out_channels, stride, residual, num_frame, num_joint] in enumerate(block_args):
            self.add_module(
                f'block-{i}_tcngcn',
                Basic_Block(
                    in_channels,
                    out_channels,
                    A,
                    k,
                    eta,
                    stride=stride,
                    num_frame=num_frame,
                    num_joint=num_joint,
                    residual=residual
                )
            )

class DE_GCN(nn.Module):
    def __init__(
        self, model_params:ModelArguments, data_params:DataArguments, num_person=1, k=8, eta=4, drop_out=0,
    ):
        super(DE_GCN, self).__init__()
        self.graph = Graph(model_params=model_params, data_params=data_params)
        A = self.graph.A  # (3, V, V)
        
        self.data_bn = nn.BatchNorm1d(num_person * data_params.num_coords * data_params.num_nodes)
        bn_init(self.data_bn, 1)
        
        block_list = [
            [model_params.in_channels, model_params.base_channels, 1, False, data_params.num_samples, data_params.num_nodes],
            [model_params.base_channels, model_params.base_channels, 1, True, data_params.num_samples, data_params.num_nodes],
            [model_params.base_channels, model_params.base_channels, 1, True, data_params.num_samples, data_params.num_nodes],
            [model_params.base_channels, model_params.base_channels, 1, True, data_params.num_samples, data_params.num_nodes],
            [model_params.base_channels, model_params.base_channels * 2, 2, True, data_params.num_samples, data_params.num_nodes],
            [model_params.base_channels * 2, model_params.base_channels * 2, 1, True, data_params.num_samples // 2, data_params.num_nodes],
            [model_params.base_channels * 2, model_params.base_channels * 2, 1, True, data_params.num_samples // 2, data_params.num_nodes],
            [model_params.base_channels * 2, model_params.base_channels * 4, 2, True, data_params.num_samples // 2, data_params.num_nodes],
            [model_params.base_channels * 4, model_params.base_channels * 4, 1, True, data_params.num_samples // 4, data_params.num_nodes],
            [model_params.base_channels * 4, model_params.base_channels * 4, 1, True, data_params.num_samples // 4, data_params.num_nodes]
        ]
        self.blockargs = [block_list[int(i - 1)] for i in model_params.gcn_include_blocks]

        self.num_stream = model_params.degcn_num_streams
        self.streams = nn.ModuleList([
            DeGCN(self.blockargs, A, k, eta) for _ in range(self.num_stream)
        ])
        self.fc = nn.ModuleList([
            nn.Linear(model_params.base_channels*4, model_params.num_classes) for _ in range(self.num_stream)
        ])
        for fc in self.fc:
            nn.init.normal_(fc.weight, 0, math.sqrt(2. / model_params.num_classes))
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        # Input reshape and batchnorm
        if x.dim() == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1)
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x_ = x

        # Multi-stream processing and pooling
        feats, logits_list = [], []
        for stream, fc in zip(self.streams, self.fc):
            x_s = stream(x_)                   # (N*M, C', T', V')
            c_new = x_s.size(1)
            # Pool over time & joints
            x_pooled = x_s.view(N, M, c_new, -1).mean(-1).mean(1)  # (N, C')
            x_d = self.drop_out(x_pooled)
            logits = fc(x_d)                   # (N, num_class)
            feats.append(x_pooled)
            logits_list.append(logits)

        # Average across streams
        feat = torch.stack(feats, dim=0).mean(dim=0)         # (N, C')
        logits = torch.stack(logits_list, dim=0).mean(dim=0) # (N, num_class)
        return feat, logits

if __name__ == "__main__":
    model = DE_GCN()