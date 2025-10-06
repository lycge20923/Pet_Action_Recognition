import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..utils.cli_args import ModelArguments, DataArguments
from .gcn.tools import *
from .gcn.graph import Graph
from ..utils.common import kp_diff_stats

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TDGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1): # r = 4 r = 16
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 2 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # This convolution (self.conv2) is redundant, 
        # but when you want to use the weight files we provide for action recognition, you have to uncomment it!
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1) 
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):

        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta  + x5 * gamma

        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(TDGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1)) # No I
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        self.beta = nn.Parameter(torch.tensor(0.5)) # 1.0 1.4 2.0
        self.gamma = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha, self.beta, self.gamma)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class TD_GCN(nn.Module):
    def __init__(self, num_nodes:int, neighbor_base:list, num_classes:list,
                is_frame_diff_stream:bool, gcn_include_blocks:list, in_channels:int=3, base_channels:int=64,
                num_person=1, drop_out=0, adaptive=True):
        super(TD_GCN, self).__init__()

        self.graph = Graph(num_nodes=num_nodes, neighbor_base=neighbor_base)

        A = self.graph.A # 3,25,25

        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.is_frame_diff_stream = is_frame_diff_stream
        self.in_channels = in_channels if not self.is_frame_diff_stream else 2
        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * self.num_nodes)
        
        block_confs = {
            1: dict(in_c=self.in_channels,      out_c=base_channels,      stride=1),
            2: dict(in_c=base_channels,  out_c=base_channels,      stride=1),
            3: dict(in_c=base_channels,  out_c=base_channels,      stride=1),
            4: dict(in_c=base_channels,  out_c=base_channels,      stride=1),
            5: dict(in_c=base_channels,  out_c=base_channels*2,    stride=2),
            6: dict(in_c=base_channels*2,out_c=base_channels*2,    stride=1),
            7: dict(in_c=base_channels*2,out_c=base_channels*2,    stride=1),
            8: dict(in_c=base_channels*2,out_c=base_channels*4,    stride=2),
            9: dict(in_c=base_channels*4,out_c=base_channels*4,    stride=1),
            10:dict(in_c=base_channels*4,out_c=base_channels*4,    stride=1),
        }
        
        self.blocks = nn.ModuleList()
        for idx in sorted(gcn_include_blocks):
            conf = block_confs[idx]
            self.blocks.append(
                TCN_GCN_unit(
                    conf['in_c'], conf['out_c'],
                    self.graph.A,
                    stride=conf['stride'],
                    adaptive=adaptive
                )
            )

        self.fc = nn.Linear(base_channels*4, self.num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_classes))
        bn_init(self.data_bn, 1)
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x        

    def forward(self, x, flow: torch.Tensor=None):
        keypoints = x.detach().clone()
        
        if self.is_frame_diff_stream:
            x = kp_diff_stats(keypoints)
        
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for block in self.blocks:
            x = block(x)
            
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        feats = x
        x = self.drop_out(x)
        logits = self.fc(x)
        
        return feats, logits
