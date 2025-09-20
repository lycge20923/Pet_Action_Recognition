import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .gcn.tools import *
from ..utils.cli_args import ModelArguments, DataArguments
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
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
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


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 2 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
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
            self.convs.append(CTRGC(in_channels, out_channels))

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
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

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
            z = self.convs[i](x, A[i], self.alpha)
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


class CTR_GCN(nn.Module):
    # def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
    #              drop_out=0, adaptive=True):
    def __init__(self, model_params:ModelArguments, data_params:DataArguments, num_person=1, adaptive=True, drop_out=0):
        super(CTR_GCN, self).__init__()

        self.graph = Graph(num_nodes=data_params.num_nodes, neighbor_base=model_params.neighbor_base)
        A = self.graph.A

        self.num_class = model_params.num_classes
        self.num_point = data_params.num_nodes
        self.in_channels = model_params.in_channels
        self.base_channels = model_params.base_channels
        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * self.num_point)

        block_list = [
            [self.in_channels, self.base_channels, 1, False],
            [self.base_channels, self.base_channels, 1, True],
            [self.base_channels, self.base_channels, 1, True],
            [self.base_channels, self.base_channels, 1, True],
            [self.base_channels, self.base_channels*2, 2, True],
            [self.base_channels*2, self.base_channels*2, 1, True],
            [self.base_channels*2, self.base_channels*2, 1, True],
            [self.base_channels*2, self.base_channels*4, 2, True],
            [self.base_channels*4, self.base_channels*4, 1, True],
            [self.base_channels*4, self.base_channels*4, 1, True],
        ]
        blockargs = [block_list[int(i - 1)] for i in model_params.gcn_include_blocks]
        self.blocks = nn.ModuleList([
            TCN_GCN_unit(ic, oc, A, stride=s, residual=r, adaptive=adaptive) for ic, oc, s, r in blockargs
        ])

        self.fc = nn.Linear(self.base_channels*4, self.num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)

        self.use_frame_diff = model_params.use_frame_diff
        if self.use_frame_diff:
            in_channels_diff = 2
            self.data_bn_diff = nn.BatchNorm1d(num_person * in_channels_diff * self.num_point)

            block_list_diff = [
                [in_channels_diff, self.base_channels, 1, False],
                [self.base_channels, self.base_channels, 1, True],
                [self.base_channels, self.base_channels, 1, True],
                [self.base_channels, self.base_channels, 1, True],
                [self.base_channels, self.base_channels*2, 2, True],
                [self.base_channels*2, self.base_channels*2, 1, True],
                [self.base_channels*2, self.base_channels*2, 1, True],
                [self.base_channels*2, self.base_channels*4, 2, True],
                [self.base_channels*4, self.base_channels*4, 1, True],
                [self.base_channels*4, self.base_channels*4, 1, True],
            ]
            blockargs = [block_list_diff[int(i - 1)] for i in model_params.gcn_include_blocks]
            self.blocks_diff = nn.ModuleList([
                TCN_GCN_unit(ic, oc, A, stride=s, residual=r, adaptive=adaptive) for ic, oc, s, r in blockargs
            ])
            self.fc_diff = nn.Linear(self.base_channels*4, self.num_class)
            nn.init.normal_(self.fc_diff.weight, 0, math.sqrt(2. / self.num_class))
            bn_init(self.data_bn_diff, 1)
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
            
    def forward(self, x, flow=None):
        keypoints = x.detach().clone()
        
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

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        feats = x
        x = self.drop_out(x)
        logits = self.fc(x)
        
        if self.use_frame_diff:
            y = kp_diff_stats(keypoints)
            y = y.unsqueeze(-1)
            N, C, T, V, M = y.size()

            y = y.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            y = self.data_bn_diff(y)
            y = y.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
            for block in self.blocks_diff:
                y = block(y)
            c_new = y.size(1)
            y = y.view(N, M, c_new, -1)
            y = y.mean(3).mean(1)
            feats_y = y
            y = self.drop_out(y)
            logits_y = self.fc_diff(y)
            
            feats = torch.stack([feats, feats_y], dim=0).mean(dim=0)
            logits = torch.stack([logits, logits_y], dim=0).mean(dim=0)
                    
        return feats, logits