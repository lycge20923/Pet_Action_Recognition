import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn.tools import *
from .gcn.graph import Graph
from .gcn.new_modules import FlowAdjacency, NodeAttention
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

    # def forward(self, x):
    def forward(self, x, keypoints=None, flow_map=None):
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
        return x
    

class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1, num_nodes=17, flow_adj=False): # 64*a, 64*b, (3, 17, 17), 4, 17
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale
        
        '''
        in_channels would never be 3 since it only exists in first block 
        '''
        rel_channels = in_channels // 8 if in_channels != 3 else 8 # 64*a//8 
        
        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        
        # self.flow_adj = FlowAdjacency(num_nodes=num_nodes,
        #                               patch_size=5,
        #                               hidden_dim=16)
        self.flow_adj = flow_adj
        if self.flow_adj:
            self.flow_adj = FlowAdjacency(num_nodes=num_nodes, patch_size=5, hidden_dims=[16, 32, 16])

    # def forward(self, x, A=None, alpha=1):
    def forward(self, x, keypoints=None, flow_seq=None, alpha=1.0, beta=1.0):
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
        
        # weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)
        A_flow = None
        if self.flow_adj and (keypoints is not None) and (flow_seq is not None):
            A_flow = self.flow_adj(flow_seq, keypoints)  # (N, V, V)
            Af = A_flow.unsqueeze(1).expand(-1, self.Nh, -1, -1) 
            A_dyn = (beta * Af).unsqueeze(1).unsqueeze(3)
            A_dyn = A_dyn.to(weights.dtype).to(weights.device)
        else:
            A_dyn = 0
        dtype = weights.dtype
        device = weights.device
        
        # test
        with torch.no_grad():
            import json
            stats = {
                "self.A_min":  self.A.min().item(),
                "self.A_max":  self.A.max().item(),
                "self.A_mean": self.A.mean().item(),
                "self.A_std":  self.A.std().item(),
                "weights_min":  weights.min().item(),
                "weights_max":  weights.max().item(),
                "weights_mean": weights.mean().item(),
                "weights_std":  weights.std().item(),
                }
            if A_flow is not None:
                stats["A_flow_min"] = A_flow.min().item()
                stats["A_flow_max"] = A_flow.max().item()
                stats["A_flow_mean"] = A_flow.mean().item()
                stats["A_flow_std"] = A_flow.std().item()
            with open('supervise_A.json', 'w', encoding='utf-8') as f:
                 json.dump(stats, f, ensure_ascii=False, indent=4)
        
        A_static = (alpha * self.A).view(1,1,self.Nh,1,V,V).to(dtype).to(device)
        weights = weights * self.alpha.to(dtype) + A_static + A_dyn
            
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
    def __init__(self, in_channels, out_channels, A, k, eta, kernel_size=5, stride=1, dilations=2, 
                 num_frame=64, num_joint=25, residual=True, flow_adj=False, node_attention=False):
        super(Basic_Block, self).__init__()
        
        num_scale = 4
        # scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1
        
        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            self.gcn = CTR_GC(in_channels, 
                              out_channels, 
                              A, 
                              self.num_scale,
                              flow_adj=flow_adj)
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
        
        # for node attentions
        self.node_attention = node_attention
        if node_attention:
            self.node_att = NodeAttention(in_channels=out_channels, num_heads=4)
            # self.node_att = NodeAttention(in_channels=out_channels, num_heads=4, A=A)
    
    # def forward(self, x):
    def forward(self, x, keypoints=None, flow_map=None):
        res = x
        x = self.gcn(x, keypoints, flow_map)
        
        # add node attentions
        if self.node_attention: 
            # print("x.shape", x.shape)
            alpha = self.node_att(x)
            
            # test
            with torch.no_grad():
                import json
                stats = {
                    "alpha_min":  alpha.min().item(),
                    "alpha_max":  alpha.max().item(),
                    "alpha_mean": alpha.mean().item(),
                    "alpha_std":  alpha.std().item()
                    }
                with open('supervise_node_attn.json', 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=4)
            
            # print("alpha:", alpha)
            B, C, T, V = x.size()
            alpha = alpha.view(B, 1, 1, V)
            res1 = self.residual1(res)
            gated = alpha * x + (1 - alpha) * res1
            x = self.relu(gated)
        else:
            x = self.relu(x + self.residual1(res))
        
        x = self.tcn(x)
        x = self.relu(x + self.residual2(res))
        return x


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class DeGCN(nn.Module):
    def __init__(self, block_args, A, k, eta, flow_adj, node_attention):
        super().__init__()
        self.blocks = nn.ModuleList([
            Basic_Block(
                in_ch, out_ch, A, k, eta,
                stride=stride,
                num_frame=num_frame,
                num_joint=num_joint,
                residual=residual,
                node_attention=node_attention,
                flow_adj=flow_adj
            )
            for in_ch, out_ch, stride, residual, num_frame, num_joint in block_args
        ])

    def forward(self, x, keypoints=None, flow_seq=None):
        # x: (N*M, C, T, V)
        for block in self.blocks:
            x = block(x, keypoints, flow_seq)
        return x

class DE_GCN(nn.Module):
    def __init__(
        self, model_params:ModelArguments, data_params:DataArguments, num_person=1, k=8, eta=4, drop_out=0,
    ):
        super(DE_GCN, self).__init__()
        self.graph = Graph(num_nodes=data_params.num_nodes, neighbor_base=model_params.neighbor_base)
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
            DeGCN(self.blockargs, A, k, eta, flow_adj=model_params.degcn_add_of_A, node_attention=model_params.degcn_add_node_attention) for _ in range(self.num_stream)
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

    # def forward(self, x):
    def forward(self, x, flow_map=None):
        keypoints = x.detach().clone()
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
            if flow_map is not None:
                x_s = stream(x_, keypoints, flow_map) 
            else:
                x_s = stream(x_) # (N*M, C', T', V')
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
    pass
    # model = MambaBlock(d_model=4)
    # print(model)