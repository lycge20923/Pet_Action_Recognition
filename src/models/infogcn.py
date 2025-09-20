import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange

from .gcn.tools import *
from .gcn.graph import Graph
from ..utils.cli_args import ModelArguments, DataArguments
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


class MS_TCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
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

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()
        return attn

class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)


    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        out = None
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :] # (nt)vv
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(UnitTCN, self).__init__()
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

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MS_TCN(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, attn=None):
        y = self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))
        return y


class Info_GCN(nn.Module):
    def __init__(self, model_params:ModelArguments, data_params:DataArguments, num_person=1, noise_ratio=0.1, k=1, gain=1, drop_out=0):
        super(Info_GCN, self).__init__()
        
        self.graph = Graph(num_nodes=data_params.num_nodes, neighbor_base=model_params.neighbor_base)
        self.A = self.graph.A
        num_head = self.A.shape[0]

        self.num_class = model_params.num_classes
        self.num_point = data_params.num_nodes
        base_channel = model_params.base_channels
        
        A = np.stack([np.eye(self.num_point)] * num_head, axis=0)
        
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * self.num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(self.num_class, base_channel*4)
        self.A_vector = self.get_A(self.graph, k)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(model_params.in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        
        block_list = [
            [base_channel, base_channel, 1], 
            [base_channel, base_channel, 1], 
            [base_channel, base_channel, 1],
            [base_channel, base_channel, 1],
            [base_channel, base_channel*2, 2],
            [base_channel*2, base_channel*2, 1],
            [base_channel*2, base_channel*2, 1],
            [base_channel*2, base_channel*4, 2], 
            [base_channel*4, base_channel*4, 1],
            [base_channel*4, base_channel*4, 1],
        ]
        blockargs = [block_list[int(i - 1)] for i in model_params.gcn_include_blocks]
        self.blocks = nn.ModuleList([
            EncodingBlock(in_channels=ic, out_channels=oc, A=A, stride=s) for ic, oc, s in blockargs
        ])

        self.fc = nn.Linear(base_channel*4, base_channel*4)
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, self.num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        
        self.use_frame_diff = model_params.use_frame_diff
        if self.use_frame_diff:
            in_channels_diff = 2
            self.to_joint_embedding_diff = nn.Linear(in_channels_diff, base_channel)
            self.pos_embedding_diff = nn.Parameter(torch.randn(1, self.num_point, base_channel))
            self.data_bn_diff = nn.BatchNorm1d(num_person * base_channel * self.num_point)
            self.blocks_diff = nn.ModuleList([
                EncodingBlock(in_channels=ic, out_channels=oc, A=A, stride=s) for ic, oc, s in blockargs
            ])
            self.fc_diff = nn.Linear(base_channel*4, base_channel*4)
            self.fc_mu_diff = nn.Linear(base_channel*4, base_channel*4)
            self.fc_logvar_diff = nn.Linear(base_channel*4, base_channel*4)
            self.decoder_diff = nn.Linear(base_channel*4, self.num_class)
            
            nn.init.xavier_uniform_(self.fc_diff.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc_mu_diff.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc_logvar_diff.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.normal_(self.decoder_diff.weight, 0, math.sqrt(2. / self.num_class))
            bn_init(self.data_bn_diff, 1)    
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph:Graph, k):
        A_outward = get_adjacency_matrix(graph.outward, self.num_point)
        I = np.eye(self.num_point)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x:torch.Tensor, flow:torch.Tensor=None):
        keypoints = x.detach().clone()
        
        x = x.unsqueeze(dim=-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(device=x.device, dtype=x.dtype).expand(N*M*T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        for block in self.blocks:
            x = block(x)
            
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        feats = self.latent_sample(z_mu, z_logvar)

        logits = self.decoder(feats)
        
        if self.use_frame_diff:
            y = kp_diff_stats(keypoints)
            y = y.unsqueeze(dim=-1)
            N, C, T, V, M = y.size()
            y = rearrange(y, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
            y = self.A_vector.to(device=y.device, dtype=y.dtype).expand(N*M*T, -1, -1) @ y

            y = self.to_joint_embedding_diff(y)
            y += self.pos_embedding_diff[:, :self.num_point]
            y = rearrange(y, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

            y = self.data_bn_diff(y)
            y = rearrange(y, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
            for block in self.blocks_diff:
                y = block(y)
                
            # N*M,C,T,V
            c_new = y.size(1)
            y = y.view(N, M, c_new, -1)
            y = y.mean(3).mean(1)
            y = F.relu(self.fc_diff(y))
            y = self.drop_out(y)

            z_mu = self.fc_mu_diff(y)
            z_logvar = self.fc_logvar_diff(y)
            feats_y = self.latent_sample(z_mu, z_logvar)
            logits_y = self.decoder_diff(feats_y)
            
            feats = torch.stack([feats, feats_y], dim=0).mean(dim=0)
            logits = torch.stack([logits, logits_y], dim=0).mean(dim=0)
            

        return feats, logits