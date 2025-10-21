import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .degcn import Basic_Block
from .gcn.graph import Graph

# KP 
class KP_EMBEDDING(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.blocks = nn.ModuleList([
                Basic_Block(in_channels=3, out_channels=32, A=self.A, num_frame=32, eta=4),
                Basic_Block(in_channels=32, out_channels=64, A=self.A, num_frame=32//2, eta=4, stride=2),
                Basic_Block(in_channels=64, out_channels=128, A=self.A, num_frame=32//4, eta=4, stride=2),
                Basic_Block(in_channels=128, out_channels=128, A=self.A, num_frame=32//4, eta=4),
                ]
            )
    def forward(self, keypoints: torch.Tensor):
        x = keypoints
        for block in self.blocks:
            x = block(x)
        return x # (B, E, T, V)


# I3D 
def _best_grid(P: int):
    # 回傳 (gh, gw)，使 gh*gw 最接近且不小於 P（常見 P=16 → 4x4）
    g = int(np.ceil(np.sqrt(P)))
    gh = gw = g
    return gh, gw

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=11, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 E: int = 256, P: int = 16):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        self.E = E
        self.P = P
        self.gh, self.gw = _best_grid(P)
        
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
        self.proj = nn.Conv3d(384+384+128+128, self.E, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(self.E, eps=0.001, momentum=0.01)
        self.proj_act = nn.ReLU(inplace=True)

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for i, end_point in enumerate(self.VALID_ENDPOINTS):
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel
        x_proj = self.proj_act(self.proj_bn(self.proj(x)))    # (B, E, T, H, W)
        
        x_tokens = F.adaptive_avg_pool3d(x_proj, output_size=(x_proj.size(2), 4, 4)) # (B, E, T, 4, 4)
        feat = x_tokens.flatten(3)                         # (B, E, T, 16)
        
        return feat
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
    
class I3D_GCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        neighbor_base,
        num_classes: int = 11,
        E: int = 128,
        T: int = 32,
        V: int = 17,
        P: int = 16,
        E_proj: int = 256,      # 投影後的共同空間維度（你可改成 128/512 等）
        Cb: int = 256,          # I3D backbone base channels
        drop: float = 0.1
    ):
        super().__init__()
        self.E, self.T, self.V, self.P = E, T, V, P
        self.num_classes = num_classes
        self.E_proj = E_proj

        # --- Embeddings ---
        graph = Graph(num_nodes=num_nodes, neighbor_base=neighbor_base)
        A = graph.A  # (3, V, V)
        self.kp_embed  = KP_EMBEDDING(A)                 # (B, E, T, V)
        self.i3d_embed = InceptionI3d(
            E=E, P=P, in_channels=2
        )                                                # (B, E, T, P)
        
        self.kp_proj = nn.Sequential(
            nn.Linear(E * V, 512, bias=False),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, E_proj, bias=True)
        )
        self.i3d_proj = nn.Sequential(
            nn.Linear(E * P, 512, bias=False),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, E_proj, bias=True)
        )

        # --- Learnable fusion weight α ---
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # α = sigmoid(alpha_logit)

        # --- Head: temporal pooling + MLP classifier ---
        self.norm = nn.LayerNorm(E_proj)
        self.mlp = nn.Sequential(
            nn.Linear(E_proj, E_proj // 2),
            nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(E_proj // 2, num_classes)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(2 * E_proj, E_proj // 2, bias=False),
            nn.GELU(),
            nn.Linear(E_proj // 2, 1, bias=True)
        )
        
        # cosine sim
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        self.pe_v = nn.Parameter(torch.randn(1, E, 1, V) * 0.02)
        self.pe_p = nn.Parameter(torch.randn(1, E, 1, P) * 0.02)
        
        # --- Learnable fusion weight α ---
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # α = sigmoid(alpha_logit)

    def forward(
        self,
        keypoints: torch.Tensor,       # (B, 3, T, V)
        optical_flows: torch.Tensor    # (B, 2, T, H, W)
        ):
        # ---- 1) Embedding ----
        f_kp  = self.kp_embed(keypoints)        # (B, E, T, V)
        f_i3d = self.i3d_embed(optical_flows)   # (B, E, T, P)
        
        f_kp  = f_kp + self.pe_v          # (B,E,T,V)
        f_i3d = f_i3d + self.pe_p         # (B,E,T,P)

        # ---- 2) Reshape to (B, T, E*V) and (B, T, E*P) ----
        # (B, E, T, V) -> (B, T, E*V)
        B, E, T, V = f_kp.shape
        f_kp_rs = f_kp.permute(0, 2, 1, 3).contiguous().view(B, T, E * V)

        # (B, E, T, P) -> (B, T, E*P)
        _, _, T2, P = f_i3d.shape
        assert T2 == T, f"T mismatch: kp T={T}, i3d T={T2}"
        f_i3d_rs = f_i3d.permute(0, 2, 1, 3).contiguous().view(B, T, E * P)

        # ---- 3) Project both to (B, T, E_proj) ----
        kp_proj  = self.kp_proj(f_kp_rs)        # (B, T, E_proj)
        i3d_proj = self.i3d_proj(f_i3d_rs)      # (B, T, E_proj)

        # 可選：正規化後再做對齊與融合（讓 cos 更穩定）
        kp_proj_n  = F.normalize(kp_proj,  dim=-1)
        i3d_proj_n = F.normalize(i3d_proj, dim=-1)
        
        # ---- 4) Weighted average fusion ----
        alpha = torch.sigmoid(self.alpha_logit)     # scalar in (0,1)
        fused = alpha * kp_proj_n + (1.0 - alpha) * i3d_proj_n   # (B, T, E_proj)

        # # ---- 4) Weighted average fusion ----
        # gate_in = torch.cat([kp_proj_n, i3d_proj_n], dim=-1)  # (B,T,2E')
        # alpha   = torch.sigmoid(self.gate(gate_in))           # (B,T,1)
        # fused   = alpha * kp_proj_n + (1 - alpha) * i3d_proj_n

        # ---- Temporal pooling -> (B, E_proj) ----
        fused_pool = fused.mean(dim=1)
        
        feats = self.norm(fused_pool)
        feats = self.mlp(feats)

        # ---- Classifier ----
        logits = self.head(feats)   # (B, num_classes)

        # ---- Cosine similarity alignment loss ----
        # 逐時間步做 cos，相當於 (B,T)
        cos_t = self.cos_sim(kp_proj_n, i3d_proj_n)
        cos_loss = 1.0 - cos_t.mean()               # scalar

        return feats, logits, cos_loss