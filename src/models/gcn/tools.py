import math
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    
def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)

def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def local_flow_stats(keypoints: torch.Tensor,
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