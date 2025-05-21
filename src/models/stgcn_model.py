import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.cli_args import ModelArguments, DataArguments

class Graph:
    def __init__(self, num_nodes:int, neibor_base:list, hop_size:int=2):
        self.num_nodes = num_nodes
        self.neibor_base = neibor_base
        self._check_format()
        self.hop_size = hop_size
        self.edges = self._get_edge()
        self.hop_dis = self._get_hop_distance()
        self.A = self.get_adjancency()
        
    def __str__(self):
        return self.A
        
    def _check_format(self):
        if len(self.neibor_base[0]) != 2: 
            raise ValueError('Each neibor pair should contain only two elements')
        if max([max(ele) for ele in self.neibor_base]) >= self.num_nodes:
            raise ValueError(f'The number of node should less than {self.num_nodes}')
    
    def _get_edge(self):
        self_link = [(i, i) for i in range(self.num_nodes)]
        neibor_link = [(i, j) for i, j in self.neibor_base]
        edges = self_link + neibor_link
        return edges
    
    def _get_hop_distance(self):
        # create initial adjancy matrix
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        
        # create for hop
        # e.g. whether under the distance hop = 0/1/2, can i go from A to C 
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        
        hop_dis = np.zeros((self.num_nodes, self.num_nodes)) + np.inf
        for d in range(self.hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d # Set all positions in hop_dis where arrive_mat[d] == True to the value d.
        return hop_dis
    
    def normalize_diagraph(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(A, Dn)
        return DAD
    
    
    def get_adjancency(self):
        # create a adjacency matrix
        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        A = np.zeros((self.hop_size + 1, self.num_nodes, self.num_nodes))
        
        # create a list for hop
        valid_hop = range(self.hop_size + 1) 
        for hop in valid_hop:
            adjacency_matrix[self.hop_dis == hop] = 1
        
        # normalize
        normalized_adjacency_matrix = self.normalize_diagraph(adjacency_matrix)
        
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalized_adjacency_matrix[self.hop_dis == hop]
        return A

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, s_kernel_size:int):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * s_kernel_size, kernel_size=1)
        
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGC_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int, t_kernel_size:int, A_size:int, dropout:float=0.5):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels=in_channels, out_channels=out_channels, s_kernel_size=A_size[0])
        
        self.M = nn.Parameter(torch.ones(A_size))
        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(t_kernel_size, 1), stride=(stride, 1), padding=((t_kernel_size-1) // 2, 0)),
                                 nn.BatchNorm2d(out_channels), 
                                 nn.ReLU())
    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x

class ST_GCN(nn.Module):
    def __init__(self, params:ModelArguments, d_params:DataArguments):
        super().__init__()
        # graph generation
        graph = Graph(num_nodes=d_params.num_nodes, neibor_base=params.neighbor_base, hop_size=params.hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(params.in_channels * A_size[1]) # 75

        # STGC_blocks
        self.stgc1 = STGC_block(params.in_channels, params.intermediate_channels, 1, params.t_kernel_size, A_size) # in_c=3, t_k_s= 9
        self.stgc2 = STGC_block(params.intermediate_channels, params.intermediate_channels, 1, params.t_kernel_size, A_size)
        self.stgc3 = STGC_block(params.intermediate_channels, params.final_channels, 2, params.t_kernel_size, A_size)
        self.stgc4 = STGC_block(params.final_channels, params.final_channels, 1, params.t_kernel_size, A_size)

        # Prediction
        self.fc = nn.Conv2d(params.final_channels, params.num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size() # batch, channel, frame, node

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)

        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    
    # for test
    num_nodes = 17
    neibor_base = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4],[3, 5],[5, 6],[6, 7],[3, 8],[8, 9], \
        [9, 10], [4, 14], [14, 15], [15, 16], [4, 11],[11, 12], [12, 13]]
    graph = Graph(num_nodes=num_nodes, neibor_base=neibor_base)
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
    print(A.shape)
    A = np.array(A)
    
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    A = np.array(A)[2]
 
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(A, cmap='viridis')  # 使用viridis颜色映射
    plt.colorbar(im)  # 
    # plt.title('Matrix Image with Values')
    # plt.xlabel('X Label')
    # plt.yl