import sys
import numpy as np

from .tools import *
from ..utils.cli_args import ModelArguments, DataArguments

class Graph:
    def __init__(self, model_params:ModelArguments, data_params:DataArguments, labeling_mode='spatial'):
        self.num_node = data_params.num_nodes
        self.self_link = [(i, i) for i in range(self.num_node)]
        inward_ori_index = model_params.neighbor_base
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A