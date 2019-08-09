#!/bin/bash

import torch
from torch import nn

from node import Node

def pairwise_distance(t):
    """
    Euclidean pairwise dist. similar to scipy.spatial.distance.pdist.
    Args:
        t (tensor): 2d tensor
    """

    n = t.size(0)
    m = t.size(0)
    d = t.size(1)

    x = t.unsqueeze(1).expand(n, m, d)
    y = t.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class SpikingLayer(nn.Module):
    def __init__(self, num_neurons, square_size, neighbourhood_size, norm=2):
        super(SpikingLayer, self).__init__()
        self.num_neurons = num_neurons

        # 1. Define topology - square
        topology = torch.rand(num_neurons, 2) * square_size

        # 2. Define node neighbours
        dist_matrix = pairwise_distance(topology)

        # 3. Initialize nodes
        nodes = nn.ModuleList([Node(topology[i, :], dist_matrix[i, :])
                               for i in range(num_neurons)])

    def forward(self, x):
        return x



class SpikingNN(nn.Module):
    def __init__(self, num_layers, num_neurons, num_classes):
        super(SpikingNN, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layer = SpikingLayer()
            layers.append(layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class SpikingELM(nn.Module):
    def __init__(self, num_layers, num_neurons, num_classes):
        super(SpikingELM, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layer = SpikingLayer()
            layers.append(layer)
        self.layer = nn.Sequential(*layers)
        self.cls_layer = nn.Linear(num_neurons, num_classes)

    def forward(self, x):
        x = self.layer(x)
        x = self.cls_layer(x)
        return x
