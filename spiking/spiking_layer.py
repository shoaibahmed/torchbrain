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

    return torch.sqrt(torch.pow(x - y, 2).sum(2))


class SpikingLayer(nn.Module):
    def __init__(self, num_neurons, square_size, neighbourhood_size, norm=2):
        super(SpikingLayer, self).__init__()
        self.num_neurons = num_neurons

        # 1. Define topology - square
        topology = torch.rand(num_neurons, 2) * square_size

        # 2. Define node neighbours
        dist_matrix = pairwise_distance(topology)

        self.is_firing = None
        self.node_x = None
        self.reset_state()

        # 3. Initialize nodes
        self.nodes = nn.ModuleList([Node(topology[i, :], dist_matrix[i, :], neighbourhood_size)
                                    for i in range(num_neurons)])

    def forward(self, x):
        self.node_x = [torch.matmul(node.get_weights(), self.is_firing) for node in self.nodes]
        for idx, node in enumerate(self.nodes):
            _ = node(self.node_x[idx])
            self.is_firing[idx] = node.is_node_firing()

        return x

    def reset_state(self):
        self.is_firing = torch.zeros(self.num_neurons)


class SpikingNN(nn.Module):
    def __init__(self, num_layers, num_neurons, square_size, num_classes, neighbourhood_size=(3, 5), num_timesteps=10):
        super(SpikingNN, self).__init__()
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        layers = []
        for i in range(num_layers):
            layer = SpikingLayer(num_neurons=num_neurons, square_size=square_size, neighbourhood_size=neighbourhood_size)
            layers.append(layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        for _ in range(self.num_timesteps):
            x = self.layer(x)
        return x


if __name__ == "__main__":
    net = SpikingNN(1, 10, 10, 10, (3, 5))
    inp = torch.rand((1, 1, 28, 28))
    out = net(inp)
