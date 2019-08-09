#!/bin/bash

import torch
from torch import nn

from node import Node


class SpikingLayer(nn.Module):
    def __init__(self, num_neurons, square_size, neighbourhood_size, norm=2):
        super(SpikingLayer, self).__init__()
        self.num_neurons = num_neurons

        # 1. Define topology - square
        topology = torch.rand(num_neurons, 2) * square_size

        # 2. Define node neighbours
        pdist = nn.PairwiseDistance(p=norm)
        dist_matrix = pdist(topology, topology)

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
    def __init__(self, num_layers, num_neurons, num_classes, num_timesteps=100):
        super(SpikingNN, self).__init__()
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        layers = []
        for i in range(num_layers):
            layer = SpikingLayer()
            layers.append(layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        for _ in range(self.num_timesteps):
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
