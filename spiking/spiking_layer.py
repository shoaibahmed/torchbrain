import torch
from torch import nn


class SpikingLayer(nn.Module):
    def __init__(self, num_neurons):
        super(SpikingLayer, self).__init__()
        self.num_neurons = num_neurons

    def forward(self, x):
        return x



class SpikingNN(nn.Module):
    def __init__(self, num_layers):
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
