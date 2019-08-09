#!/bin/python

import torch
from torch import nn


class Node(nn.Module):
    def __init__(self, pos, neighbours_dist, neighbourhood_size):
        super(Node, self).__init__()

        neighbours_excitory = neighbours_dist < neighbourhood_size[0]
        neighbours_inhibitory = neighbours_dist > neighbourhood_size[1]
        self.weights = neighbours_excitory * 5 + neighbours_inhibitory * torch.exp(neighbours_dist / 10) * -2

        self.a = torch.tensor([0.02])
        self.b = torch.tensor([0.2])
        self.c = -65 + 15 * torch.square(torch.rand())
        self.d = 8 - 6 * torch.square(torch.rand())
        self.pos = pos

        self.v = torch.tensor([-65])
        self.u = self.b * self.v
        self.threshold = torch.tensor([30])
        self.is_firing = False

    def get_pos(self):
        return self.pos

    def is_node_firing(self):
        firing = self.is_firing
        self.is_firing = False
        return firing

    def get_weights(self):
        return self.weights

    def forward(self, x):
        # Add random noise to the input
        input = x + 3 * torch.randn(x.size())

        # Reset memory if the neuron fired
        self.is_firing = self.v > self.threshold
        if self.is_firing:
            self.v = self.c
            self.u += self.d

        # Update the dynamics of the model (stability improved with smaller steps)
        for _ in range(2):
            self.v += 0.5 * (0.04 * self.v ** 2 + 5 * self.v * 140 - self.u + input)
        self.u += self.a * (self.b * self.v - self.u)

        return x
