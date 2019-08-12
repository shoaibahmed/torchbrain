#!/bin/python

import torch
from torch import nn


class Node(nn.Module):
    def __init__(self, pos, neighbours_dist, neighbourhood_size):
        super(Node, self).__init__()

        neighbours_excitory = (neighbours_dist < neighbourhood_size[0]).float()
        neighbours_inhibitory = (neighbours_dist > neighbourhood_size[1]).float()
        self.weights = neighbours_excitory * 5 + neighbours_inhibitory * torch.exp(-neighbours_dist / 10) * -2.0

        self.a = torch.tensor([0.02])
        self.b = torch.tensor([0.2])
        self.c = -65.0 + 15.0 * torch.pow(torch.rand(1), 2)
        self.d = 8.0 - 6.0 * torch.pow(torch.rand(1), 2)
        self.pos = pos

        self.v = torch.tensor([-65.0])
        self.u = self.b * self.v
        self.threshold = torch.tensor([30.0])
        self.is_firing = False

    def is_node_firing(self):
        firing = self.is_firing
        return firing

    def get_weights(self):
        return self.weights

    def get_pos(self):
        return self.pos

    def forward(self, x):
        # Add random noise to the input
        inp = x + (3.0 * torch.randn(x.size()))  # FIXME: Verify that the level of noise induced is not so high

        # Reset memory if the neuron fired
        self.is_firing = self.v >= self.threshold
        if self.is_firing:
            self.v = self.c.clone()
            self.u = self.u + self.d

        assert self.v <= self.threshold, f"Activation exceeded the threshold value: {self.v} > {self.threshold}"

        # Update the dynamics of the model (stability improved with smaller steps)
        for _ in range(2):
            self.v = self.v + 0.5 * (0.04 * torch.pow(self.v, 2) + 5 * self.v + 140 - self.u + inp)
        self.u = self.u + self.a * (self.b * self.v - self.u)

        return self.v.item()

    def reset(self):
        self.v = torch.tensor([-65.0])
        self.u = self.b * self.v
        self.is_firing = False
