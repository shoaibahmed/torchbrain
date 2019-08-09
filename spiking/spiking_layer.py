#!/bin/bash

import numpy as np
import torch
from torch import nn

from node import Node


def pairwise_distance(u, v=None):
    """
    Euclidean pairwise dist. similar to scipy.spatial.distance.pdist.
    Args:
        u (tensor): 2d tensor
        v (tensor): 2d tensor (if none, computes pairwise distance with the same vector)
    """
    if v is None:
        v = u

    n = u.size(0)
    m = v.size(0)
    assert u.size(1) == v.size(1)
    d = u.size(1)

    x = u.unsqueeze(1).expand(n, m, d)
    y = v.unsqueeze(0).expand(n, m, d)

    return torch.sqrt(torch.pow(x - y, 2).sum(2))


class SpikingLayer(nn.Module):
    def __init__(self, num_neurons, square_size, neighbourhood_size):
        super(SpikingLayer, self).__init__()
        self.num_neurons = num_neurons

        # 1. Define topology - square
        topology = torch.rand(num_neurons, 2) * square_size

        # 2. Define node neighbours
        dist_matrix = pairwise_distance(topology)

        self.is_firing = None
        self.node_x = None
        self.firing_matrix = []

        # 3. Initialize nodes
        self.nodes = nn.ModuleList([Node(topology[i, :], dist_matrix[i, :], neighbourhood_size)
                                    for i in range(num_neurons)])

        # Set the state of the model to none
        self.reset_state()

    def forward(self, x):
        self.node_x = [torch.matmul(node.get_weights(), self.is_firing) for node in self.nodes]
        for idx, node in enumerate(self.nodes):
            _ = node(self.node_x[idx])
            self.is_firing[idx] = node.is_node_firing()
        self.firing_matrix.append(self.is_firing.cpu().clone().numpy())
        return None  # FIXME

    def reset_state(self):
        self.firing_matrix = []
        self.is_firing = torch.zeros(self.num_neurons)
        for node in self.nodes:
            node.reset()

    def get_firing_matrix(self):
        return self.firing_matrix


class LGNLayer(nn.Module):
    # TODO: Add possibility to add multiple layers of Retina or LGN - layer idx
    def __init__(self, num_neurons_retina, num_neurons_lgn, square_size, neighbourhood_size):
        super(LGNLayer, self).__init__()
        self.num_neurons = num_neurons_retina

        # Parameters of the LGN
        self.eta = 0.1
        self.decay_rt = 0.01

        self.mu_wts = 2.5
        self.sigma_wts = 0.14

        # 1. Define topology - square
        topology = torch.rand(num_neurons_retina, 2) * square_size
        topology_3d = torch.cat([topology, torch.full((num_neurons_retina, 1), 0.0)], dim=-1)  # Fixed z coordinate based on the layer

        # 2. Define node neighbours
        dist_matrix = pairwise_distance(topology)

        # 3. Spawn LGN nodes at random locations on the retina
        topology_lgn = torch.randint(0, num_neurons_retina, (num_neurons_lgn,))
        topology_lgn = topology[topology_lgn]
        topology_lgn_3d = torch.cat([topology_lgn, torch.full((num_neurons_lgn, 1), 1.0)], dim=-1)  # Fixed z coordinate based on the layer

        # 4. Define the distance matrix for the nodes in the retina and LGN
        self.dist_matrix_lgn = pairwise_distance(topology_lgn_3d, topology_3d)

        # 5. Define LGN synaptic weights
        lgn_weights = torch.normal(self.mu_wts, self.sigma_wts, size=(num_neurons_lgn, num_neurons_retina))
        self.lgn_weights = lgn_weights / torch.mean(lgn_weights, dim=1, keepdim=True)  # Normalize the input to a particular LGN neuron
        self.lgn_threshold = torch.normal(70.0, 2.0, size=(num_neurons_lgn,))  # Number of LGN layers

        self.is_firing = None
        self.node_x = None
        self.firing_matrix = []

        # 3. Initialize nodes
        self.nodes = nn.ModuleList([Node(topology[i, :], dist_matrix[i, :], neighbourhood_size)
                                    for i in range(num_neurons_retina)])

        # Set the state of the model to none
        self.reset_state()

    def forward(self, x):
        self.node_x = [torch.matmul(node.get_weights(), self.is_firing) for node in self.nodes]
        for idx, node in enumerate(self.nodes):
            _ = node(self.node_x[idx])
            self.is_firing[idx] = node.is_node_firing()
        self.firing_matrix.append(self.is_firing.cpu().clone().numpy())

        # TODO: Hebian learning for the LGN nodes
        y1_allLGN = torch.matmul(self.lgn_weights, self.is_firing)
        y1_allLGN[y1_allLGN < 0] = 0.9

        yAct_allLGN = y1_allLGN - self.lgn_threshold
        yAct_allLGN[yAct_allLGN < 0] = 0.0

        maxAct, maxInd_LGN = torch.max(yAct_allLGN, dim=-1)

        if (yAct_allLGN[maxInd_LGN] > 0).any():
            # Modify weights ONLY for maxInd_LGN

            x_input = self.is_firing
            wt_input = self.lgn_weights[maxInd_LGN, :]

            for _ in range(2):
                wt_input = wt_input + 0.5 * (self.eta * (yAct_allLGN[maxInd_LGN]) * x_input)

            self.lgn_weights[:, maxInd_LGN] = wt_input

            # Modifying threshold! If threshold is much larger than activity :reduce ELSE increase
            self.lgn_threshold[maxInd_LGN] = self.lgn_threshold[maxInd_LGN] + 0.005 * yAct_allLGN[maxInd_LGN]

            # Normalize weights to a constant strength
            self.lgn_weights[maxInd_LGN, :] = self.lgn_weights[maxInd_LGN, :] / \
                                              torch.mean(self.lgn_weights[maxInd_LGN, :]) * self.mu_wts

        return None  # FIXME

    def reset_state(self):
        self.firing_matrix = []
        self.is_firing = torch.zeros(self.num_neurons)
        for node in self.nodes:
            node.reset()

    def get_firing_matrix(self):
        return self.firing_matrix


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
        # Evolve the dynamics of the model over time
        for _ in range(self.num_timesteps):
            _ = self.layer(x)

        # Get neuron firing pattern and reset its state
        for idx in range(len(self.layer)):
            firing_matrix = self.layer[idx].get_firing_matrix()
            self.layer[idx].reset_state()

        return firing_matrix


class SpikingLGN(nn.Module):
    # TODO: Separate out the retina and LGN layers
    def __init__(self, num_retina_layers, num_lgn_layers, num_neurons_retina, num_neurons_lgn,
                 square_size, num_classes, neighbourhood_size=(3, 5), num_timesteps=10):
        super(SpikingLGN, self).__init__()
        self.num_retina_layers = num_retina_layers
        self.num_lgn_layers = num_lgn_layers
        self.num_timesteps = num_timesteps

        layers = []
        for i in range(num_lgn_layers):
            layer = LGNLayer(num_neurons_retina, num_neurons_lgn, square_size=square_size, neighbourhood_size=neighbourhood_size)
            layers.append(layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        # Evolve the dynamics of the model over time
        for _ in range(self.num_timesteps):
            _ = self.layer(x)

        # Get neuron firing pattern and reset its state
        for idx in range(len(self.layer)):
            # TODO: Add compatibility for LGN layers
            firing_matrix = self.layer[idx].get_firing_matrix()
            self.layer[idx].reset_state()

        # TODO: Add classification here?

        return firing_matrix


if __name__ == "__main__":
    # net = SpikingNN(1, 1500, 50, 10, (2, 4), num_timesteps=500)
    net = SpikingLGN(num_retina_layers=1, num_lgn_layers=1, num_neurons_retina=1500, num_neurons_lgn=400,
                     square_size=50, num_classes=10, neighbourhood_size=(3, 5), num_timesteps=100)
    inp = torch.rand((1, 1, 28, 28))
    out = net(inp)
    out = np.array(out)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 9))
    plt.imshow(out)
    plt.xlabel('Neuron ID')
    plt.ylabel('Activation over time')
    plt.tight_layout()
    plt.show()

    print(out.shape)
