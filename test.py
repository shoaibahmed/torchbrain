#!/bin/python

import numpy as np
import torch
import matplotlib.pyplot as plt
from spiking import SpikingLGN

from torchvision.datasets import MNIST
from torchvision import transforms


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the network
    # net = SpikingNN(1, 1500, 50, 10, (2, 4), num_timesteps=500)
    net = SpikingLGN(num_retina_layers=1, num_lgn_layers=1, num_neurons_retina=1500, num_neurons_lgn=400,
                     square_size=28, neighbourhood_size=(3, 5), num_timesteps=1200, device=device).to(device)

    # Define the inputs
    # inp = torch.rand((1, 1, 28, 28)).to(device)
    train_dataset = MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    inp = train_dataset[0][0][None, :, :, :]  # Returns a tuple
    inp = inp.to(device)
    print(inp.shape)

    # Forward prop the input to the network
    out = net(inp)
    out = np.array(out)

    # Create the raster plot (neuron activation map)
    plt.figure(figsize=(10, 10))
    plt.imshow(out)
    plt.title('Raster Plot')
    plt.xlabel('Neuron ID')
    plt.ylabel('Activation over time')
    plt.tight_layout()
    plt.show()

    print(out.shape)
