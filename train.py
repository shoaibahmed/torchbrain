#!/bin/python

import os
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchvision import transforms
from torchvision import datasets
from spiking import SpikingELM
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    # Clear output directory
    if os.path.exists(args.output_dir):
        print("Removing old directory!")
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    # Create model
    model = SpikingELM(num_retina_layers=1, num_lgn_layers=1, num_neurons_retina=1500, num_neurons_lgn=400,
                       square_size=50, num_classes=args.num_classes, hidden_rep=128, neighbourhood_size=(3, 5),
                       num_timesteps=250, device=device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    # Create dataloader
    input_size = (28, 28)
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    dataset = datasets.MNIST(args.root_dir, train=True, transform=data_transform)
    data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.training_epochs):
        # Start training
        model.train()

        pbar = tqdm(enumerate(data_loader))
        for iter_idx, (X, y) in pbar:
            # Move the data to PyTorch on the desired device
            X = X.float().to(device)
            y = y.long().to(device)

            # Get model predictions
            pred = model(X)

            # Optimize
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            # scheduler.step(val_loss)

            if iter_idx % args.display_step == 0:
                pbar.set_description(f"Epoch {epoch+1} | Iteration: {iter_idx+1} | Loss: {loss:.5f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))


if __name__ == "__main__":
    # Command line arguments
    parser = ArgumentParser("Spiking NN training script")

    # Base args
    parser.add_argument("-t", "--train-model", action="store_true", default=False, help="Train model")
    parser.add_argument("-c", "--test-model", action="store_true", default=False, help="Test model")
    parser.add_argument("-o", "--output-dir", action="store", type=str, default="./output", help="Output directory")
    parser.add_argument("-e", "--training-epochs", action="store", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch-size", action="store", type=int, default=1, help="Batch Size")
    parser.add_argument("-d", "--display-step", action="store", type=int, default=2, help="Display step where the loss should be displayed")

    # Input reader args
    parser.add_argument("--root-dir", action="store", type=str, default="./data/", help="Root directory containing the data")
    parser.add_argument("--num-classes", action="store", type=int, default=10, help="Number of classes in the dataset")

    # Parse command line args
    args = parser.parse_args()
    print(args)

    if args.train_model:
        print("Training model")
        train(args)

    if args.test_model:
        print("Testing model")
