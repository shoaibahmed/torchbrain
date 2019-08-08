#!/bin/python

import os
import shutil
from optparse import OptionParser

import torch
from torchvision import transforms
from torchvision import datasets
from spiking import SpikingELM
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(options):
    # Clear output directory
    if os.path.exists(options.outputDir):
        print ("Removing old directory!")
        shutil.rmtree(options.outputDir)
    os.mkdir(options.outputDir)

    # Create model
    model = SpikingELM()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    # Create dataloader
    dataTransform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    dataset = datasets.MNIST(options.rootDir, split=Data.TRAIN, transform=dataTransform)
    dataLoader = DataLoader(dataset=dataset, num_workers=8, batch_size=options.batchSize, shuffle=True)
    assert options.numClasses == dataset.getNumClasses(), "Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!" % (dataset.getNumClasses(), options.numClasses)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(options.trainingEpochs):
        # Start training
        model.train()

        for iterationIdx, data in enumerate(dataLoader):
            X = data["data"]
            y = data["label"]

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

            if iterationIdx % options.displayStep == 0:
                print("Epoch %d | Iteration: %d | Loss: %.5f" % (epoch, iterationIdx, loss))

        # Save model
        torch.save(model.state_dict(), os.path.join(options.outputDir, "model.pth"))


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    # Base options
    parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")
    parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
    parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
    parser.add_option("-o", "--outputDir", action="store", type="string", dest="outputDir", default="./output", help="Output directory")
    parser.add_option("-e", "--training_epochs", action="store", type="int", dest="trainingEpochs", default=10, help="Number of training epochs")
    parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=22, help="Batch Size (will be divided equally among different GPUs in multi-GPU settings)")
    parser.add_option("-d", "--displayStep", action="store", type="int", dest="displayStep", default=2, help="Display step where the loss should be displayed")

    # Input Reader Params
    parser.add_option("--rootDir", action="store", type="string", dest="rootDir", default="../data/", help="Root directory containing the data")
    parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=32, help="Number of classes in the dataset")

    parser.add_option("--useTorchVisionModels", action="store_true", dest="useTorchVisionModels", default=False, help="Use pre-trained models from the torchvision library")

    # Parse command line options
    (options, args) = parser.parse_args()
    print(options)

    if options.trainModel:
        print("Training model")
        train(options)

    if options.testModel:
        print("Testing model")
