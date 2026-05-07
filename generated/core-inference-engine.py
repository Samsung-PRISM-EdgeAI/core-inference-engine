"""
Module implementing the core-inference-engine for disease identification and fruit sorting in apple fruit using a reduced-layer CNN architecture.
Source paper: Using Machine Learning to Identify Diseases and Perform Sorting in Apple Fruit
Paper URL: https://www.semanticscholar.org/paper/0d2c38b39f73003ec23ee3e2382e319e20bd5d18
Mathematical idea: The paper proposes a reduced-layer CNN architecture to classify apple fruit into different categories based on the presence of diseases. 
The architecture is designed to be efficient and can run on handheld devices with limited resources.

Key hyperparameters and their default values:
- num_layers: 5
- num_filters: 32
- kernel_size: 3
- learning_rate: 0.001
- batch_size: 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AppleFruitDataset(Dataset):
    """
    A custom dataset class for apple fruit images.

    Parameters
    ----------
    images : torch.Tensor
        A tensor of images.
    labels : torch.Tensor
        A tensor of labels corresponding to the images.

    Returns
    -------
    image : torch.Tensor
        A single image.
    label : torch.Tensor
        The label corresponding to the image.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

class ReducedLayerCNN(nn.Module):
    """
    A reduced-layer CNN architecture for disease identification and fruit sorting in apple fruit.

    Parameters
    ----------
    num_layers : int
        The number of layers in the CNN architecture. Default is 5.
    num_filters : int
        The number of filters in each layer. Default is 32.
    kernel_size : int
        The size of the kernel in each layer. Default is 3.

    Returns
    -------
    output : torch.Tensor
        The output of the CNN architecture.
    """
    def __init__(self, num_layers=5, num_filters=32, kernel_size=3):
        super(ReducedLayerCNN, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(3, num_filters, kernel_size=kernel_size) for _ in range(num_layers)])
        self.fc = nn.Linear(num_filters * 224 * 224, 10)  # assuming input size is 224x224

    def forward(self, x):
        # iterate over each layer and apply convolution and max pooling
        for layer in self.layers:
            x = torch.relu(layer(x))  # apply convolution and relu activation
            x = nn.functional.max_pool2d(x, kernel_size=2)  # apply max pooling
        # flatten the output and apply fully connected layer
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x

def train(model, device, loader, optimizer, criterion):
    """
    Train the model on the given loader.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    device : torch.device
        The device to train on.
    loader : DataLoader
        The loader to train on.
    optimizer : optim.Optimizer
        The optimizer to use.
    criterion : nn.Module
        The criterion to use.

    Returns
    -------
    loss : float
        The average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, device, loader, criterion):
    """
    Test the model on the given loader.

    Parameters
    ----------
    model : nn.Module
        The model to test.
    device : torch.device
        The device to test on.
    loader : DataLoader
        The loader to test on.
    criterion : nn.Module
        The criterion to use.

    Returns
    -------
    loss : float
        The average loss over the epoch.
    accuracy : float
        The accuracy over the epoch.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

if __name__ == "__main__":
    # create a sample dataset
    images = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 10, (100,))
    dataset = AppleFruitDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create a model, device, optimizer, and criterion
    model = ReducedLayerCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train the model
    for epoch in range(10):
        loss = train(model, device, loader, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # test the model
    loss, accuracy = test(model, device, loader, criterion)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")