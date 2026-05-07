"""
Quantized Probabilistic AI for Gear Fault Diagnosis in Motor Drives

This module implements the core algorithm described in the paper "Quantized Probabilistic AI for Gear Fault Diagnosis in Motor Drives" (http://arxiv.org/abs/2605.05032v1).
The mathematical idea behind this algorithm is to use quantization-aware training for INT8 models to reduce latency via quantization.
The key hyperparameters are:
- num_classes (int): The number of gear fault classes. Default value: 5
- num_features (int): The number of input features. Default value: 10
- batch_size (int): The batch size for training. Default value: 32
- learning_rate (float): The learning rate for the optimizer. Default value: 0.001
- num_epochs (int): The number of training epochs. Default value: 100

Author: Samsung R&D
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GearFaultDataset(Dataset):
    """
    A dataset class for gear fault data.

    Parameters
    ----------
    features : np.ndarray
        The input features.
    labels : np.ndarray
        The corresponding labels.
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

class QuantizedProbabilisticAI(nn.Module):
    """
    A PyTorch module for quantized probabilistic AI.

    Parameters
    ----------
    num_classes : int
        The number of gear fault classes.
    num_features : int
        The number of input features.
    """
    def __init__(self, num_classes=5, num_features=10):
        super(QuantizedProbabilisticAI, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)  # fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # fully connected layer
        self.quant = torch.quantization Quantize(  # quantization module
            num_bits=8,
            symmetric=True,
            narrow_range=False,
            rounding='nearest'
        )

    def forward(self, x):
        # apply quantization to the input
        x = self.quant(x)
        # apply activation functions
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, dataloader, optimizer, criterion, num_epochs=100):
    """
    Train the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.
    device : torch.device
        The device to use for training.
    dataloader : DataLoader
        The data loader.
    optimizer : Optimizer
        The optimizer to use.
    criterion : nn.Module
        The loss function.
    num_epochs : int
        The number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d, Loss: %.3f' % (epoch+1, running_loss / (i+1)))

def core_inference_engine(model, device, inputs):
    """
    The core inference engine.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.
    device : torch.device
        The device to use for inference.
    inputs : torch.tensor
        The input tensor.

    Returns
    -------
    outputs : torch.tensor
        The output tensor.
    """
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        return outputs

if __name__ == "__main__":
    # create a sample dataset
    features = np.random.rand(100, 10)
    labels = np.random.randint(0, 5, 100)
    dataset = GearFaultDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # initialize the model, device, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantizedProbabilisticAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train the model
    train(model, device, dataloader, optimizer, criterion)

    # create a sample input tensor
    inputs = torch.randn(1, 10)

    # run the core inference engine
    outputs = core_inference_engine(model, device, inputs)
    print(outputs)