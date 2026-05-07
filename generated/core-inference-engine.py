"""
Module implementing the Human-AI Co-Mentorship in Project-Based Learning core algorithm.

Based on the paper: Human-AI Co-Mentorship in Project-Based Learning: A Case Study in Financial Forecasting
Available at: http://arxiv.org/abs/2605.05144v1

This module implements a co-mentorship approach where human and AI collaborate to achieve financial forecasting goals.
The mathematical idea behind this approach is to combine the strengths of human intuition and AI's ability to process large amounts of data.
The key hyperparameters are:
    - learning_rate (default: 0.01): the rate at which the model learns from the data
    - num_epochs (default: 100): the number of times the model sees the data
    - batch_size (default: 32): the number of data points to process at a time

Usage:
    See the example usage in the if __name__ == "__main__": block.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FinancialForecastingDataset(Dataset):
    """
    A custom dataset class for financial forecasting data.

    Parameters
    ----------
    data : numpy.ndarray
        The input data.
    labels : numpy.ndarray
        The corresponding labels.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class HumanAICoMentorshipModel(nn.Module):
    """
    A PyTorch model implementing the Human-AI Co-Mentorship approach.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    output_dim : int
        The number of output features.
    """
    def __init__(self, input_dim, output_dim):
        super(HumanAICoMentorshipModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # input layer (input_dim) -> hidden layer (128)
        self.fc2 = nn.Linear(128, output_dim)  # hidden layer (128) -> output layer (output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted output.
        """
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

def train_model(model, device, dataset, batch_size, num_epochs, learning_rate):
    """
    Train the Human-AI Co-Mentorship model.

    Parameters
    ----------
    model : HumanAICoMentorshipModel
        The model to train.
    device : torch.device
        The device to use for training.
    dataset : FinancialForecastingDataset
        The dataset to train on.
    batch_size : int
        The batch size to use.
    num_epochs : int
        The number of epochs to train for.
    learning_rate : float
        The learning rate to use.
    """
    # create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            # move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print loss
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}')

def core_inference_engine(data, labels, learning_rate=0.01, num_epochs=100, batch_size=32):
    """
    The core inference engine function.

    Parameters
    ----------
    data : numpy.ndarray
        The input data.
    labels : numpy.ndarray
        The corresponding labels.
    learning_rate : float, optional
        The learning rate to use (default: 0.01).
    num_epochs : int, optional
        The number of epochs to train for (default: 100).
    batch_size : int, optional
        The batch size to use (default: 32).

    Returns
    -------
    HumanAICoMentorshipModel
        The trained model.
    """
    # create dataset and data loader
    dataset = FinancialForecastingDataset(data, labels)

    # create model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HumanAICoMentorshipModel(input_dim=data.shape[1], output_dim=labels.shape[1])
    model.to(device)

    # train model
    train_model(model, device, dataset, batch_size, num_epochs, learning_rate)

    return model

if __name__ == "__main__":
    # create sample data
    np.random.seed(0)
    data = np.random.rand(100, 10)
    labels = np.random.rand(100, 5)

    # train model
    model = core_inference_engine(data, labels)

    # use model for prediction
    input_data = np.random.rand(1, 10)
    input_tensor = torch.from_numpy(input_data).float()
    output = model(input_tensor)
    print(f'Predicted output: {output}')