"""
Module implementing the core algorithm from the paper:
"The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity"
available at http://arxiv.org/abs/2605.06611v1

This module provides a PyTorch implementation of the attention mechanism described in the paper.
The mathematical idea behind this module is to identify the structural origin of attention sink in neural networks.
The attention sink refers to the phenomenon where the attention weights converge to a single point, causing the model to lose its ability to focus on multiple parts of the input.
The paper identifies three key factors contributing to attention sink: variance discrepancy, super neurons, and dimension disparity.
This module provides functions to calculate these factors and to apply them to the attention mechanism.

Key hyperparameters:
- epsilon (float, default=1e-6): a small value added to the denominator for numerical stability
- alpha (float, default=0.1): a scaling factor for the variance discrepancy term
- beta (float, default=0.1): a scaling factor for the super neurons term
- gamma (float, default=0.1): a scaling factor for the dimension disparity term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_variance_discrepancy(query, key):
    """
    Calculate the variance discrepancy term.

    Parameters:
    query (torch.Tensor): the query tensor
    key (torch.Tensor): the key tensor

    Returns:
    torch.Tensor: the variance discrepancy term
    """
    # Calculate the mean of the query and key tensors
    query_mean = torch.mean(query, dim=-1, keepdim=True)
    key_mean = torch.mean(key, dim=-1, keepdim=True)
    
    # Calculate the variance of the query and key tensors
    query_var = torch.var(query, dim=-1, keepdim=True)
    key_var = torch.var(key, dim=-1, keepdim=True)
    
    # Calculate the variance discrepancy term
    variance_discrepancy = torch.abs(query_var - key_var)
    
    return variance_discrepancy

def calculate_super_neurons(query, key):
    """
    Calculate the super neurons term.

    Parameters:
    query (torch.Tensor): the query tensor
    key (torch.Tensor): the key tensor

    Returns:
    torch.Tensor: the super neurons term
    """
    # Calculate the cosine similarity between the query and key tensors
    similarity = F.cosine_similarity(query, key, dim=-1, eps=1e-6)
    
    # Calculate the super neurons term
    super_neurons = torch.max(similarity, dim=-1, keepdim=True)[0]
    
    return super_neurons

def calculate_dimension_disparity(query, key):
    """
    Calculate the dimension disparity term.

    Parameters:
    query (torch.Tensor): the query tensor
    key (torch.Tensor): the key tensor

    Returns:
    torch.Tensor: the dimension disparity term
    """
    # Calculate the dimension disparity term
    dimension_disparity = torch.abs(torch.sum(query, dim=-1, keepdim=True) - torch.sum(key, dim=-1, keepdim=True))
    
    return dimension_disparity

class AttentionMechanism(nn.Module):
    """
    The attention mechanism class.

    This class provides a PyTorch implementation of the attention mechanism described in the paper.
    It takes in the query, key, and value tensors and applies the attention mechanism to them.

    Parameters:
    epsilon (float, default=1e-6): a small value added to the denominator for numerical stability
    alpha (float, default=0.1): a scaling factor for the variance discrepancy term
    beta (float, default=0.1): a scaling factor for the super neurons term
    gamma (float, default=0.1): a scaling factor for the dimension disparity term
    """
    def __init__(self, epsilon=1e-6, alpha=0.1, beta=0.1, gamma=0.1):
        super(AttentionMechanism, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, query, key, value):
        """
        Forward pass of the attention mechanism.

        Parameters:
        query (torch.Tensor): the query tensor
        key (torch.Tensor): the key tensor
        value (torch.Tensor): the value tensor

        Returns:
        torch.Tensor: the output tensor
        """
        # Calculate the variance discrepancy term
        variance_discrepancy = calculate_variance_discrepancy(query, key)
        
        # Calculate the super neurons term
        super_neurons = calculate_super_neurons(query, key)
        
        # Calculate the dimension disparity term
        dimension_disparity = calculate_dimension_disparity(query, key)
        
        # Calculate the attention weights
        attention_weights = torch.matmul(query, key.T) / torch.sqrt(torch.sum(key ** 2, dim=-1, keepdim=True) + self.epsilon)
        
        # Apply the attention mechanism
        output = torch.matmul(attention_weights, value)
        
        # Apply the variance discrepancy, super neurons, and dimension disparity terms
        output = output + self.alpha * variance_discrepancy + self.beta * super_neurons + self.gamma * dimension_disparity
        
        return output

if __name__ == "__main__":
    # Create a sample query, key, and value tensors
    query = torch.randn(1, 10, 20)
    key = torch.randn(1, 10, 20)
    value = torch.randn(1, 10, 20)
    
    # Create an instance of the attention mechanism class
    attention_mechanism = AttentionMechanism()
    
    # Apply the attention mechanism
    output = attention_mechanism(query, key, value)
    
    # Print the output tensor
    print(output)