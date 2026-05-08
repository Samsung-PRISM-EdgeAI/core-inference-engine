import torch
import torch.nn as nn
import torch.nn.functional as F

"""
EMO: Pretraining Mixture of Experts for Emergent Modularity (http://arxiv.org/abs/2605.06663v1)

This module provides a production-quality Python implementation of the core inference algorithm
for the EMO (Emergent Modularity) Mixture-of-Experts (MoE) architecture.

Mathematical Idea:
The EMO MoE layer processes an input tensor by dynamically selecting and activating a subset
of its constituent "expert" networks. For each input sample in a batch, a lightweight gating
network (router) computes scores for all available experts. Based on these scores, the top-K
experts are identified. Only these K selected experts are then computed for the respective
input samples. Their individual outputs are subsequently weighted by the router's normalized
scores and combined to produce the final output for each input sample. This selective expert
activation is crucial for reducing memory footprint and computational latency, making the
architecture highly suitable for resource-constrained on-device AI applications.

Key Hyperparameters and their default values:
- input_dim (int): The dimensionality of the input features. Default: 128
- output_dim (int): The dimensionality of the output features. Default: 128
- hidden_dim (int): The hidden dimensionality within each expert network. Default: 256
- num_experts (int): The total number of expert networks available in the MoE layer. Default: 8
- k (int): The number of top experts to select and activate for each input sample. Default: 2
- bias (bool): Whether to include bias terms in the linear layers of experts and router. Default: True
"""

class EMOExpert(nn.Module):
    """
    A single expert network within the EMO Mixture-of-Experts layer.
    It's a simple two-layer feed-forward network with a GELU activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bias: bool = True):
        """
        Initializes an EMOExpert.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input features.
        hidden_dim : int
            The hidden dimensionality within the expert network.
        output_dim : int
            The dimensionality of the output features.
        bias : bool, optional
            Whether to include bias terms in the linear layers. Defaults to True.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the expert (shape: `batch_size_for_expert, input_dim`).

        Returns
        -------
        torch.Tensor
            Output tensor from the expert (shape: `batch_size_for_expert, output_dim`).
        """
        return self.fc2(self.gelu(self.fc1(x)))

class EMORouter(nn.Module):
    """
    The gating network (router) for the EMO Mixture-of-Experts layer.
    It determines the importance of each expert for a given input.
    """
    def __init__(self, input_dim: int, num_experts: int, bias: bool = True):
        """
        Initializes an EMORouter.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input features.
        num_experts : int
            The total number of expert networks.
        bias : bool, optional
            Whether to include bias terms in the linear layer. Defaults to True.
        """
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the router.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the router (shape: `batch_size, input_dim`).

        Returns
        -------
        torch.Tensor
            Logits for each expert (shape: `batch_size, num_experts`).
        """
        return self.gate(x)

class EMOLayer(nn.Module):
    """
    The core-inference-engine for the EMO