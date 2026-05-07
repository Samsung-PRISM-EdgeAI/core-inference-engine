"""
Module implementing Adaptive Inverted-Index Routing for Granular Mixtures-of-Experts.
Based on the paper "Adaptive Inverted-Index Routing for Granular Mixtures-of-Experts" (http://arxiv.org/abs/2605.04952v1).

The mathematical idea behind this paper is to improve the efficiency of Mixture-of-Experts (MoE) models by introducing a routing architecture.
In a MoE model, each input is routed to a specific expert based on a gating function. The routing architecture in this paper uses an inverted index
to efficiently store and retrieve the routing information. The inverted index is a data structure that maps each expert to the inputs that are routed to it.
The key hyperparameters in this implementation are:
- num_experts (int): The number of experts in the MoE model. Default value: 10.
- embedding_dim (int): The dimensionality of the input embeddings. Default value: 128.
- hidden_dim (int): The dimensionality of the hidden layer in the gating function. Default value: 128.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInvertedIndexRouting(nn.Module):
    """
    Implements Adaptive Inverted-Index Routing for Granular Mixtures-of-Experts.

    Parameters
    ----------
    num_experts : int
        The number of experts in the MoE model.
    embedding_dim : int
        The dimensionality of the input embeddings.
    hidden_dim : int
        The dimensionality of the hidden layer in the gating function.

    Returns
    -------
    torch.Tensor
        The routed output.
    """
    def __init__(self, num_experts=10, embedding_dim=128, hidden_dim=128):
        super(AdaptiveInvertedIndexRouting, self).__init__()
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Initialize the inverted index
        self.inverted_index = torch.zeros((num_experts, embedding_dim))

    def forward(self, input_embeddings):
        """
        Routes the input embeddings to the experts using the inverted index.

        Parameters
        ----------
        input_embeddings : torch.Tensor
            The input embeddings.

        Returns
        -------
        torch.Tensor
            The routed output.
        """
        # Compute the similarity between the input embeddings and the inverted index
        # This is done using a dot product, which is equivalent to a cosine similarity with a normalization step
        similarities = F.relu(torch.matmul(input_embeddings, self.inverted_index.T))  # (batch_size, num_experts)
        # Normalize the similarities to obtain the routing weights
        routing_weights = F.softmax(similarities, dim=1)  # (batch_size, num_experts)
        # Compute the routed output by multiplying the input embeddings with the routing weights
        routed_output = torch.matmul(routing_weights, self.inverted_index)  # (batch_size, embedding_dim)
        return routed_output

def core_inference_engine(input_embeddings, num_experts=10, embedding_dim=128, hidden_dim=128):
    """
    The core inference engine function that uses the AdaptiveInvertedIndexRouting module.

    Parameters
    ----------
    input_embeddings : torch.Tensor
        The input embeddings.
    num_experts : int, optional
        The number of experts in the MoE model. Default value: 10.
    embedding_dim : int, optional
        The dimensionality of the input embeddings. Default value: 128.
    hidden_dim : int, optional
        The dimensionality of the hidden layer in the gating function. Default value: 128.

    Returns
    -------
    torch.Tensor
        The routed output.
    """
    # Initialize the AdaptiveInvertedIndexRouting module
    routing_module = AdaptiveInvertedIndexRouting(num_experts, embedding_dim, hidden_dim)
    # Route the input embeddings using the AdaptiveInvertedIndexRouting module
    routed_output = routing_module(input_embeddings)
    return routed_output

if __name__ == "__main__":
    # Example usage
    input_embeddings = torch.randn(32, 128)  # (batch_size, embedding_dim)
    routed_output = core_inference_engine(input_embeddings)
    print(routed_output.shape)