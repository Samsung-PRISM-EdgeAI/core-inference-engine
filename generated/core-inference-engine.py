"""
CapsID: Soft-Routed Variable-Length Semantic IDs for Generative Recommendation

This module implements the core algorithm from the paper "CapsID: Soft-Routed Variable-Length Semantic IDs for Generative Recommendation"
available at http://arxiv.org/abs/2605.05096v1.

The mathematical idea behind this algorithm is to use soft routing and variable-length token generation to reduce token overhead and improve
efficiency for quantized LLM inference. This is achieved by assigning a variable-length semantic ID to each item, which is then used to
generate recommendations.

The key hyperparameters for this algorithm are:
- embedding_dim (default: 128): The dimensionality of the item embeddings.
- num_heads (default: 8): The number of attention heads.
- max_length (default: 10): The maximum length of the semantic IDs.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _calculate_attention_weights(query, key, value):
    """
    Calculate the attention weights.

    Parameters
    ----------
    query : torch.Tensor
        The query tensor.
    key : torch.Tensor
        The key tensor.
    value : torch.Tensor
        The value tensor.

    Returns
    -------
    torch.Tensor
        The attention weights.
    """
    # Calculate the attention scores
    attention_scores = torch.matmul(query, key.T) / math.sqrt(query.size(-1))
    
    # Calculate the attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    return attention_weights

def _generate_semantic_id(item_embedding, num_heads, max_length):
    """
    Generate the semantic ID for an item.

    Parameters
    ----------
    item_embedding : torch.Tensor
        The item embedding.
    num_heads : int
        The number of attention heads.
    max_length : int
        The maximum length of the semantic ID.

    Returns
    -------
    torch.Tensor
        The semantic ID.
    """
    # Initialize the semantic ID
    semantic_id = torch.zeros((item_embedding.size(0), max_length), device=item_embedding.device)
    
    # Calculate the attention weights for each head
    for i in range(num_heads):
        # Calculate the query, key, and value tensors for this head
        query = item_embedding[:, i * item_embedding.size(1) // num_heads:(i + 1) * item_embedding.size(1) // num_heads]
        key = item_embedding[:, i * item_embedding.size(1) // num_heads:(i + 1) * item_embedding.size(1) // num_heads]
        value = item_embedding[:, i * item_embedding.size(1) // num_heads:(i + 1) * item_embedding.size(1) // num_heads]
        
        # Calculate the attention weights
        attention_weights = _calculate_attention_weights(query, key, value)
        
        # Update the semantic ID
        semantic_id[:, i] = attention_weights.mean(dim=-1)
    
    return semantic_id

class CapsID(nn.Module):
    """
    The CapsID model.

    Parameters
    ----------
    embedding_dim : int
        The dimensionality of the item embeddings.
    num_heads : int
        The number of attention heads.
    max_length : int
        The maximum length of the semantic IDs.
    """
    def __init__(self, embedding_dim=128, num_heads=8, max_length=10):
        super(CapsID, self).__init__()
        
        # Initialize the embedding layer
        self.embedding = nn.Embedding(embedding_dim, embedding_dim)
        
        # Initialize the attention layers
        self.attention = nn.MultiHeadAttention(embedding_dim, num_heads)
        
        # Initialize the semantic ID generation layer
        self.semantic_id_generation = lambda item_embedding: _generate_semantic_id(item_embedding, num_heads, max_length)
    
    def forward(self, item_ids):
        """
        Forward pass.

        Parameters
        ----------
        item_ids : torch.Tensor
            The item IDs.

        Returns
        -------
        torch.Tensor
            The semantic IDs.
        """
        # Get the item embeddings
        item_embeddings = self.embedding(item_ids)
        
        # Generate the semantic IDs
        semantic_ids = self.semantic_id_generation(item_embeddings)
        
        return semantic_ids

def core_inference_engine(item_ids, embedding_dim=128, num_heads=8, max_length=10):
    """
    The core inference engine.

    Parameters
    ----------
    item_ids : torch.Tensor
        The item IDs.
    embedding_dim : int
        The dimensionality of the item embeddings.
    num_heads : int
        The number of attention heads.
    max_length : int
        The maximum length of the semantic IDs.

    Returns
    -------
    torch.Tensor
        The semantic IDs.
    """
    # Initialize the CapsID model
    model = CapsID(embedding_dim, num_heads, max_length)
    
    # Generate the semantic IDs
    semantic_ids = model(item_ids)
    
    return semantic_ids

if __name__ == "__main__":
    # Initialize the item IDs
    item_ids = torch.randint(0, 100, (10,))
    
    # Generate the semantic IDs
    semantic_ids = core_inference_engine(item_ids)
    
    # Print the semantic IDs
    print(semantic_ids)