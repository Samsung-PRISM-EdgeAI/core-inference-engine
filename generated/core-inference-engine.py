import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

"""
Source Paper: An Overview of Google Brain and Its Applications
URL: https://www.semanticscholar.org/paper/a17f0aede0a4e957de4d4c2e1d33e22411c727bb

Summary:
The flagged paper is a high-level survey of the Google Brain ecosystem. For Samsung's 
on-device AI stack, the most critical operational takeaway from the Brain research 
lineage is the implementation of efficient, scalable inference patterns—specifically 
the use of lightweight Attention mechanisms and Feed-Forward Networks (FFNs) that 
form the basis of the Transformer architecture.

The 'core-inference-engine' implemented here focuses on a memory-efficient, 
quantization-ready Transformer Inference Block. It implements the mathematical 
operation: 
    Output = LayerNorm(x + MultiHeadAttention(x)) 
    Output = LayerNorm(Output + FeedForward(Output))

Key Hyperparameters:
- d_model (int): Dimensionality of the input embeddings. Default: 512.
- n_heads (int): Number of attention heads for parallel representation. Default: 8.
- d_ff (int): Hidden layer size of the feed-forward network. Default: 2048.
- dropout (float): Dropout probability for regularization. Default: 0.1.
"""

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in the 
    foundational Brain/Transformer research.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize the MultiHeadAttention module.

        Parameters
        ----------
        d_model : int
            The dimension of the input embeddings.
        n_heads : int
            The number of parallel attention heads.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform forward pass of Multi-Head Attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch, seq_len, d_model).
        k : torch.Tensor
            Key tensor of shape (batch, seq_len, d_model).
        v : torch.Tensor
            Value tensor of shape (batch, seq_len, d_model).
        mask : torch.Tensor, optional
            Mask tensor to prevent attention to specific tokens.

        Returns
        -------
        torch.Tensor
            The weighted sum of values of shape (batch, seq_len, d_model).
        """
        batch_size = q.size(0)
        
        # 1. Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 3. Context aggregation
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.w_o(output)

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network used in the Transformer block.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the FFN module.

        Parameters
        ----------
        d_model : int
            Input and output dimensionality.
        d_ff : int
            Hidden layer dimensionality.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FFN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Processed tensor of shape (batch, seq_len, d_model).
        """
        return self.net(x)

def core_inference_engine(
    input_tensor: torch.Tensor, 
    weights: dict, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    The core inference engine for the edge-optimized Transformer block.
    This function implements the forward pass logic required for on-device 
    deployment, ensuring tensor shapes are maintained for downstream tasks.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input sequence embeddings of shape (batch, seq_len, d_model).
    weights : dict
        A dictionary containing the state_dict for the MultiHeadAttention 
        and FeedForwardNetwork modules.
    mask : torch.Tensor, optional
        Attention mask to ignore padding tokens.

    Returns
    -------
    torch.Tensor
        The refined feature representation of shape (batch, seq_len, d_model).
    """
    # Extract hyperparameters from weight dimensions
    # weights['mha'] state_dict is used to infer d_model
    d_model = weights['mha'].w_q.out_features
    n_heads = weights['mha'].n_heads
    d_ff = weights['ffn'].net[0].out_features
    
    # 1. Multi-Head Attention Step
    # Residual Connection: x = LayerNorm(x + Attention(x))
    attn_out = weights['mha'](input_tensor, input_tensor, input_tensor, mask)
    x = weights['norm1'](input_tensor + attn_out)
    
    # 2. Feed Forward Step
    # Residual Connection: x = LayerNorm(x + FFN(x))
    ffn_out = weights['ffn'](x)
    x = weights['norm2'](x + ffn_out)
    
    return x

if __name__ == "__main__":
    # --- Simulation of Samsung On-Device Environment ---
    torch.manual_seed(42)
    
    # Hyperparameters
    B, S, D = 2, 10, 512  # Batch=2, SeqLen=10, ModelDim=512
    H = 8                 # Heads
    D_FF = 2048           # FeedForward Dim
    
    # Initialize components to simulate a loaded model
    mha = MultiHeadAttention(D, H)
    ffn = FeedForwardNetwork(D, D_FF)
    norm1 = nn.LayerNorm(D)
    norm2 = nn.LayerNorm(D)
    
    # Pack weights into a dictionary as expected by the core_inference_engine
    model_weights = {
        'mha': mha,
        'ffn': ffn,
        'norm1': norm1,
        'norm2': norm2
    }
    
    # Create dummy input: (Batch, SeqLen, ModelDim)
    sample_input = torch.randn(B, S, D)
    
    # Optional: Create a causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0)
    
    print(f"Input Shape: {sample_input.shape}")
    
    # Execute the core inference engine
    try:
        output = core_inference_engine(
            input_tensor=sample_input, 
            weights=model_weights, 
            mask=causal_mask
        )
        print(f"Output Shape: {output.shape}")
        print("Inference successful. Tensor variance:", torch.var(output).item())
    except Exception as e:
        print(f"Inference failed: {e}")