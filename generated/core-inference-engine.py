import torch

"""
CapsID: Soft‑Routed Variable‑Length Semantic IDs for Generative Recommendation (arXiv:2605.05096v1)
This module implements the core inference engine of the CapsID algorithm.
Mathematical summary:
    For each item representation x (vector of dimension D) the engine computes
    similarity scores between x and a set of learnable capsule embeddings c_i
    (i = 1 … K).  The scores are passed through a temperature‑scaled softmax
    (or hard argmax) to obtain routing weights w_i that sum to 1.  The semantic
    ID for the item is then the weighted sum of the capsule embeddings:
        id = Σ_i w_i * c_i.
    The distribution of w_i determines the effective length of the ID
    (more active capsules → longer representation), enabling variable‑length
    soft‑routed identifiers.
Key hyperparameters (default values):
    - num_capsules (int, default 32): number of capsule embeddings to consider.
    - dim (int, default 64): dimensionality of item and capsule vectors.
    - temperature (float, default 0.1): softmax temperature; lower → sharper routing.
    - use_softmax (bool, default True): whether to apply softmax (True) or hard argmax (False).
"""

def core_inference_engine(
    item_features: torch.Tensor,
    capsule_embeddings: torch.Tensor,
    num_capsules: int = 32,
    dim: int = 64,
    temperature: float = 0.1,
    use_softmax: bool = True,
) -> torch.Tensor:
    """
    Compute soft‑routed variable‑length semantic IDs for a batch of items.

    The algorithm follows the CapsID paper: for each item vector x (shape [N, D])
    we compute dot‑product similarities with K capsule embeddings c_j (shape [K, D]),
    yielding logits S_ij = x_i · c_j.  Softmax (optionally temperature‑scaled)
    converts these logits into routing weights w_ij that sum to 1 across capsules.
    The semantic ID for each item is then the weighted sum of the capsule vectors:
        id_i = Σ_j w_ij * c_j
    The distribution of w_ij determines the effective length of the ID (more
    capsules activated → longer representation).  The function returns a tensor
    of shape [N, D] containing the ID vectors.

    Parameters
    ----------
    item_features : torch.Tensor
        Item representations, shape (batch_size, dim).  dtype=float32.
    capsule_embeddings : torch.Tensor
        Learnable capsule embeddings, shape (num_capsules, dim).
    num_capsules : int, optional
        Number of capsule embeddings to consider. Must be <= capsule_embeddings.shape[0].
        Default is 32.
    dim : int, optional
        Dimensionality of the item and capsule vectors. Must match
        `item_features.shape[1]` and `capsule_embeddings.shape[1]`. Default is 64.
    temperature : float, optional
        Temperature for the softmax; lower values produce sharper routing
        (more peaked weights). Default is 0.1.
    use_softmax : bool, optional
        If True (default) apply temperature‑scaled softmax; if False, use
        argmax (hard routing) which yields a one‑hot ID (all weight on the
        highest‑scoring capsule). Default is True.

    Returns
    -------
    torch.Tensor
        Semantic ID vectors, shape (batch_size, dim).  dtype=float32.
    """
    # Validate input dimensions
    if item_features.dim() != 2:
        raise ValueError("item_features must be a 2D tensor (batch, dim)")
    batch_size, feat_dim = item_features.shape
    if feat_dim != dim:
        raise ValueError(f"item_features dimension {feat_dim} does not match dim {dim}")

    if capsule_embeddings.dim() != 2:
        raise ValueError("capsule_embeddings must be a 2D tensor (num_capsules, dim)")
    if capsule_embeddings.shape[1] != dim:
        raise ValueError(f"capsule_embeddings dimension {capsule_embeddings.shape[1]} does not match dim {dim}")

    K = capsule_embeddings.shape[0]
    if num_capsules > K:
        raise ValueError(f"num_capsules ({num_capsules}) cannot exceed number of capsules ({K})")

    # Select the capsules to use (first `num_capsules` or all if not specified)
    caps = capsule_embeddings[:num_capsules]  # shape (K_sel, dim)

    # Compute logits: dot product between each item and each capsule
    # logits shape (batch_size, K_sel)
    logits = torch.matmul(item_features, caps.t())

    # Apply temperature‑scaled softmax if requested
    if use_softmax:
        scaled_logits = logits / temperature
        weights = torch.softmax(scaled_logits, dim=1)  # (batch_size, K_sel)
    else:
        # Hard routing: select the capsule with highest logit for each item
        max_idx = torch.argmax(logits, dim=1)  # (batch_size,)
        weights = torch.zeros_like(logits)
        weights[torch.arange(batch_size), max_idx] = 1.0

    # Weighted sum of capsule embeddings to obtain semantic ID
    # ids shape (batch_size, dim)
    ids = torch.matmul(weights, caps)

    return ids


if __name__ == "__main__":
    # Concrete usage example
    batch_size = 4
    dim = 64
    num_capsules = 32

    # Random item features (e.g., embeddings from a recommendation model)
    item_feat = torch.randn(batch_size, dim)

    # Random capsule embeddings (in practice these would be learned)
    capsule_emb = torch.randn(num_capsules, dim)

    # Run the core inference engine
    with torch.no_grad():
        semantic_ids = core_inference_engine(
            item_feat,
            capsule_emb,
            num_capsules=num_capsules,
            dim=dim,
            temperature=0.05,
            use_softmax=True,
        )

    print("Item features shape:", item_feat.shape)
    print("Capsule embeddings shape:", capsule_emb.shape)
    print("Semantic IDs shape:", semantic_ids.shape)
    print("First ID vector (first 5 values):", semantic_ids[0, :5])