import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
EMO: Pretraining Mixture of Experts for Emergent Modularity

Source Paper: http://arxiv.org/abs/2605.06663v1

Mathematical Idea:
This module implements a sparse Mixture-of-Experts (MoE) architecture, a neural network design
that aims for high capacity with efficient computation. Instead of a single large neural network
processing all inputs, it employs multiple smaller, specialized "expert" networks. For each
incoming input token or feature vector, a "router" (or gating network) dynamically evaluates
all available experts and selects a small, fixed number (`top_k`) of the most relevant ones.
Only these selected experts perform computations on the input. Their individual outputs are
then combined, weighted by the router's confidence scores, to produce the final output.

The core mathematical idea involves:
1.  **Gating Network:** A linear transformation followed by a softmax function to produce
    probability distributions over experts for each input token. During training, Gumbel noise
    can be added to the logits to encourage exploration and better expert specialization.
2.  **Top-K Selection:** From the router's probabilities, the `top_k` experts with the highest
    scores are chosen. Their weights are then normalized to sum to 1.
3.  **Dispatch:** Input tokens are routed to their selected experts. For efficiency and memory
    management, each expert