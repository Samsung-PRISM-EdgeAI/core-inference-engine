import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

"""
UniPool: A Globally Shared Expert Pool for Mixture-of-Experts
Paper URL: http://arxiv.org/abs/2605.06665v1

Mathematical Idea:
UniPool addresses the parameter inefficiency of traditional Mixture-of-Experts (MoE)
models by proposing a globally shared expert pool. Instead of each MoE layer
having its own dedicated set of experts, all MoE layers in a model draw from a
single, shared pool of experts. This significantly reduces the total parameter
count.

To enable effective sharing and balanced utilization of this global pool, UniPool
introduces two key mechanisms:
1.  NormRouter: A routing mechanism that normalizes the router's output logits
    by their L2-norm before applying the softmax function. This helps in
    stabilizing routing decisions and potentially improving expert utilization.
2.  Pool-level Auxiliary Loss: An auxiliary loss function applied to the entire
    expert pool (rather than per-layer) to encourage balanced expert usage across
    all tokens processed by any MoE layer in the model. This loss is formulated
    as `L_aux = λ * sum_i (P_i * C_i)`, where `P_i` is the fraction of routing
    probability assigned to expert `i`, and `C_i` is the fraction of tokens
    actually dispatched to expert `i` (considering