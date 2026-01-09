"""
Differentiable Optimal Transport (DOT) - PyTorch operators for optimal transport.

High-Level API:
    # Sinkhorn optimal transport from cost matrix
    result = dot.sinkhorn(cost_matrix, reg=1.0)
    P = result.transport_plan  # (B, n, n) doubly-stochastic
    cost = result.cost  # (B,) transport cost

    # Soft permutation from similarity scores
    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0)

Module API (nn.Module wrappers):
    sinkhorn = dot.Sinkhorn(reg=1.0, n_iters=20)
    result = sinkhorn(cost_matrix)

    soft_perm = dot.SinkhornFromScores(tau=1.0, n_iters=20)
    P = soft_perm(log_scores)

Low-Level API:
    from dot import _ops
    P = _ops.sinkhorn(log_alpha, tau, n_iters)
"""

# Load extension first
from . import _ops

# High-level API
from .sinkhorn import (
    sinkhorn,
    sinkhorn_from_scores,
    SinkhornResult,
    Sinkhorn,
    SinkhornFromScores,
)

__all__ = [
    # High-level API
    'sinkhorn',
    'sinkhorn_from_scores',
    'SinkhornResult',
    'Sinkhorn',
    'SinkhornFromScores',
]
