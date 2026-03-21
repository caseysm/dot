"""
Differentiable Optimal Transport (DOT) - PyTorch operators for optimal transport.

High-Level API:
    # Sinkhorn optimal transport from cost matrix
    result = dot.sinkhorn(cost_matrix, reg=1.0)
    P = result.transport_plan  # (B, n, n) doubly-stochastic
    cost = result.cost  # (B,) transport cost

    # Soft permutation from similarity scores
    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0)

    # Bidirectional softmax for unordered matching
    P = dot.bidirectional_softmax(sim_matrix, tau=1.0)

Module API (nn.Module wrappers):
    sinkhorn = dot.Sinkhorn(reg=1.0, n_iters=20)
    result = sinkhorn(cost_matrix)

    soft_perm = dot.SinkhornFromScores(tau=1.0, n_iters=20)
    P = soft_perm(log_scores)

    soft_match = dot.BidirectionalSoftmax(tau=1.0)
    P = soft_match(sim_matrix)

Low-Level API:
    from dot import _ops
    P = _ops.sinkhorn(log_alpha, tau, n_iters)
    outputs = _ops.bidirectional_softmax(sim, tau, lengths)
"""

# Load extension first
from . import _ops

# High-level API - Sinkhorn
from .sinkhorn import (
    sinkhorn,
    sinkhorn_from_scores,
    spectral_preflight,
    SinkhornResult,
    Sinkhorn,
    SinkhornFromScores,
)

# High-level API - Bidirectional Softmax
from .bidirectional_softmax import (
    bidirectional_softmax,
    bidirectional_softmax_with_lengths,
    BidirectionalSoftmax,
)

__all__ = [
    # Sinkhorn API
    'sinkhorn',
    'sinkhorn_from_scores',
    'spectral_preflight',
    'SinkhornResult',
    'Sinkhorn',
    'SinkhornFromScores',
    # Bidirectional Softmax API
    'bidirectional_softmax',
    'bidirectional_softmax_with_lengths',
    'BidirectionalSoftmax',
]
