"""
Differentiable Optimal Transport (DOT) - PyTorch operators for optimal transport.

High-Level API:
    # Sinkhorn algorithm
    result = dot.sinkhorn(cost_matrix, reg=1.0)

    # Get transport plan
    P = result.transport_plan

Module API (nn.Module wrappers):
    sinkhorn = dot.Sinkhorn(reg=1.0, max_iter=100)
    result = sinkhorn(cost_matrix)

Low-Level API:
    from dot import ops
    transport_plan = ops.sinkhorn_forward(cost_matrix, reg, max_iter)
"""

# Load extension first
from . import _ops

# High-level API
from .sinkhorn import (
    sinkhorn,
    SinkhornResult,
    Sinkhorn,
)

__all__ = [
    # High-level API
    'sinkhorn',
    'SinkhornResult',
    'Sinkhorn',
]
