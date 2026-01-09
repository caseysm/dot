"""
Sinkhorn algorithm for optimal transport.

The Sinkhorn algorithm computes the entropy-regularized optimal transport plan
between two distributions given a cost matrix.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from . import _ops


@dataclass
class SinkhornResult:
    """Result from Sinkhorn optimal transport computation.

    Attributes:
        transport_plan: The optimal transport plan P of shape (B, M, N)
        cost: The transport cost sum(P * C) of shape (B,)
        converged: Whether the algorithm converged
    """
    transport_plan: torch.Tensor
    cost: torch.Tensor
    converged: bool = True


class _SinkhornFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn algorithm."""

    @staticmethod
    def forward(
        ctx,
        cost_matrix: torch.Tensor,
        reg: float,
        max_iter: int,
        tol: float,
        a: Optional[torch.Tensor],
        b: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of Sinkhorn algorithm."""
        result = _ops.sinkhorn_forward(cost_matrix, reg, max_iter, tol, a, b)
        transport_plan = result[0]

        ctx.save_for_backward(cost_matrix, transport_plan)
        ctx.reg = reg

        return transport_plan

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using implicit differentiation."""
        cost_matrix, transport_plan = ctx.saved_tensors
        reg = ctx.reg

        grad_cost = _ops.sinkhorn_backward(
            grad_output, cost_matrix, transport_plan, reg
        )

        return grad_cost, None, None, None, None, None


def sinkhorn(
    cost_matrix: torch.Tensor,
    reg: float = 1.0,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> SinkhornResult:
    """Compute entropy-regularized optimal transport using Sinkhorn algorithm.

    Args:
        cost_matrix: Cost matrix of shape (B, M, N) or (M, N)
        reg: Entropic regularization strength (default: 1.0)
        a: Source distribution of shape (B, M) or (M,). Defaults to uniform.
        b: Target distribution of shape (B, N) or (N,). Defaults to uniform.
        max_iter: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance

    Returns:
        SinkhornResult containing transport_plan, cost, and convergence status
    """
    # Handle unbatched input
    unbatched = cost_matrix.dim() == 2
    if unbatched:
        cost_matrix = cost_matrix.unsqueeze(0)
        if a is not None:
            a = a.unsqueeze(0)
        if b is not None:
            b = b.unsqueeze(0)

    transport_plan = _SinkhornFunction.apply(
        cost_matrix, reg, max_iter, tol, a, b
    )

    # Compute cost
    cost = (transport_plan * cost_matrix).sum(dim=(-2, -1))

    if unbatched:
        transport_plan = transport_plan.squeeze(0)
        cost = cost.squeeze(0)

    return SinkhornResult(
        transport_plan=transport_plan,
        cost=cost,
        converged=True,
    )


class Sinkhorn(nn.Module):
    """Neural network module for Sinkhorn optimal transport.

    Args:
        reg: Entropic regularization strength
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Example:
        >>> sinkhorn = Sinkhorn(reg=1.0)
        >>> result = sinkhorn(cost_matrix)
        >>> transport_plan = result.transport_plan
    """

    def __init__(
        self,
        reg: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-9,
    ):
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol

    def forward(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> SinkhornResult:
        """Compute optimal transport plan.

        Args:
            cost_matrix: Cost matrix of shape (B, M, N) or (M, N)
            a: Source distribution (optional, defaults to uniform)
            b: Target distribution (optional, defaults to uniform)

        Returns:
            SinkhornResult with transport plan and cost
        """
        return sinkhorn(
            cost_matrix,
            reg=self.reg,
            a=a,
            b=b,
            max_iter=self.max_iter,
            tol=self.tol,
        )
