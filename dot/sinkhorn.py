"""
Sinkhorn algorithm for optimal transport and soft permutation matrices.

The Sinkhorn-Knopp algorithm produces doubly-stochastic matrices (soft permutations)
from log-space scores. For optimal transport, use negative cost as input.

Two backward pass implementations:
1. Unrolled: Differentiates through all iterations (exact for finite iterations)
2. Implicit: Uses implicit function theorem at convergence (memory efficient)
"""

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn

from . import _ops


@dataclass
class SinkhornResult:
    """Result from Sinkhorn computation.

    Attributes:
        transport_plan: Doubly-stochastic matrix P of shape (B, n, n)
        cost: Transport cost sum(P * C) of shape (B,) if cost_matrix provided
        converged: Whether the algorithm converged
    """
    transport_plan: torch.Tensor
    cost: Optional[torch.Tensor] = None
    converged: bool = True


class _SinkhornUnrolledFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with unrolled backward pass."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float,
        n_iters: int,
    ) -> torch.Tensor:
        """Forward pass of Sinkhorn algorithm."""
        P = _ops.sinkhorn(log_alpha, tau, n_iters)
        ctx.save_for_backward(log_alpha)
        ctx.tau = tau
        ctx.n_iters = n_iters
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        """Backward pass using unrolled differentiation."""
        log_alpha, = ctx.saved_tensors
        tau = ctx.tau
        n_iters = ctx.n_iters

        # Call forward+backward together (re-computes forward with intermediates)
        _, grad_log_alpha, _ = _ops.sinkhorn_with_grads_unrolled(
            log_alpha, grad_P.contiguous(), tau, n_iters
        )

        return grad_log_alpha, None, None


class _SinkhornImplicitFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with implicit backward pass."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float,
        n_iters: int,
        backward_iters: int,
    ) -> torch.Tensor:
        """Forward pass of Sinkhorn algorithm."""
        P = _ops.sinkhorn(log_alpha, tau, n_iters)
        ctx.save_for_backward(log_alpha, P)
        ctx.tau = tau
        ctx.n_iters = n_iters
        ctx.backward_iters = backward_iters
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        """Backward pass using implicit differentiation."""
        log_alpha, P = ctx.saved_tensors
        tau = ctx.tau
        n_iters = ctx.n_iters
        backward_iters = ctx.backward_iters

        _, grad_log_alpha, _ = _ops.sinkhorn_with_grads_implicit(
            log_alpha, grad_P.contiguous(), tau, n_iters, backward_iters
        )

        return grad_log_alpha, None, None, None


def sinkhorn(
    cost_matrix: torch.Tensor,
    reg: float = 1.0,
    n_iters: int = 20,
    backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
    backward_iters: Optional[int] = None,
) -> SinkhornResult:
    """Compute entropy-regularized optimal transport using Sinkhorn algorithm.

    Produces a doubly-stochastic matrix (soft permutation) from a cost matrix.

    Args:
        cost_matrix: Cost matrix of shape (B, n, n) or (n, n)
        reg: Regularization strength / temperature (default: 1.0)
        n_iters: Number of Sinkhorn iterations (default: 20)
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)

    Returns:
        SinkhornResult containing transport_plan and cost
    """
    # Handle unbatched input
    unbatched = cost_matrix.dim() == 2
    if unbatched:
        cost_matrix = cost_matrix.unsqueeze(0)

    # Convert cost to log_alpha: use -C/tau as input scores
    # The kernel internally divides by tau, so we just negate the cost
    log_alpha = -cost_matrix

    if backward_iters is None:
        backward_iters = n_iters

    if backward_mode == 'unrolled':
        transport_plan = _SinkhornUnrolledFunction.apply(log_alpha, reg, n_iters)
    else:
        transport_plan = _SinkhornImplicitFunction.apply(log_alpha, reg, n_iters, backward_iters)

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


def sinkhorn_from_scores(
    log_alpha: torch.Tensor,
    tau: float = 1.0,
    n_iters: int = 20,
    backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
    backward_iters: Optional[int] = None,
    return_log: bool = False,
) -> torch.Tensor:
    """Compute doubly-stochastic matrix directly from log-space scores.

    This is the low-level interface that works directly with log_alpha scores.
    Use this when you have similarity scores or logits, not costs.

    Args:
        log_alpha: Log-space scores of shape (B, n, n) or (n, n)
        tau: Temperature parameter (default: 1.0)
        n_iters: Number of Sinkhorn iterations (default: 20)
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)
        return_log: If True, return log(P) instead of P

    Returns:
        Doubly-stochastic matrix P or log(P) of same shape as input
    """
    # Handle unbatched input
    unbatched = log_alpha.dim() == 2
    if unbatched:
        log_alpha = log_alpha.unsqueeze(0)

    if backward_iters is None:
        backward_iters = n_iters

    if return_log:
        # No autograd for log version (use for inference only)
        result = _ops.sinkhorn_log(log_alpha, tau, n_iters)
    elif backward_mode == 'unrolled':
        result = _SinkhornUnrolledFunction.apply(log_alpha, tau, n_iters)
    else:
        result = _SinkhornImplicitFunction.apply(log_alpha, tau, n_iters, backward_iters)

    if unbatched:
        result = result.squeeze(0)

    return result


class Sinkhorn(nn.Module):
    """Neural network module for Sinkhorn optimal transport.

    Args:
        reg: Regularization strength / temperature
        n_iters: Number of Sinkhorn iterations
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)

    Example:
        >>> sinkhorn = Sinkhorn(reg=1.0)
        >>> result = sinkhorn(cost_matrix)
        >>> transport_plan = result.transport_plan
    """

    def __init__(
        self,
        reg: float = 1.0,
        n_iters: int = 20,
        backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
        backward_iters: Optional[int] = None,
    ):
        super().__init__()
        self.reg = reg
        self.n_iters = n_iters
        self.backward_mode = backward_mode
        self.backward_iters = backward_iters

    def forward(self, cost_matrix: torch.Tensor) -> SinkhornResult:
        """Compute optimal transport plan.

        Args:
            cost_matrix: Cost matrix of shape (B, n, n) or (n, n)

        Returns:
            SinkhornResult with transport plan and cost
        """
        return sinkhorn(
            cost_matrix,
            reg=self.reg,
            n_iters=self.n_iters,
            backward_mode=self.backward_mode,
            backward_iters=self.backward_iters,
        )


class SinkhornFromScores(nn.Module):
    """Neural network module for soft permutation from scores.

    This module takes log-space similarity scores and produces
    doubly-stochastic matrices (soft permutations).

    Args:
        tau: Temperature parameter
        n_iters: Number of Sinkhorn iterations
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)

    Example:
        >>> soft_perm = SinkhornFromScores(tau=1.0)
        >>> P = soft_perm(log_scores)  # [B, n, n] doubly-stochastic
    """

    def __init__(
        self,
        tau: float = 1.0,
        n_iters: int = 20,
        backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
        backward_iters: Optional[int] = None,
    ):
        super().__init__()
        self.tau = tau
        self.n_iters = n_iters
        self.backward_mode = backward_mode
        self.backward_iters = backward_iters

    def forward(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Compute soft permutation matrix.

        Args:
            log_alpha: Log-space scores of shape (B, n, n) or (n, n)

        Returns:
            Doubly-stochastic matrix P of same shape
        """
        return sinkhorn_from_scores(
            log_alpha,
            tau=self.tau,
            n_iters=self.n_iters,
            backward_mode=self.backward_mode,
            backward_iters=self.backward_iters,
        )
