"""
Bidirectional Softmax for soft sequence matching.

Computes soft matching without monotonic constraint:
    out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))

The geometric mean of row and column softmax provides a symmetric soft assignment
that considers both row and column competition, useful for unordered matching.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from . import _ops


class _BidirectionalSoftmaxFunction(torch.autograd.Function):
    """Autograd function for bidirectional softmax."""

    @staticmethod
    def forward(
        ctx,
        sim_matrix: torch.Tensor,
        tau: float,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of bidirectional softmax."""
        results = _ops.bidirectional_softmax(sim_matrix, tau, lengths)
        output, row_softmax, col_softmax = results[0], results[1], results[2]

        ctx.save_for_backward(sim_matrix, output, row_softmax, col_softmax, lengths)
        ctx.tau = tau
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using stored softmax buffers."""
        sim_matrix, output, row_softmax, col_softmax, lengths = ctx.saved_tensors
        tau = ctx.tau

        results = _ops.bidirectional_softmax_backward(
            sim_matrix, output, grad_output.contiguous(),
            row_softmax, col_softmax, tau, lengths
        )
        grad_sim, grad_tau = results[0], results[1]

        return grad_sim, None, None


def _make_lengths(B: int, L1: int, L2: int, device: torch.device) -> torch.Tensor:
    """Create default lengths tensor."""
    lengths = torch.empty(B, 2, dtype=torch.int32, device=device)
    lengths[:, 0] = L1
    lengths[:, 1] = L2
    return lengths


def bidirectional_softmax(
    sim_matrix: torch.Tensor,
    tau: float = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute bidirectional softmax for soft sequence matching.

    Computes geometric mean of row and column softmax:
        out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))

    This creates a symmetric soft assignment considering both row and column
    competition, useful for unordered sequence matching.

    Args:
        sim_matrix: Similarity scores of shape (B, L1, L2) or (L1, L2)
        tau: Temperature parameter (default: 1.0). Lower = sharper matching.
        lengths: Optional [B, 2] tensor of (L1, L2) lengths per batch.
                 If None, uses full dimensions.

    Returns:
        Soft matching matrix of same shape as input, values in [0, 1].

    Example:
        >>> sim = torch.randn(2, 10, 15)  # Batch of 2, matching 10 to 15 elements
        >>> P = bidirectional_softmax(sim, tau=0.5)
        >>> P.shape
        torch.Size([2, 10, 15])
    """
    # Handle unbatched input
    unbatched = sim_matrix.dim() == 2
    if unbatched:
        sim_matrix = sim_matrix.unsqueeze(0)

    B = sim_matrix.size(0)
    L1 = sim_matrix.size(1)
    L2 = sim_matrix.size(2)

    # Create default lengths if not provided
    if lengths is None:
        lengths = _make_lengths(B, L1, L2, sim_matrix.device)

    result = _BidirectionalSoftmaxFunction.apply(sim_matrix, tau, lengths)

    if unbatched:
        result = result.squeeze(0)

    return result


def bidirectional_softmax_with_lengths(
    sim_matrix: torch.Tensor,
    lengths: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Bidirectional softmax with explicit variable-length support.

    Same as bidirectional_softmax but with lengths as required parameter.

    Args:
        sim_matrix: Similarity scores [B, max_L1, max_L2]
        lengths: Sequence lengths [B, 2] containing (L1, L2) per batch
        tau: Temperature parameter

    Returns:
        Soft matching matrix [B, max_L1, max_L2]
    """
    return bidirectional_softmax(sim_matrix, tau=tau, lengths=lengths)


class BidirectionalSoftmax(nn.Module):
    """Neural network module for bidirectional softmax.

    Computes soft matching without monotonic constraint using geometric
    mean of row and column softmax.

    Args:
        tau: Temperature parameter (default: 1.0)

    Example:
        >>> soft_match = BidirectionalSoftmax(tau=0.5)
        >>> sim = torch.randn(2, 10, 15)
        >>> P = soft_match(sim)
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        sim_matrix: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft matching matrix.

        Args:
            sim_matrix: Similarity scores [B, L1, L2] or [L1, L2]
            lengths: Optional [B, 2] tensor of lengths per batch

        Returns:
            Soft matching matrix of same shape as input
        """
        return bidirectional_softmax(sim_matrix, tau=self.tau, lengths=lengths)
