"""Tests for Sinkhorn optimal transport."""

import pytest
import torch


def test_sinkhorn_import():
    """Test that dot package can be imported."""
    import dot
    assert hasattr(dot, 'sinkhorn')
    assert hasattr(dot, 'Sinkhorn')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda():
    """Test Sinkhorn on CUDA."""
    import dot

    # Create random cost matrix
    B, M, N = 2, 10, 12
    cost = torch.rand(B, M, N, device='cuda')

    result = dot.sinkhorn(cost, reg=1.0)

    # Check output shapes
    assert result.transport_plan.shape == (B, M, N)
    assert result.cost.shape == (B,)

    # Check transport plan is valid (rows and columns sum to ~1/M and ~1/N)
    row_sums = result.transport_plan.sum(dim=-1)
    col_sums = result.transport_plan.sum(dim=-2)

    assert torch.allclose(row_sums, torch.full_like(row_sums, 1.0 / M), atol=1e-3)
    assert torch.allclose(col_sums, torch.full_like(col_sums, 1.0 / N), atol=1e-3)


def test_sinkhorn_cpu():
    """Test Sinkhorn on CPU."""
    import dot

    # Create random cost matrix
    B, M, N = 2, 10, 12
    cost = torch.rand(B, M, N)

    result = dot.sinkhorn(cost, reg=1.0)

    # Check output shapes
    assert result.transport_plan.shape == (B, M, N)
    assert result.cost.shape == (B,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_backward():
    """Test Sinkhorn backward pass."""
    import dot

    B, M, N = 2, 8, 10
    cost = torch.rand(B, M, N, device='cuda', requires_grad=True)

    result = dot.sinkhorn(cost, reg=1.0)
    loss = result.cost.sum()
    loss.backward()

    assert cost.grad is not None
    assert cost.grad.shape == cost.shape


def test_sinkhorn_module():
    """Test Sinkhorn nn.Module wrapper."""
    import dot

    sinkhorn = dot.Sinkhorn(reg=0.5, max_iter=50)

    B, M, N = 1, 5, 5
    cost = torch.rand(B, M, N)

    result = sinkhorn(cost)
    assert result.transport_plan.shape == (B, M, N)
