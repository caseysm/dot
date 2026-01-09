"""Tests for Sinkhorn optimal transport."""

import pytest
import torch


def test_sinkhorn_import():
    """Test that dot package can be imported."""
    import dot
    assert hasattr(dot, 'sinkhorn')
    assert hasattr(dot, 'sinkhorn_from_scores')
    assert hasattr(dot, 'Sinkhorn')
    assert hasattr(dot, 'SinkhornFromScores')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda():
    """Test Sinkhorn on CUDA."""
    import dot

    # Create random cost matrix (square for this implementation)
    B, n = 2, 10
    cost = torch.rand(B, n, n, device='cuda')

    result = dot.sinkhorn(cost, reg=1.0)

    # Check output shapes
    assert result.transport_plan.shape == (B, n, n)
    assert result.cost.shape == (B,)

    # Check transport plan is valid (rows and columns sum to 1 - doubly stochastic)
    row_sums = result.transport_plan.sum(dim=-1)
    col_sums = result.transport_plan.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


def test_sinkhorn_cpu():
    """Test Sinkhorn on CPU."""
    import dot

    # Create random cost matrix
    B, n = 2, 10
    cost = torch.rand(B, n, n)

    result = dot.sinkhorn(cost, reg=1.0)

    # Check output shapes
    assert result.transport_plan.shape == (B, n, n)
    assert result.cost.shape == (B,)

    # Check doubly-stochastic property
    row_sums = result.transport_plan.sum(dim=-1)
    col_sums = result.transport_plan.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_backward_cuda():
    """Test Sinkhorn backward pass on CUDA."""
    import dot

    B, n = 2, 8
    cost = torch.rand(B, n, n, device='cuda', requires_grad=True)

    result = dot.sinkhorn(cost, reg=1.0)
    loss = result.cost.sum()
    loss.backward()

    assert cost.grad is not None
    assert cost.grad.shape == cost.shape


def test_sinkhorn_backward_cpu():
    """Test Sinkhorn backward pass on CPU."""
    import dot

    B, n = 2, 8
    cost = torch.rand(B, n, n, requires_grad=True)

    result = dot.sinkhorn(cost, reg=1.0)
    loss = result.cost.sum()
    loss.backward()

    assert cost.grad is not None
    assert cost.grad.shape == cost.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_backward_modes_cuda():
    """Test both backward modes produce gradients on CUDA."""
    import dot

    B, n = 2, 8
    cost = torch.rand(B, n, n, device='cuda')

    # Test unrolled
    cost_unrolled = cost.clone().requires_grad_(True)
    result_unrolled = dot.sinkhorn(cost_unrolled, reg=1.0, backward_mode='unrolled')
    result_unrolled.cost.sum().backward()

    # Test implicit
    cost_implicit = cost.clone().requires_grad_(True)
    result_implicit = dot.sinkhorn(cost_implicit, reg=1.0, backward_mode='implicit')
    result_implicit.cost.sum().backward()

    assert cost_unrolled.grad is not None
    assert cost_implicit.grad is not None

    # Both should produce similar transport plans
    assert torch.allclose(result_unrolled.transport_plan, result_implicit.transport_plan, atol=1e-4)


def test_sinkhorn_module():
    """Test Sinkhorn nn.Module wrapper."""
    import dot

    sinkhorn = dot.Sinkhorn(reg=0.5, n_iters=50)

    B, n = 1, 5
    cost = torch.rand(B, n, n)

    result = sinkhorn(cost)
    assert result.transport_plan.shape == (B, n, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_from_scores_cuda():
    """Test sinkhorn_from_scores on CUDA."""
    import dot

    B, n = 2, 10
    log_alpha = torch.randn(B, n, n, device='cuda')

    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0)

    assert P.shape == (B, n, n)

    # Check doubly-stochastic
    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


def test_sinkhorn_from_scores_cpu():
    """Test sinkhorn_from_scores on CPU."""
    import dot

    B, n = 2, 10
    log_alpha = torch.randn(B, n, n)

    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0)

    assert P.shape == (B, n, n)

    # Check doubly-stochastic
    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_from_scores_backward_cuda():
    """Test sinkhorn_from_scores backward on CUDA."""
    import dot

    B, n = 2, 8
    log_alpha = torch.randn(B, n, n, device='cuda', requires_grad=True)

    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0)
    loss = P.sum()
    loss.backward()

    assert log_alpha.grad is not None
    assert log_alpha.grad.shape == log_alpha.shape


def test_unbatched_input():
    """Test that unbatched inputs work correctly."""
    import dot

    n = 10
    cost = torch.rand(n, n)

    result = dot.sinkhorn(cost, reg=1.0)

    assert result.transport_plan.shape == (n, n)
    assert result.cost.shape == ()

    # Check doubly-stochastic
    row_sums = result.transport_plan.sum(dim=-1)
    col_sums = result.transport_plan.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)
