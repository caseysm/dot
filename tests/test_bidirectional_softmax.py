"""Tests for Bidirectional Softmax."""

import pytest
import torch


def test_bidirectional_softmax_import():
    """Test that bidirectional_softmax can be imported."""
    import dot
    assert hasattr(dot, 'bidirectional_softmax')
    assert hasattr(dot, 'bidirectional_softmax_with_lengths')
    assert hasattr(dot, 'BidirectionalSoftmax')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bidirectional_softmax_cuda():
    """Test bidirectional softmax on CUDA."""
    import dot

    B, L1, L2 = 2, 10, 15
    sim = torch.randn(B, L1, L2, device='cuda')

    P = dot.bidirectional_softmax(sim, tau=1.0)

    # Check output shape
    assert P.shape == (B, L1, L2)

    # Check output is non-negative (sqrt of non-negative)
    assert (P >= 0).all()

    # Check output is bounded (sqrt of product of probabilities)
    assert (P <= 1).all()


def test_bidirectional_softmax_cpu():
    """Test bidirectional softmax on CPU."""
    import dot

    B, L1, L2 = 2, 10, 15
    sim = torch.randn(B, L1, L2)

    P = dot.bidirectional_softmax(sim, tau=1.0)

    # Check output shape
    assert P.shape == (B, L1, L2)

    # Check output is non-negative
    assert (P >= 0).all()

    # Check output is bounded
    assert (P <= 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bidirectional_softmax_backward_cuda():
    """Test bidirectional softmax backward pass on CUDA."""
    import dot

    B, L1, L2 = 2, 8, 12
    sim = torch.randn(B, L1, L2, device='cuda', requires_grad=True)

    P = dot.bidirectional_softmax(sim, tau=1.0)
    loss = P.sum()
    loss.backward()

    assert sim.grad is not None
    assert sim.grad.shape == sim.shape


def test_bidirectional_softmax_backward_cpu():
    """Test bidirectional softmax backward pass on CPU."""
    import dot

    B, L1, L2 = 2, 8, 12
    sim = torch.randn(B, L1, L2, requires_grad=True)

    P = dot.bidirectional_softmax(sim, tau=1.0)
    loss = P.sum()
    loss.backward()

    assert sim.grad is not None
    assert sim.grad.shape == sim.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bidirectional_softmax_variable_lengths_cuda():
    """Test bidirectional softmax with variable lengths on CUDA."""
    import dot

    B, max_L1, max_L2 = 2, 10, 15
    sim = torch.randn(B, max_L1, max_L2, device='cuda')

    # Different lengths for each batch element
    lengths = torch.tensor([[8, 12], [6, 10]], dtype=torch.int32, device='cuda')

    P = dot.bidirectional_softmax_with_lengths(sim, lengths, tau=1.0)

    assert P.shape == (B, max_L1, max_L2)

    # Check that elements outside valid region are zero
    # Batch 0: L1=8, L2=12
    assert (P[0, 8:, :] == 0).all()
    assert (P[0, :, 12:] == 0).all()

    # Batch 1: L1=6, L2=10
    assert (P[1, 6:, :] == 0).all()
    assert (P[1, :, 10:] == 0).all()


def test_bidirectional_softmax_variable_lengths_cpu():
    """Test bidirectional softmax with variable lengths on CPU."""
    import dot

    B, max_L1, max_L2 = 2, 10, 15
    sim = torch.randn(B, max_L1, max_L2)

    # Different lengths for each batch element
    lengths = torch.tensor([[8, 12], [6, 10]], dtype=torch.int32)

    P = dot.bidirectional_softmax_with_lengths(sim, lengths, tau=1.0)

    assert P.shape == (B, max_L1, max_L2)

    # Check that elements outside valid region are zero
    assert (P[0, 8:, :] == 0).all()
    assert (P[0, :, 12:] == 0).all()
    assert (P[1, 6:, :] == 0).all()
    assert (P[1, :, 10:] == 0).all()


def test_bidirectional_softmax_module():
    """Test BidirectionalSoftmax nn.Module wrapper."""
    import dot

    soft_match = dot.BidirectionalSoftmax(tau=0.5)

    B, L1, L2 = 1, 5, 7
    sim = torch.randn(B, L1, L2)

    P = soft_match(sim)
    assert P.shape == (B, L1, L2)


def test_unbatched_input():
    """Test that unbatched inputs work correctly."""
    import dot

    L1, L2 = 10, 15
    sim = torch.randn(L1, L2)

    P = dot.bidirectional_softmax(sim, tau=1.0)

    assert P.shape == (L1, L2)
    assert (P >= 0).all()
    assert (P <= 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_temperature_effect_cuda():
    """Test that temperature affects sharpness on CUDA."""
    import dot

    B, L1, L2 = 1, 5, 5
    # Create similarity with one clear maximum per row/col
    sim = torch.zeros(B, L1, L2, device='cuda')
    for i in range(min(L1, L2)):
        sim[0, i, i] = 10.0  # Strong diagonal

    # Low temperature should give sharper output
    P_sharp = dot.bidirectional_softmax(sim, tau=0.1)
    # High temperature should give softer output
    P_soft = dot.bidirectional_softmax(sim, tau=10.0)

    # Check that low temp has more mass on diagonal
    diag_sharp = P_sharp[0].diag().sum()
    diag_soft = P_soft[0].diag().sum()

    assert diag_sharp > diag_soft


def test_temperature_effect_cpu():
    """Test that temperature affects sharpness on CPU."""
    import dot

    B, L1, L2 = 1, 5, 5
    sim = torch.zeros(B, L1, L2)
    for i in range(min(L1, L2)):
        sim[0, i, i] = 10.0

    P_sharp = dot.bidirectional_softmax(sim, tau=0.1)
    P_soft = dot.bidirectional_softmax(sim, tau=10.0)

    diag_sharp = P_sharp[0].diag().sum()
    diag_soft = P_soft[0].diag().sum()

    assert diag_sharp > diag_soft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_consistency():
    """Test that CPU and CUDA produce same results."""
    import dot

    B, L1, L2 = 2, 8, 10
    sim_cpu = torch.randn(B, L1, L2)
    sim_cuda = sim_cpu.cuda()

    P_cpu = dot.bidirectional_softmax(sim_cpu, tau=1.0)
    P_cuda = dot.bidirectional_softmax(sim_cuda, tau=1.0)

    assert torch.allclose(P_cpu, P_cuda.cpu(), atol=1e-5)
