"""Tests for Sinkhorn optimal transport."""

import pytest
import torch
from unittest import mock


def _uniform(batch: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((batch, size), 1.0 / size, device=device, dtype=dtype)


def _reference_from_scores(
    log_alpha: torch.Tensor,
    tau: float = 1.0,
    n_iters: int = 50,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    unbatched = log_alpha.dim() == 2
    if unbatched:
        log_alpha = log_alpha.unsqueeze(0)

    batch, n, m = log_alpha.shape
    if a is None:
        a = _uniform(batch, n, log_alpha.device, log_alpha.dtype)
    elif a.dim() == 1:
        a = a.unsqueeze(0)
    if b is None:
        b = _uniform(batch, m, log_alpha.device, log_alpha.dtype)
    elif b.dim() == 1:
        b = b.unsqueeze(0)

    x = log_alpha / tau
    log_a = a.log().unsqueeze(-1)
    log_b = b.log().unsqueeze(-2)
    for _ in range(n_iters):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True) + log_a
        x = x - torch.logsumexp(x, dim=-2, keepdim=True) + log_b
    p = x.exp()
    return p.squeeze(0) if unbatched else p


def _vanilla_convergence_iters(
    log_alpha: torch.Tensor,
    tau: float,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> int:
    if log_alpha.dim() == 2:
        log_alpha = log_alpha.unsqueeze(0)
    batch, n, m = log_alpha.shape
    if a is None:
        a = _uniform(batch, n, log_alpha.device, log_alpha.dtype)
    elif a.dim() == 1:
        a = a.unsqueeze(0)
    if b is None:
        b = _uniform(batch, m, log_alpha.device, log_alpha.dtype)
    elif b.dim() == 1:
        b = b.unsqueeze(0)

    kernel = log_alpha / tau
    log_a = a.log()
    log_b = b.log()
    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)
    for it in range(max_iters):
        u_next = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
        v_next = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
        if max((u_next - u).abs().max().item(), (v_next - v).abs().max().item()) < tol:
            return it + 1
        u, v = u_next, v_next
    return max_iters


def _assert_marginals(plan: torch.Tensor, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-3):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    row_sums = plan.sum(dim=-1)
    col_sums = plan.sum(dim=-2)
    assert torch.allclose(row_sums, a, atol=atol)
    assert torch.allclose(col_sums, b, atol=atol)


def test_sinkhorn_import():
    import dot

    assert hasattr(dot, 'sinkhorn')
    assert hasattr(dot, 'sinkhorn_from_scores')
    assert hasattr(dot, 'Sinkhorn')
    assert hasattr(dot, 'SinkhornFromScores')


def test_sinkhorn_cpu_uniform_square():
    import dot

    B, n = 2, 10
    cost = torch.rand(B, n, n)
    result = dot.sinkhorn(cost, reg=1.0)

    assert result.transport_plan.shape == (B, n, n)
    assert result.cost.shape == (B,)
    uniform = _uniform(B, n, cost.device, cost.dtype)
    _assert_marginals(result.transport_plan, uniform, uniform)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda_uniform_square():
    import dot

    B, n = 2, 10
    cost = torch.rand(B, n, n, device="cuda")
    result = dot.sinkhorn(cost, reg=1.0)

    uniform = _uniform(B, n, cost.device, cost.dtype)
    _assert_marginals(result.transport_plan, uniform, uniform)


def test_sinkhorn_cpu_uniform_rectangular():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m)
    result = dot.sinkhorn(cost, reg=1.0)

    assert result.transport_plan.shape == (B, n, m)
    _assert_marginals(
        result.transport_plan,
        _uniform(B, n, cost.device, cost.dtype),
        _uniform(B, m, cost.device, cost.dtype),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda_uniform_rectangular():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m, device="cuda")
    result = dot.sinkhorn(cost, reg=1.0)

    _assert_marginals(
        result.transport_plan,
        _uniform(B, n, cost.device, cost.dtype),
        _uniform(B, m, cost.device, cost.dtype),
    )


def test_explicit_uniform_matches_default():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m)
    a = _uniform(B, n, cost.device, cost.dtype)
    b = _uniform(B, m, cost.device, cost.dtype)

    default = dot.sinkhorn(cost, reg=1.0, n_iters=50)
    explicit = dot.sinkhorn(cost, reg=1.0, n_iters=50, a=a, b=b)

    assert torch.allclose(default.transport_plan, explicit.transport_plan, atol=1e-4)
    assert torch.allclose(default.cost, explicit.cost, atol=1e-4)


def test_nonuniform_marginals_rectangular_cpu():
    import dot

    B, n, m = 1, 8, 16
    cost = torch.rand(B, n, m)
    a = torch.tensor([[0.20, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]])
    b = _uniform(B, m, cost.device, cost.dtype)

    result = dot.sinkhorn(cost, reg=1.0, n_iters=100, a=a, b=b)
    _assert_marginals(result.transport_plan, a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nonuniform_marginals_rectangular_cuda():
    import dot

    B, n, m = 1, 8, 16
    cost = torch.rand(B, n, m, device="cuda")
    a = torch.tensor([[0.20, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]], device="cuda")
    b = _uniform(B, m, cost.device, cost.dtype)

    result = dot.sinkhorn(cost, reg=1.0, n_iters=100, a=a, b=b)
    _assert_marginals(result.transport_plan, a, b)


def test_reference_match_cpu():
    import dot

    B, n, m = 2, 6, 9
    cost = torch.rand(B, n, m)
    a = torch.softmax(torch.randn(B, n), dim=-1)
    b = torch.softmax(torch.randn(B, m), dim=-1)

    result = dot.sinkhorn(cost, reg=0.7, n_iters=75, a=a, b=b)
    ref = _reference_from_scores(-cost, tau=0.7, n_iters=75, a=a, b=b)

    assert torch.allclose(result.transport_plan, ref, atol=1e-4)


def test_sinkhorn_backward_cpu():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m, requires_grad=True)
    a = torch.softmax(torch.randn(B, n), dim=-1)
    b = torch.softmax(torch.randn(B, m), dim=-1)

    result = dot.sinkhorn(cost, reg=1.0, a=a, b=b)
    result.cost.sum().backward()

    assert cost.grad is not None
    assert cost.grad.shape == cost.shape
    assert a.grad is None
    assert b.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_backward_cuda():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m, device="cuda", requires_grad=True)
    a = torch.softmax(torch.randn(B, n, device="cuda"), dim=-1)
    b = torch.softmax(torch.randn(B, m, device="cuda"), dim=-1)

    result = dot.sinkhorn(cost, reg=1.0, a=a, b=b)
    result.cost.sum().backward()

    assert cost.grad is not None
    assert cost.grad.shape == cost.shape


def test_backward_modes_cpu():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m)
    a = torch.softmax(torch.randn(B, n), dim=-1)
    b = torch.softmax(torch.randn(B, m), dim=-1)

    cost_unrolled = cost.clone().requires_grad_(True)
    result_unrolled = dot.sinkhorn(cost_unrolled, reg=1.0, backward_mode="unrolled", a=a, b=b)
    result_unrolled.cost.sum().backward()

    cost_implicit = cost.clone().requires_grad_(True)
    result_implicit = dot.sinkhorn(cost_implicit, reg=1.0, backward_mode="implicit", a=a, b=b)
    result_implicit.cost.sum().backward()

    assert cost_unrolled.grad is not None
    assert cost_implicit.grad is not None
    assert torch.allclose(result_unrolled.transport_plan, result_implicit.transport_plan, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backward_modes_cuda():
    import dot

    B, n, m = 2, 8, 16
    cost = torch.rand(B, n, m, device="cuda")
    a = torch.softmax(torch.randn(B, n, device="cuda"), dim=-1)
    b = torch.softmax(torch.randn(B, m, device="cuda"), dim=-1)

    cost_unrolled = cost.clone().requires_grad_(True)
    result_unrolled = dot.sinkhorn(cost_unrolled, reg=1.0, backward_mode="unrolled", a=a, b=b)
    result_unrolled.cost.sum().backward()

    cost_implicit = cost.clone().requires_grad_(True)
    result_implicit = dot.sinkhorn(cost_implicit, reg=1.0, backward_mode="implicit", a=a, b=b)
    result_implicit.cost.sum().backward()

    assert cost_unrolled.grad is not None
    assert cost_implicit.grad is not None
    assert torch.allclose(result_unrolled.transport_plan, result_implicit.transport_plan, atol=1e-4)


def test_sinkhorn_from_scores_cpu():
    import dot

    B, n, m = 2, 8, 16
    log_alpha = torch.randn(B, n, m)
    a = torch.softmax(torch.randn(B, n), dim=-1)
    b = torch.softmax(torch.randn(B, m), dim=-1)

    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0, a=a, b=b)
    assert P.shape == (B, n, m)
    _assert_marginals(P, a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_from_scores_cuda():
    import dot

    B, n, m = 2, 8, 16
    log_alpha = torch.randn(B, n, m, device="cuda")
    a = torch.softmax(torch.randn(B, n, device="cuda"), dim=-1)
    b = torch.softmax(torch.randn(B, m, device="cuda"), dim=-1)

    P = dot.sinkhorn_from_scores(log_alpha, tau=1.0, a=a, b=b)
    assert P.shape == (B, n, m)
    _assert_marginals(P, a, b)


def test_unbatched_input():
    import dot

    n, m = 8, 16
    cost = torch.rand(n, m)
    a = torch.softmax(torch.randn(n), dim=-1)
    b = torch.softmax(torch.randn(m), dim=-1)

    result = dot.sinkhorn(cost, reg=1.0, a=a, b=b)

    assert result.transport_plan.shape == (n, m)
    assert result.cost.shape == ()
    _assert_marginals(result.transport_plan.unsqueeze(0), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rectangular_cuda_matches_cpu():
    import dot

    B, n, m = 2, 8, 16
    cost_cpu = torch.rand(B, n, m)
    a_cpu = torch.softmax(torch.randn(B, n), dim=-1)
    b_cpu = torch.softmax(torch.randn(B, m), dim=-1)

    result_cpu = dot.sinkhorn(cost_cpu, reg=1.0, a=a_cpu, b=b_cpu)
    result_cuda = dot.sinkhorn(cost_cpu.cuda(), reg=1.0, a=a_cpu.cuda(), b=b_cpu.cuda())

    assert torch.allclose(result_cpu.transport_plan, result_cuda.transport_plan.cpu(), atol=1e-3)


def test_differentiable_reg_cpu():
    import dot

    cost = torch.rand(1, 5, 5)
    reg = torch.tensor(0.1, requires_grad=True)

    result = dot.sinkhorn(cost, reg=reg, n_iters=50)
    result.cost.sum().backward()

    assert reg.grad is not None
    assert reg.grad.abs() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_differentiable_reg_cuda():
    import dot

    cost = torch.rand(1, 5, 5, device="cuda")
    reg = torch.tensor(0.1, requires_grad=True, device="cuda")

    result = dot.sinkhorn(cost, reg=reg, n_iters=50)
    result.cost.sum().backward()

    assert reg.grad is not None
    assert reg.grad.abs() > 0


def test_differentiable_reg_matches_finite_difference():
    import dot

    cost = torch.eye(5).unsqueeze(0)
    eps = torch.tensor(0.2, requires_grad=True)

    result = dot.sinkhorn(cost, reg=eps, n_iters=80)
    result.cost.sum().backward()
    analytic = eps.grad.item()

    delta = 1e-3
    plus = dot.sinkhorn(cost, reg=float(eps.detach() + delta), n_iters=80).cost.sum().item()
    minus = dot.sinkhorn(cost, reg=float(eps.detach() - delta), n_iters=80).cost.sum().item()
    finite_diff = (plus - minus) / (2 * delta)

    assert analytic > 0
    assert abs(analytic - finite_diff) < 5e-2


def test_composed_gradient_through_points_and_reg():
    import dot

    source = torch.randn(5, 3, requires_grad=True)
    target = torch.randn(5, 3)
    eps = torch.tensor(0.1, requires_grad=True)

    cost = torch.cdist(source, target).unsqueeze(0)
    result = dot.sinkhorn(cost, reg=eps, n_iters=50)
    result.cost.sum().backward()

    assert source.grad is not None
    assert eps.grad is not None


def test_momentum_matches_vanilla():
    import dot

    cost = torch.rand(1, 12, 12)
    vanilla = dot.sinkhorn(cost, reg=1.0, n_iters=80)
    momentum = dot.sinkhorn(cost, reg=1.0, n_iters=80, method="momentum", omega=1.5)

    assert torch.allclose(vanilla.transport_plan, momentum.transport_plan, atol=1e-4)


def test_momentum_auto_omega_runs():
    import dot

    cost = torch.rand(1, 12, 12)
    result = dot.sinkhorn(cost, reg=0.1, n_iters=80, method="momentum")

    assert result.transport_plan.shape == (1, 12, 12)
    assert result.n_iters_used is not None


def test_momentum_uses_fewer_iterations_than_vanilla():
    import dot

    n = 32
    grid = torch.linspace(-1.0, 1.0, n)
    cost = (grid[:, None] - grid[None, :]).abs().pow(2).unsqueeze(0)
    tau = 0.02
    vanilla_iters = _vanilla_convergence_iters(-cost, tau=tau, max_iters=120)
    momentum = dot.sinkhorn(cost, reg=tau, n_iters=120, method="momentum", omega=1.4)

    assert momentum.n_iters_used is not None
    assert momentum.n_iters_used < vanilla_iters


def test_anderson_matches_vanilla():
    import dot

    cost = torch.rand(1, 12, 12)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=120)
    anderson = dot.sinkhorn(cost, reg=0.1, n_iters=120, method="anderson", anderson_k=5)

    assert torch.allclose(vanilla.transport_plan, anderson.transport_plan, atol=1e-4)


def test_anderson_uses_fewer_iterations_than_vanilla():
    import dot

    n = 32
    grid = torch.linspace(-1.0, 1.0, n)
    cost = (grid[:, None] - grid[None, :]).abs().pow(2).unsqueeze(0)
    tau = 0.02
    vanilla_iters = _vanilla_convergence_iters(-cost, tau=tau, max_iters=120)
    anderson = dot.sinkhorn(cost, reg=tau, n_iters=120, method="anderson", anderson_k=5)

    assert anderson.n_iters_used is not None
    assert anderson.n_iters_used < vanilla_iters


def test_anderson_backward_matches_vanilla():
    import dot

    cost = torch.rand(1, 10, 10)

    cost_vanilla = cost.clone().requires_grad_(True)
    vanilla = dot.sinkhorn(cost_vanilla, reg=0.1, n_iters=120)
    vanilla.cost.sum().backward()

    cost_anderson = cost.clone().requires_grad_(True)
    anderson = dot.sinkhorn(cost_anderson, reg=0.1, n_iters=120, method="anderson", anderson_k=5)
    anderson.cost.sum().backward()

    assert torch.allclose(vanilla.transport_plan, anderson.transport_plan, atol=1e-4)
    assert torch.allclose(cost_vanilla.grad, cost_anderson.grad, atol=1e-4)


def test_anderson_falls_back_when_solve_fails():
    import dot

    cost = torch.rand(1, 10, 10)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=80)

    with mock.patch("torch.linalg.solve", side_effect=RuntimeError("singular")):
        anderson = dot.sinkhorn(cost, reg=0.1, n_iters=80, method="anderson", anderson_k=5)

    assert torch.allclose(vanilla.transport_plan, anderson.transport_plan, atol=1e-4)


def test_adam_matches_vanilla_on_well_conditioned_problem():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 12, 12)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=200)
    adam = dot.sinkhorn(cost, reg=0.1, n_iters=200, method="adam", lr=0.5)

    assert torch.allclose(vanilla.transport_plan, adam.transport_plan, atol=5e-4)


def test_adam_handles_nonuniform_costs():
    import dot

    n = 24
    grid = torch.linspace(-2.0, 2.0, n)
    cost = ((grid[:, None] - grid[None, :]).abs().pow(2) * torch.linspace(1.0, 3.0, n).unsqueeze(0)).unsqueeze(0)
    result = dot.sinkhorn(cost, reg=0.02, n_iters=40, method="adam", lr=0.5)

    assert torch.isfinite(result.transport_plan).all()
    assert result.n_iters_used is not None


def test_adam_rmsprop_regime_can_beat_vanilla():
    import dot

    n = 32
    grid = torch.linspace(-3.0, 3.0, n)
    cost = ((grid[:, None] - grid[None, :]).abs().pow(2) * torch.linspace(1.0, 5.0, n).unsqueeze(0)).unsqueeze(0)
    tau = 0.02
    ref = dot.sinkhorn(cost, reg=tau, n_iters=800)
    vanilla = dot.sinkhorn(cost, reg=tau, n_iters=20)
    rmsprop_like = dot.sinkhorn(
        cost,
        reg=tau,
        n_iters=20,
        method="adam",
        lr=0.25,
        beta1=0.0,
        beta2=0.999,
        bias_correction=False,
    )

    vanilla_err = (vanilla.transport_plan - ref.transport_plan).abs().mean()
    rmsprop_err = (rmsprop_like.transport_plan - ref.transport_plan).abs().mean()
    assert rmsprop_err < vanilla_err


def test_adam_bias_correction_can_hurt_deterministic_residuals():
    import dot

    n = 32
    grid = torch.linspace(-3.0, 3.0, n)
    cost = ((grid[:, None] - grid[None, :]).abs().pow(2) * torch.linspace(1.0, 5.0, n).unsqueeze(0)).unsqueeze(0)
    tau = 0.02
    ref = dot.sinkhorn(cost, reg=tau, n_iters=800)
    adam_default = dot.sinkhorn(cost, reg=tau, n_iters=20, method="adam", lr=0.5)
    rmsprop_like = dot.sinkhorn(
        cost,
        reg=tau,
        n_iters=20,
        method="adam",
        lr=0.25,
        beta1=0.0,
        beta2=0.999,
        bias_correction=False,
    )

    default_err = (adam_default.transport_plan - ref.transport_plan).abs().mean()
    rmsprop_err = (rmsprop_like.transport_plan - ref.transport_plan).abs().mean()
    assert rmsprop_err < default_err


def test_adam_backward_works():
    import dot

    cost = torch.rand(1, 10, 10, requires_grad=True)
    result = dot.sinkhorn(cost, reg=0.1, n_iters=80, method="adam", lr=0.5)
    result.cost.sum().backward()

    assert cost.grad is not None


def test_scheduled_vanilla_matches_unscheduled_when_reg_start_equals_target():
    import dot

    cost = torch.rand(1, 12, 12)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=200)
    scheduled = dot.sinkhorn(cost, reg=0.1, n_iters=200, schedule="cosine", reg_start=0.1)

    assert torch.allclose(vanilla.transport_plan, scheduled.transport_plan, atol=1e-4)


def test_scheduled_vanilla_handles_small_epsilon_better_than_vanilla():
    import dot

    n = 24
    torch.manual_seed(4)
    base = torch.rand(n, n)
    cost = ((base + base.t()) / 2).unsqueeze(0)
    tau = 0.001
    ref = dot.sinkhorn(cost, reg=tau, n_iters=600)
    vanilla = dot.sinkhorn(cost, reg=tau, n_iters=10)
    scheduled = dot.sinkhorn(cost, reg=tau, n_iters=10, schedule="linear", reg_start=0.1)

    vanilla_err = (vanilla.transport_plan - ref.transport_plan).abs().mean()
    scheduled_err = (scheduled.transport_plan - ref.transport_plan).abs().mean()
    assert scheduled_err < vanilla_err


def test_scheduled_vanilla_backward_works():
    import dot

    cost = torch.rand(1, 10, 10, requires_grad=True)
    result = dot.sinkhorn(cost, reg=0.01, n_iters=80, schedule="cosine", reg_start=0.1)
    result.cost.sum().backward()

    assert cost.grad is not None


def test_scheduled_anderson_matches_unscheduled_when_reg_start_equals_target():
    import dot

    cost = torch.rand(1, 12, 12)
    anderson = dot.sinkhorn(cost, reg=0.1, n_iters=120, method="anderson", anderson_k=5)
    scheduled = dot.sinkhorn(
        cost,
        reg=0.1,
        n_iters=120,
        method="anderson",
        anderson_k=5,
        schedule="cosine",
        reg_start=0.1,
    )

    assert torch.allclose(anderson.transport_plan, scheduled.transport_plan, atol=1e-4)


def test_scheduled_anderson_works():
    import dot

    cost = torch.rand(1, 12, 12, requires_grad=True)
    result = dot.sinkhorn(cost, reg=0.01, n_iters=80, method="anderson", schedule="cosine", reg_start=0.1)
    result.cost.sum().backward()

    assert cost.grad is not None


@pytest.mark.parametrize("n", [5, 10, 20])
def test_newton_matches_vanilla_on_small_problems(n: int):
    import dot

    torch.manual_seed(n)
    cost = torch.rand(1, n, n)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=200)
    newton = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="newton")

    assert torch.allclose(vanilla.transport_plan, newton.transport_plan, atol=1e-5)


def test_newton_converges_in_few_iterations():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 20, 20)
    result = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="newton")

    assert result.n_iters_used is not None
    assert result.n_iters_used <= 5


def test_newton_supports_explicit_marginals():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 12, 9)
    a = torch.rand(12)
    a = a / a.sum()
    b = torch.rand(9)
    b = b / b.sum()
    result = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="newton", a=a, b=b)

    assert torch.allclose(result.transport_plan.sum(dim=-1).squeeze(0), a, atol=1e-5)
    assert torch.allclose(result.transport_plan.sum(dim=-2).squeeze(0), b, atol=1e-5)


def test_newton_backward_works():
    import dot

    cost = torch.rand(1, 10, 10, requires_grad=True)
    result = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="newton")
    result.cost.sum().backward()

    assert cost.grad is not None


def test_auto_dispatch_selects_newton_for_small_problems():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 11, 11)
    auto = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="auto")
    newton = dot.sinkhorn(cost, reg=0.1, n_iters=5, method="newton")

    assert torch.allclose(auto.transport_plan, newton.transport_plan, atol=1e-6)


def test_muon_matches_vanilla():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 12, 12)
    vanilla = dot.sinkhorn(cost, reg=0.1, n_iters=120)
    muon = dot.sinkhorn(cost, reg=0.1, n_iters=120, method="muon", lr=0.05)

    assert torch.allclose(vanilla.transport_plan, muon.transport_plan, atol=1e-4)


def test_muon_handles_ill_conditioned_problems():
    import dot

    n = 24
    grid = torch.linspace(-1.0, 1.0, n)
    cost = ((grid[:, None] - grid[None, :]).abs().pow(2) + 1e-4 * torch.eye(n)).unsqueeze(0)
    ref = dot.sinkhorn(cost, reg=0.005, n_iters=400)
    muon = dot.sinkhorn(cost, reg=0.005, n_iters=40, method="muon", lr=0.05)

    assert torch.isfinite(muon.transport_plan).all()
    assert muon.n_iters_used is not None
    assert (muon.transport_plan - ref.transport_plan).abs().mean() < 5e-3


def test_muon_backward_works():
    import dot

    cost = torch.rand(1, 10, 10, requires_grad=True)
    result = dot.sinkhorn(cost, reg=0.1, n_iters=80, method="muon", lr=0.5)
    result.cost.sum().backward()

    assert cost.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda_large_rectangular_matches_reference():
    import dot

    torch.manual_seed(0)
    cost = torch.rand(1, 320, 640, device="cuda")

    result = dot.sinkhorn(cost, reg=0.1, n_iters=50)
    ref = _reference_from_scores(-cost, tau=0.1, n_iters=50)

    assert torch.allclose(result.transport_plan, ref, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sinkhorn_cuda_very_large_square_preserves_marginals():
    import dot

    torch.manual_seed(0)
    n = 1300
    cost = torch.rand(1, n, n, device="cuda")

    result = dot.sinkhorn(cost, reg=0.1, n_iters=50)
    uniform = _uniform(1, n, cost.device, cost.dtype)

    assert torch.isfinite(result.transport_plan).all()
    _assert_marginals(result.transport_plan, uniform, uniform, atol=2e-3)
