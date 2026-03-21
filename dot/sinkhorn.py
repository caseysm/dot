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
import math

import torch
import torch.nn as nn

from . import _ops


@dataclass
class SinkhornResult:
    """Result from Sinkhorn computation.

    Attributes:
        transport_plan: Transport plan P of shape (B, n, m)
        cost: Transport cost sum(P * C) of shape (B,) if cost_matrix provided
        converged: Whether the algorithm converged
    """
    transport_plan: torch.Tensor
    cost: Optional[torch.Tensor] = None
    converged: bool = True
    n_iters_used: Optional[int] = None


DEFAULT_SPECTRAL_PREFLIGHT_ALPHA = 1.0
SPECTRAL_PREFLIGHT_MIN_ITERS = 3
SPECTRAL_PREFLIGHT_MAX_ITERS = 1000
SCHEDULE_NONE = 0
SCHEDULE_LINEAR = 1
SCHEDULE_EXPONENTIAL = 2
SCHEDULE_COSINE = 3


class _SinkhornUnrolledFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with unrolled backward pass."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of Sinkhorn algorithm."""
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)
        P = _ops.sinkhorn(log_alpha, tau_value, n_iters, log_a, log_b)
        ctx.save_for_backward(
            log_alpha,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters = n_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        """Backward pass using unrolled differentiation."""
        log_alpha, tau_tensor, log_a, log_b = ctx.saved_tensors
        tau = ctx.tau
        n_iters = ctx.n_iters
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        # Call forward+backward together (re-computes forward with intermediates)
        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_unrolled(
            log_alpha, grad_P.contiguous(), tau, n_iters, log_a, log_b
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None


class _SinkhornImplicitFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with implicit backward pass."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of Sinkhorn algorithm."""
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)
        P = _ops.sinkhorn(log_alpha, tau_value, n_iters, log_a, log_b)
        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters = n_iters
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        """Backward pass using implicit differentiation."""
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        tau = ctx.tau
        n_iters = ctx.n_iters
        backward_iters = ctx.backward_iters
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha, grad_P.contiguous(), tau, n_iters, backward_iters, log_a, log_b
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None


class _SinkhornCudaDualImplicitFunction(torch.autograd.Function):
    """Autograd function for CUDA dual-state Sinkhorn variants."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        tol: float,
        method_code: int,
        omega: float,
        anderson_k: int,
        mixing_beta: float,
        lr: float,
        beta1: float,
        beta2: float,
        eps_adam: float,
        bias_correction: bool,
        reg_start: float,
        schedule_code: int,
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)
        result, n_iters_used, converged = _ops.sinkhorn_dual_forward(
            log_alpha,
            tau_value,
            n_iters,
            tol,
            log_a,
            log_b,
            False,
            method_code,
            omega,
            anderson_k,
            mixing_beta,
            lr,
            beta1,
            beta2,
            eps_adam,
            bias_correction,
            reg_start,
            schedule_code,
        )
        ctx.save_for_backward(
            log_alpha,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters = int(n_iters_used.item())
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        ctx.converged = bool(converged.item())
        return result

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class _SinkhornCudaDualUnrolledFunction(torch.autograd.Function):
    """Autograd function for CUDA dual-state Sinkhorn with unrolled backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        tol: float,
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)
        result, n_iters_used, converged = _ops.sinkhorn_dual_forward(
            log_alpha,
            tau_value,
            n_iters,
            tol,
            log_a,
            log_b,
            False,
            0,
            1.5,
            5,
            1.0,
            1.0,
            0.9,
            0.999,
            1e-8,
            True,
            -1.0,
            SCHEDULE_NONE,
        )
        ctx.save_for_backward(
            log_alpha,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters = int(n_iters_used.item())
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        ctx.converged = bool(converged.item())
        return result

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_unrolled(
            log_alpha, grad_P.contiguous(), ctx.tau, ctx.n_iters, log_a, log_b
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None


class _SinkhornMomentumImplicitFunction(torch.autograd.Function):
    """Autograd function for momentum Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        omega: float,
        reg_start: Optional[float],
        schedule: Optional[str],
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_momentum_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            omega,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        ctx.n_iters_used = n_iters_used
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None, None


class _SinkhornAndersonImplicitFunction(torch.autograd.Function):
    """Autograd function for Anderson-accelerated Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        anderson_k: int,
        mixing_beta: float,
        reg_start: Optional[float],
        schedule: Optional[str],
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_anderson_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            anderson_k,
            mixing_beta,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters_used = n_iters_used
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None, None, None


class _SinkhornAdamImplicitFunction(torch.autograd.Function):
    """Autograd function for Adam-style Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        eps_adam: float,
        bias_correction: bool,
        reg_start: Optional[float],
        schedule: Optional[str],
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_adam_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            lr,
            beta1,
            beta2,
            eps_adam,
            bias_correction,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters_used = n_iters_used
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None, None, None, None, None, None


class _SinkhornScheduledImplicitFunction(torch.autograd.Function):
    """Autograd function for scheduled-epsilon Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        reg_start: float,
        schedule: str,
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_scheduled_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters_used = n_iters_used
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None


class _SinkhornMuonImplicitFunction(torch.autograd.Function):
    """Autograd function for Muon-style Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        lr: float,
        reg_start: Optional[float],
        schedule: Optional[str],
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_muon_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            lr,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters_used = n_iters_used
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None, None


class _SinkhornNewtonImplicitFunction(torch.autograd.Function):
    """Autograd function for Newton Sinkhorn with implicit backward."""

    @staticmethod
    def forward(
        ctx,
        log_alpha: torch.Tensor,
        tau: float | torch.Tensor,
        n_iters: int,
        backward_iters: int,
        log_a: Optional[torch.Tensor],
        log_b: Optional[torch.Tensor],
        reg_start: Optional[float],
        schedule: Optional[str],
    ) -> torch.Tensor:
        tau_is_tensor = torch.is_tensor(tau)
        tau_tensor = tau if tau_is_tensor else torch.tensor([], device=log_alpha.device, dtype=log_alpha.dtype)
        if tau_is_tensor and tau_tensor.numel() != 1:
            raise ValueError("reg/tau tensor must be scalar")
        tau_value = float(tau_tensor.detach().item()) if tau_is_tensor else float(tau)

        P, n_iters_used = _run_newton_forward(
            log_alpha,
            tau_value,
            n_iters,
            log_a,
            log_b,
            reg_start,
            schedule,
        )

        ctx.save_for_backward(
            log_alpha,
            P,
            tau_tensor if tau_is_tensor else torch.Tensor(),
            log_a if log_a is not None else torch.Tensor(),
            log_b if log_b is not None else torch.Tensor(),
        )
        ctx.tau = tau_value
        ctx.n_iters_used = n_iters_used
        ctx.backward_iters = backward_iters
        ctx.tau_is_tensor = tau_is_tensor
        ctx.has_log_a = log_a is not None
        ctx.has_log_b = log_b is not None
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        log_alpha, P, tau_tensor, log_a, log_b = ctx.saved_tensors
        log_a = log_a if ctx.has_log_a else None
        log_b = log_b if ctx.has_log_b else None

        _, grad_log_alpha, grad_tau = _ops.sinkhorn_with_grads_implicit(
            log_alpha,
            grad_P.contiguous(),
            ctx.tau,
            ctx.n_iters_used,
            ctx.backward_iters,
            log_a,
            log_b,
        )

        tau_grad = None
        if ctx.tau_is_tensor:
            tau_grad = grad_tau.sum().to(device=tau_tensor.device, dtype=tau_tensor.dtype)

        return grad_log_alpha, tau_grad, None, None, None, None, None, None


def _default_log_marginals(
    log_alpha: torch.Tensor,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n, m = log_alpha.shape
    if log_a is None:
        log_a = torch.full((batch, n), -math.log(float(n)), device=log_alpha.device, dtype=log_alpha.dtype)
    if log_b is None:
        log_b = torch.full((batch, m), -math.log(float(m)), device=log_alpha.device, dtype=log_alpha.dtype)
    return log_a, log_b


def _dual_fixed_point_map(
    state: torch.Tensor,
    kernel: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
) -> torch.Tensor:
    n = log_a.size(-1)
    u = state[:, :n]
    v = state[:, n:]
    u_next = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
    v_next = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
    return torch.cat([u_next, v_next], dim=-1)


def _state_to_plan(kernel: torch.Tensor, state: torch.Tensor, n: int) -> torch.Tensor:
    u = state[:, :n]
    v = state[:, n:]
    return (kernel + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()


def _estimate_overrelaxation_omega(
    log_alpha: torch.Tensor,
    tau: float,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    pilot_iters: int = 10,
) -> float:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    kernel = log_alpha / tau
    batch, n, m = log_alpha.shape
    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)
    prev_delta = None
    ratio = 0.0
    for _ in range(max(pilot_iters, 2)):
        u_next = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
        v_next = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
        delta = max((u_next - u).abs().max().item(), (v_next - v).abs().max().item())
        if prev_delta is not None and prev_delta > 0.0:
            ratio = min(max(delta / prev_delta, 0.0), 0.999)
        prev_delta = max(delta, 1e-12)
        u, v = u_next, v_next

    sigma_sq = ratio * ratio
    omega = 2.0 / (1.0 + math.sqrt(max(1.0 - sigma_sq, 1e-6)))
    return float(min(max(omega, 1.0), 1.95))


def _compute_scheduled_reg(
    reg_start: float,
    reg_target: float,
    frac: float,
    schedule: str,
) -> float:
    if schedule == "linear":
        return reg_start * (1.0 - frac) + reg_target * frac
    if schedule == "cosine":
        return reg_target + 0.5 * (reg_start - reg_target) * (1.0 + math.cos(math.pi * frac))
    if schedule == "exponential":
        return reg_target * (reg_start / reg_target) ** (1.0 - frac)
    raise ValueError("schedule must be 'linear', 'exponential', or 'cosine'")


def _scheduled_tau(
    step: int,
    total_steps: int,
    reg_target: float,
    reg_start: Optional[float],
    schedule: Optional[str],
) -> float:
    if schedule is None:
        return reg_target
    if total_steps <= 1:
        return reg_target
    start = reg_start if reg_start is not None else 10.0 * reg_target
    frac = step / float(total_steps - 1)
    return _compute_scheduled_reg(start, reg_target, frac, schedule)


def _method_code(method: str) -> int:
    if method == "vanilla":
        return 0
    if method == "momentum":
        return 1
    if method == "anderson":
        return 2
    if method == "adam":
        return 3
    raise ValueError(f"Unsupported CUDA dual method: {method}")


def _schedule_code(schedule: Optional[str]) -> int:
    if schedule is None:
        return SCHEDULE_NONE
    if schedule == "linear":
        return SCHEDULE_LINEAR
    if schedule == "exponential":
        return SCHEDULE_EXPONENTIAL
    if schedule == "cosine":
        return SCHEDULE_COSINE
    raise ValueError(f"Unsupported schedule: {schedule}")


def spectral_preflight(
    log_alpha: torch.Tensor,
    tau: float | torch.Tensor,
    tol: float = 1e-6,
    n_power: int = 5,
    alpha: Optional[float] = None,
) -> int | torch.Tensor:
    """Predict conservative Sinkhorn iteration counts from a spectral estimate."""
    if tol <= 0:
        raise ValueError("spectral_preflight requires tol > 0")
    tau_value = float(tau.detach().item()) if torch.is_tensor(tau) else float(tau)
    if tau_value <= 0:
        raise ValueError("tau/reg must be positive")
    alpha_value = DEFAULT_SPECTRAL_PREFLIGHT_ALPHA if alpha is None else float(alpha)

    unbatched = log_alpha.dim() == 2
    if unbatched:
        log_alpha = log_alpha.unsqueeze(0)

    if log_alpha.is_cuda:
        tau_est = _ops.sinkhorn_spectral_preflight(log_alpha, tau_value, n_power)
    else:
        log_kernel = log_alpha / tau_value
        log_P = log_kernel - torch.logsumexp(log_kernel, dim=-1, keepdim=True)
        P = log_P.exp()
        positions = torch.arange(1, log_alpha.size(-1) + 1, device=log_alpha.device, dtype=log_alpha.dtype)
        batch_offsets = torch.arange(1, log_alpha.size(0) + 1, device=log_alpha.device, dtype=log_alpha.dtype).unsqueeze(1)
        v = torch.sin(positions.unsqueeze(0) * 12.9898 + batch_offsets * 78.233)
        v = v - v.mean(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for _ in range(n_power):
            u = torch.matmul(P, v.unsqueeze(-1)).squeeze(-1)
            u = u - u.mean(dim=-1, keepdim=True)
            v = torch.matmul(P.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
            v = v - v.mean(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = torch.matmul(P, v.unsqueeze(-1)).squeeze(-1)
        u = u - u.mean(dim=-1, keepdim=True)
        tau_est = u.norm(dim=-1)

    tau_hat = tau_est.clamp(min=1e-6, max=0.999999)
    pred = torch.ceil(alpha_value * math.log(1.0 / tol) / (-torch.log(tau_hat))).to(torch.int64)
    pred = pred.clamp(min=SPECTRAL_PREFLIGHT_MIN_ITERS, max=SPECTRAL_PREFLIGHT_MAX_ITERS)
    if unbatched:
        return int(pred.item())
    return pred


def _run_momentum_forward(
    log_alpha: torch.Tensor,
    tau: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    omega: float,
    reg_start: Optional[float] = None,
    schedule: Optional[str] = None,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    batch, n, m = log_alpha.shape
    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau, reg_start, schedule)
        kernel = log_alpha / tau_current
        u_cd = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
        u_next = (1.0 - omega) * u + omega * u_cd

        v_cd = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
        v_next = (1.0 - omega) * v + omega * v_cd

        delta = max((u_next - u).abs().max().item(), (v_next - v).abs().max().item())
        u, v = u_next, v_next
        if delta < tol:
            break

    kernel_target = log_alpha / tau
    return (kernel_target + u.unsqueeze(-1) + v.unsqueeze(-2)).exp(), iter_idx + 1


def _run_anderson_forward(
    log_alpha: torch.Tensor,
    tau: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    anderson_k: int,
    mixing_beta: float,
    reg_start: Optional[float] = None,
    schedule: Optional[str] = None,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    batch, n, m = log_alpha.shape
    dim = n + m
    state = torch.zeros((batch, dim), device=log_alpha.device, dtype=log_alpha.dtype)
    history_x: list[torch.Tensor] = []
    history_f: list[torch.Tensor] = []

    eye = torch.eye(anderson_k, device=log_alpha.device, dtype=log_alpha.dtype)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau, reg_start, schedule)
        kernel = log_alpha / tau_current
        x_new = _dual_fixed_point_map(state, kernel, log_a, log_b)
        residual = x_new - state
        res_norm = residual.norm(dim=-1).max().item()

        history_x.append(state.detach())
        history_f.append(residual.detach())
        if len(history_x) > anderson_k:
            history_x.pop(0)
            history_f.pop(0)

        if len(history_x) >= 2:
            h = len(history_x)
            X = torch.stack(history_x, dim=-1)
            F = torch.stack(history_f, dim=-1)
            gram = torch.matmul(F.transpose(-2, -1), F) + 1e-8 * eye[:h, :h].unsqueeze(0)
            ones = torch.ones((batch, h, 1), device=log_alpha.device, dtype=log_alpha.dtype)
            try:
                weights = torch.linalg.solve(gram, ones)
                alpha = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                candidates = X + mixing_beta * F
                state = (candidates * alpha.squeeze(-1).unsqueeze(1)).sum(dim=-1)
            except RuntimeError:
                state = x_new
        else:
            state = x_new

        if res_norm < tol:
            break
    kernel_target = log_alpha / tau
    return _state_to_plan(kernel_target, state, n), iter_idx + 1


def _run_adam_forward(
    log_alpha: torch.Tensor,
    tau: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps_adam: float,
    bias_correction: bool,
    reg_start: Optional[float] = None,
    schedule: Optional[str] = None,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    batch, n, m = log_alpha.shape

    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)
    m_u = torch.zeros_like(u)
    v_u = torch.zeros_like(u)
    m_v = torch.zeros_like(v)
    v_v = torch.zeros_like(v)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau, reg_start, schedule)
        kernel = log_alpha / tau_current
        u_target = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
        g_u = u_target - u
        m_u = beta1 * m_u + (1.0 - beta1) * g_u
        v_u = beta2 * v_u + (1.0 - beta2) * (g_u * g_u)
        if bias_correction:
            m_u_hat = m_u / (1.0 - beta1 ** (iter_idx + 1))
            v_u_hat = v_u / (1.0 - beta2 ** (iter_idx + 1))
        else:
            m_u_hat = m_u
            v_u_hat = v_u
        u_next = u + lr * m_u_hat / (torch.sqrt(v_u_hat) + eps_adam)

        v_target = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
        g_v = v_target - v
        m_v = beta1 * m_v + (1.0 - beta1) * g_v
        v_v = beta2 * v_v + (1.0 - beta2) * (g_v * g_v)
        if bias_correction:
            m_v_hat = m_v / (1.0 - beta1 ** (iter_idx + 1))
            v_v_hat = v_v / (1.0 - beta2 ** (iter_idx + 1))
        else:
            m_v_hat = m_v
            v_v_hat = v_v
        v_next = v + lr * m_v_hat / (torch.sqrt(v_v_hat) + eps_adam)

        delta = max((u_next - u).abs().max().item(), (v_next - v).abs().max().item())
        u, v = u_next, v_next
        if delta < tol:
            break

    kernel_target = log_alpha / tau
    return (kernel_target + u.unsqueeze(-1) + v.unsqueeze(-2)).exp(), iter_idx + 1


def _run_scheduled_forward(
    log_alpha: torch.Tensor,
    tau_target: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    reg_start: float,
    schedule: str,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    batch, n, m = log_alpha.shape
    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau_target, reg_start, schedule)
        kernel = log_alpha / tau_current
        u_next = log_a - torch.logsumexp(kernel + v.unsqueeze(-2), dim=-1)
        v_next = log_b - torch.logsumexp(kernel + u_next.unsqueeze(-1), dim=-2)
        delta = max((u_next - u).abs().max().item(), (v_next - v).abs().max().item())
        u, v = u_next, v_next
        if iter_idx >= max(n_iters // 2, 1) and delta < tol:
            break

    kernel_target = log_alpha / tau_target
    return (kernel_target + u.unsqueeze(-1) + v.unsqueeze(-2)).exp(), iter_idx + 1


def _project_birkhoff_tangent(gradient: torch.Tensor) -> torch.Tensor:
    row_mean = gradient.mean(dim=-1, keepdim=True)
    col_mean = gradient.mean(dim=-2, keepdim=True)
    grand_mean = gradient.mean(dim=(-2, -1), keepdim=True)
    return gradient - row_mean - col_mean + grand_mean


def _cleanup_transport_plan(
    plan: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
) -> torch.Tensor:
    log_plan = plan.clamp_min(1e-30).log()
    log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True) + log_a.unsqueeze(-1)
    log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True) + log_b.unsqueeze(-2)
    return log_plan.exp()


def _iterative_cleanup_transport_plan(
    plan: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    n_steps: int = 5,
) -> torch.Tensor:
    log_plan = plan.clamp_min(1e-30).log()
    for _ in range(n_steps):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True) + log_a.unsqueeze(-1)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True) + log_b.unsqueeze(-2)
    return log_plan.exp()


def _run_muon_forward(
    log_alpha: torch.Tensor,
    tau: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    lr: float,
    reg_start: Optional[float] = None,
    schedule: Optional[str] = None,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    warm_start_iters = 5
    target_iters = max(n_iters, 200)
    current_tau = _scheduled_tau(0, n_iters, tau, reg_start, schedule)
    current_plan = _ops.sinkhorn(log_alpha, current_tau, warm_start_iters, log_a, log_b)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau, reg_start, schedule)
        target_plan = _ops.sinkhorn(log_alpha, tau_current, target_iters, log_a, log_b)
        target_log = target_plan.clamp_min(1e-30).log()
        current_log = current_plan.clamp_min(1e-30).log()
        gradient = current_plan * (target_log - current_log)
        tangent = _project_birkhoff_tangent(gradient)
        step = (lr * tangent / current_plan.clamp_min(1e-30)).clamp(min=-20.0, max=20.0)
        candidate_plan = current_plan * torch.exp(step)
        next_plan = _cleanup_transport_plan(candidate_plan, log_a, log_b)
        delta = (next_plan - current_plan).abs().max().item()
        current_plan = next_plan
        if delta < tol:
            break

    if schedule is not None:
        current_plan = _cleanup_transport_plan(current_plan, log_a, log_b)
    return current_plan, warm_start_iters + iter_idx + 1


def _run_newton_forward(
    log_alpha: torch.Tensor,
    tau: float,
    n_iters: int,
    log_a: Optional[torch.Tensor],
    log_b: Optional[torch.Tensor],
    reg_start: Optional[float] = None,
    schedule: Optional[str] = None,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    log_a, log_b = _default_log_marginals(log_alpha, log_a, log_b)
    batch, n, m = log_alpha.shape
    if n > 100 or m > 100:
        raise ValueError("method='newton' only supports n, m <= 100")

    a = log_a.exp()
    b = log_b.exp()
    u = torch.zeros((batch, n), device=log_alpha.device, dtype=log_alpha.dtype)
    v = torch.zeros((batch, m), device=log_alpha.device, dtype=log_alpha.dtype)

    for iter_idx in range(n_iters):
        tau_current = _scheduled_tau(iter_idx, n_iters, tau, reg_start, schedule)
        kernel = log_alpha / tau_current
        P = (kernel + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()
        row_sum = P.sum(dim=-1)
        col_sum = P.sum(dim=-2)
        grad_r = row_sum - a
        grad_c = col_sum - b
        grad = torch.cat([grad_r, grad_c], dim=-1)
        grad_norm = grad.abs().amax(dim=-1)
        if grad_norm.max().item() < tol:
            break

        next_u = u.clone()
        next_v = v.clone()
        for batch_idx in range(batch):
            grad_norm_b = grad_norm[batch_idx].item()
            H = torch.zeros((n + m, n + m), device=log_alpha.device, dtype=log_alpha.dtype)
            H[:n, :n] = torch.diag(row_sum[batch_idx])
            H[n:, n:] = torch.diag(col_sum[batch_idx])
            H[:n, n:] = P[batch_idx]
            H[n:, :n] = P[batch_idx].transpose(0, 1)
            rhs = -grad[batch_idx]

            H[0, :] = 0
            H[:, 0] = 0
            H[0, 0] = 1
            rhs[0] = 0

            try:
                delta = torch.linalg.solve(H, rhs)
                accepted = False
                for shrink in range(6):
                    step_scale = 0.5 ** shrink
                    cand_u = u[batch_idx] + step_scale * delta[:n]
                    cand_v = v[batch_idx] + step_scale * delta[n:]
                    cand_log_plan = kernel[batch_idx] + cand_u.unsqueeze(-1) + cand_v.unsqueeze(-2)
                    if not torch.isfinite(cand_log_plan).all() or cand_log_plan.max().item() > 80.0:
                        continue
                    cand_plan = cand_log_plan.exp()
                    cand_grad_r = cand_plan.sum(dim=-1) - a[batch_idx]
                    cand_grad_c = cand_plan.sum(dim=-2) - b[batch_idx]
                    cand_grad_norm = max(cand_grad_r.abs().max().item(), cand_grad_c.abs().max().item())
                    if cand_grad_norm < grad_norm_b:
                        refined_state = _dual_fixed_point_map(
                            torch.cat([cand_u.unsqueeze(0), cand_v.unsqueeze(0)], dim=-1),
                            kernel[batch_idx:batch_idx + 1],
                            log_a[batch_idx:batch_idx + 1],
                            log_b[batch_idx:batch_idx + 1],
                        )
                        next_u[batch_idx] = refined_state[0, :n]
                        next_v[batch_idx] = refined_state[0, n:]
                        accepted = True
                        break
                if not accepted:
                    state = torch.cat([u[batch_idx:batch_idx + 1], v[batch_idx:batch_idx + 1]], dim=-1)
                    next_state = _dual_fixed_point_map(
                        state,
                        kernel[batch_idx:batch_idx + 1],
                        log_a[batch_idx:batch_idx + 1],
                        log_b[batch_idx:batch_idx + 1],
                    )
                    next_u[batch_idx] = next_state[0, :n]
                    next_v[batch_idx] = next_state[0, n:]
            except RuntimeError:
                state = torch.cat([u[batch_idx:batch_idx + 1], v[batch_idx:batch_idx + 1]], dim=-1)
                next_state = _dual_fixed_point_map(
                    state,
                    kernel[batch_idx:batch_idx + 1],
                    log_a[batch_idx:batch_idx + 1],
                    log_b[batch_idx:batch_idx + 1],
                )
                next_u[batch_idx] = next_state[0, :n]
                next_v[batch_idx] = next_state[0, n:]

        u = next_u
        v = next_v

    kernel_target = log_alpha / tau
    plan = (kernel_target + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()
    return _iterative_cleanup_transport_plan(plan, log_a, log_b, n_steps=10), iter_idx + 1


def _prepare_marginal(
    marginal: Optional[torch.Tensor],
    batch: int,
    size: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if marginal is None:
        return None

    if marginal.dim() == 1:
        marginal = marginal.unsqueeze(0).expand(batch, -1)
    if marginal.shape != (batch, size):
        raise ValueError(f"{name} must have shape ({batch}, {size}) or ({size},)")
    if torch.any(marginal <= 0):
        raise ValueError(f"{name} must be strictly positive")

    marginal = marginal.to(device=device, dtype=dtype)
    sums = marginal.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must sum to 1 along the last dimension")
    return marginal


def sinkhorn(
    cost_matrix: torch.Tensor,
    reg: float | torch.Tensor = 1.0,
    n_iters: int | Literal['auto'] = 20,
    backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
    backward_iters: Optional[int] = None,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    method: Literal['vanilla', 'momentum', 'anderson', 'adam', 'newton', 'muon', 'auto'] = 'vanilla',
    omega: Optional[float] = None,
    anderson_k: int = 5,
    mixing_beta: float = 1.0,
    lr: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    bias_correction: bool = True,
    reg_start: Optional[float] = None,
    schedule: Optional[Literal['linear', 'exponential', 'cosine']] = None,
    tol: float = 1e-6,
    dual: bool = True,
) -> SinkhornResult:
    """Compute entropy-regularized optimal transport using Sinkhorn algorithm.

    Produces a transport plan from a cost matrix. Square inputs recover the
    doubly-stochastic soft-permutation behavior.

    Args:
        cost_matrix: Cost matrix of shape (B, n, m) or (n, m)
        reg: Regularization strength / temperature (default: 1.0)
        n_iters: Number of Sinkhorn iterations or 'auto' for spectral preflight
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)
        a: Optional source marginal of shape (B, n) or (n,)
        b: Optional target marginal of shape (B, m) or (m,)
        method: 'vanilla', 'momentum', 'anderson', 'adam', 'newton', 'muon', or 'auto'
        omega: Optional momentum factor for method='momentum'
        anderson_k: History size for method='anderson'
        mixing_beta: Damping factor for method='anderson'
        lr: Step size for method='adam' or method='muon'
        beta1: Adam first-moment decay for method='adam'
        beta2: Adam second-moment decay for method='adam'
        eps_adam: Adam epsilon for method='adam'
        bias_correction: Whether to apply Adam bias correction for method='adam'.
            On hard deterministic Sinkhorn problems, a second-moment-only
            regime (`beta1=0`, `bias_correction=False`) can outperform the
            default bias-corrected variant.
        reg_start: Starting regularization for schedule (default: 10 * reg)
        schedule: Optional regularization schedule applied to the chosen method
        tol: Convergence tolerance. `tol=0` forces all iterations.
        dual: Whether to prefer the dual-state CUDA path when available.

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
    a = _prepare_marginal(a, cost_matrix.size(0), cost_matrix.size(-2), "a", cost_matrix.device, cost_matrix.dtype)
    b = _prepare_marginal(b, cost_matrix.size(0), cost_matrix.size(-1), "b", cost_matrix.device, cost_matrix.dtype)
    log_a = torch.log(a) if a is not None else None
    log_b = torch.log(b) if b is not None else None

    tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
    if isinstance(n_iters, str):
        if n_iters != "auto":
            raise ValueError("n_iters must be an int or 'auto'")
        predicted_iters = spectral_preflight(log_alpha.detach(), tau_value, tol=tol)
        n_iters = int(predicted_iters.max().item()) if torch.is_tensor(predicted_iters) else int(predicted_iters)

    if backward_iters is None:
        backward_iters = n_iters

    if method == 'auto':
        if cost_matrix.size(-2) <= 100 and cost_matrix.size(-1) <= 100:
            method = 'newton'
        else:
            method = 'vanilla'

    effective_method = method
    effective_schedule = schedule
    effective_reg_start = reg_start
    if dual and cost_matrix.is_cuda and backward_mode == 'implicit':
        if effective_method == 'momentum' and omega is None:
            effective_method = 'vanilla'
        elif effective_method == 'anderson' and effective_schedule is None:
            if tau_value >= 0.05:
                effective_method = 'vanilla'
            elif tau_value <= 0.01:
                effective_schedule = 'exponential'
                if effective_reg_start is None:
                    effective_reg_start = 10.0 * tau_value
        elif effective_method == 'adam' and effective_schedule is None:
            is_default_adam = (
                lr == 1.0
                and beta1 == 0.9
                and beta2 == 0.999
                and eps_adam == 1e-8
                and bias_correction
            )
            if tau_value >= 0.05 or is_default_adam:
                effective_method = 'vanilla'

    n_iters_used: Optional[int] = None
    use_cuda_dual = (
        dual
        and cost_matrix.is_cuda
        and (
            (backward_mode == 'implicit' and effective_method in {'vanilla', 'momentum', 'anderson', 'adam'})
            or (backward_mode == 'unrolled' and effective_method == 'vanilla' and effective_schedule is None)
        )
    )
    needs_grad = (
        cost_matrix.requires_grad
        or (torch.is_tensor(reg) and reg.requires_grad)
        or (a is not None and a.requires_grad)
        or (b is not None and b.requires_grad)
    )

    if use_cuda_dual:
        if backward_mode == 'unrolled':
            if needs_grad:
                transport_plan = _SinkhornCudaDualUnrolledFunction.apply(
                    log_alpha,
                    reg,
                    n_iters,
                    log_a,
                    log_b,
                    tol,
                )
                with torch.no_grad():
                    _, used_tensor, converged_tensor = _ops.sinkhorn_dual_forward(
                        log_alpha.detach(),
                        tau_value,
                        n_iters,
                        tol,
                        log_a.detach() if log_a is not None else None,
                        log_b.detach() if log_b is not None else None,
                        False,
                        0,
                        1.5,
                        5,
                        1.0,
                        1.0,
                        0.9,
                        0.999,
                        1e-8,
                        True,
                        -1.0,
                        SCHEDULE_NONE,
                    )
            else:
                transport_plan, used_tensor, converged_tensor = _ops.sinkhorn_dual_forward(
                    log_alpha,
                    tau_value,
                    n_iters,
                    tol,
                    log_a,
                    log_b,
                    False,
                    0,
                    1.5,
                    5,
                    1.0,
                    1.0,
                    0.9,
                    0.999,
                    1e-8,
                    True,
                    -1.0,
                    SCHEDULE_NONE,
                )
        else:
            method_code = _method_code(effective_method)
            omega_value = float(omega) if omega is not None else 1.4
            reg_start_value = -1.0 if effective_reg_start is None else float(effective_reg_start)
            schedule_code = _schedule_code(effective_schedule)

            if needs_grad:
                transport_plan = _SinkhornCudaDualImplicitFunction.apply(
                    log_alpha,
                    reg,
                    n_iters,
                    backward_iters,
                    log_a,
                    log_b,
                    tol,
                    method_code,
                    omega_value,
                    anderson_k,
                    mixing_beta,
                    lr,
                    beta1,
                    beta2,
                    eps_adam,
                    bias_correction,
                    reg_start_value,
                    schedule_code,
                )
                with torch.no_grad():
                    _, used_tensor, converged_tensor = _ops.sinkhorn_dual_forward(
                        log_alpha.detach(),
                        tau_value,
                        n_iters,
                        tol,
                        log_a.detach() if log_a is not None else None,
                        log_b.detach() if log_b is not None else None,
                        False,
                        method_code,
                        omega_value,
                        anderson_k,
                        mixing_beta,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        bias_correction,
                        reg_start_value,
                        schedule_code,
                    )
            else:
                transport_plan, used_tensor, converged_tensor = _ops.sinkhorn_dual_forward(
                    log_alpha,
                    tau_value,
                    n_iters,
                    tol,
                    log_a,
                    log_b,
                    False,
                    method_code,
                    omega_value,
                    anderson_k,
                    mixing_beta,
                    lr,
                    beta1,
                    beta2,
                    eps_adam,
                    bias_correction,
                    reg_start_value,
                    schedule_code,
                )

        n_iters_used = int(used_tensor.item())
        converged = bool(converged_tensor.item()) if tol > 0 else True
    elif method == 'momentum':
        if backward_mode != 'implicit':
            raise ValueError("method='momentum' only supports implicit backward")
        omega_value = float(omega) if omega is not None else _estimate_overrelaxation_omega(
            log_alpha.detach(),
            tau_value,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
        )
        transport_plan = _SinkhornMomentumImplicitFunction.apply(
            log_alpha,
            reg,
            n_iters,
            backward_iters,
            log_a,
            log_b,
            omega_value,
            reg_start,
            schedule,
        )
        _, n_iters_used = _run_momentum_forward(
            log_alpha.detach(),
            tau_value,
            n_iters,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
            omega_value,
            reg_start,
            schedule,
        )
    elif method == 'anderson':
        if backward_mode != 'implicit':
            raise ValueError("method='anderson' only supports implicit backward")
        tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
        transport_plan = _SinkhornAndersonImplicitFunction.apply(
            log_alpha,
            reg,
            n_iters,
            backward_iters,
            log_a,
            log_b,
            anderson_k,
            mixing_beta,
            reg_start,
            schedule,
        )
        _, n_iters_used = _run_anderson_forward(
            log_alpha.detach(),
            tau_value,
            n_iters,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
            anderson_k,
            mixing_beta,
            reg_start,
            schedule,
        )
    elif method == 'adam':
        if backward_mode != 'implicit':
            raise ValueError("method='adam' only supports implicit backward")
        tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
        transport_plan = _SinkhornAdamImplicitFunction.apply(
            log_alpha,
            reg,
            n_iters,
            backward_iters,
            log_a,
            log_b,
            lr,
            beta1,
            beta2,
            eps_adam,
            bias_correction,
            reg_start,
            schedule,
        )
        _, n_iters_used = _run_adam_forward(
            log_alpha.detach(),
            tau_value,
            n_iters,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
            lr,
            beta1,
            beta2,
            eps_adam,
            bias_correction,
            reg_start,
            schedule,
        )
    elif method == 'muon':
        if backward_mode != 'implicit':
            raise ValueError("method='muon' only supports implicit backward")
        tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
        transport_plan = _SinkhornMuonImplicitFunction.apply(
            log_alpha,
            reg,
            n_iters,
            backward_iters,
            log_a,
            log_b,
            lr,
            reg_start,
            schedule,
        )
        _, n_iters_used = _run_muon_forward(
            log_alpha.detach(),
            tau_value,
            n_iters,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
            lr,
            reg_start,
            schedule,
        )
    elif method == 'newton':
        if backward_mode != 'implicit':
            raise ValueError("method='newton' only supports implicit backward")
        tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
        transport_plan = _SinkhornNewtonImplicitFunction.apply(
            log_alpha,
            reg,
            n_iters,
            backward_iters,
            log_a,
            log_b,
            reg_start,
            schedule,
        )
        _, n_iters_used = _run_newton_forward(
            log_alpha.detach(),
            tau_value,
            n_iters,
            log_a.detach() if log_a is not None else None,
            log_b.detach() if log_b is not None else None,
            reg_start,
            schedule,
        )
    else:
        if backward_mode == 'unrolled':
            if schedule is not None:
                raise ValueError("schedule is only supported with implicit backward")
            transport_plan = _SinkhornUnrolledFunction.apply(log_alpha, reg, n_iters, log_a, log_b)
        elif backward_mode == 'implicit':
            if schedule is not None:
                tau_value = float(reg.detach().item()) if torch.is_tensor(reg) else float(reg)
                reg_start_value = float(reg_start) if reg_start is not None else 10.0 * tau_value
                transport_plan = _SinkhornScheduledImplicitFunction.apply(
                    log_alpha, reg, n_iters, backward_iters, log_a, log_b, reg_start_value, schedule
                )
                _, n_iters_used = _run_scheduled_forward(
                    log_alpha.detach(),
                    tau_value,
                    n_iters,
                    log_a.detach() if log_a is not None else None,
                    log_b.detach() if log_b is not None else None,
                    reg_start_value,
                    schedule,
                )
            else:
                transport_plan = _SinkhornImplicitFunction.apply(log_alpha, reg, n_iters, backward_iters, log_a, log_b)
        else:
            raise ValueError("backward_mode must be 'unrolled' or 'implicit'")
    if method == 'vanilla':
        if n_iters_used is None:
            n_iters_used = n_iters
    if not use_cuda_dual:
        converged = True if tol == 0 else (n_iters_used is not None and n_iters_used < n_iters)

    # Compute cost
    cost = (transport_plan * cost_matrix).sum(dim=(-2, -1))

    if unbatched:
        transport_plan = transport_plan.squeeze(0)
        cost = cost.squeeze(0)

    return SinkhornResult(
        transport_plan=transport_plan,
        cost=cost,
        converged=converged,
        n_iters_used=n_iters_used,
    )


def sinkhorn_from_scores(
    log_alpha: torch.Tensor,
    tau: float | torch.Tensor = 1.0,
    n_iters: int | Literal['auto'] = 20,
    backward_mode: Literal['unrolled', 'implicit'] = 'implicit',
    backward_iters: Optional[int] = None,
    return_log: bool = False,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    dual: bool = True,
) -> torch.Tensor:
    """Compute normalized transport plan directly from log-space scores.

    This is the low-level interface that works directly with log_alpha scores.
    Use this when you have similarity scores or logits, not costs.

    Args:
        log_alpha: Log-space scores of shape (B, n, m) or (n, m)
        tau: Temperature parameter (default: 1.0)
        n_iters: Number of Sinkhorn iterations or 'auto' for spectral preflight
        backward_mode: 'unrolled' for exact gradients, 'implicit' for memory efficiency
        backward_iters: Iterations for implicit backward (default: same as n_iters)
        return_log: If True, return log(P) instead of P
        a: Optional source marginal of shape (B, n) or (n,)
        b: Optional target marginal of shape (B, m) or (m,)
        tol: Convergence tolerance. `tol=0` forces all iterations.
        dual: Whether to prefer the dual-state path when available.

    Returns:
        Normalized transport plan P or log(P) of same shape as input
    """
    # Handle unbatched input
    unbatched = log_alpha.dim() == 2
    if unbatched:
        log_alpha = log_alpha.unsqueeze(0)
    a = _prepare_marginal(a, log_alpha.size(0), log_alpha.size(-2), "a", log_alpha.device, log_alpha.dtype)
    b = _prepare_marginal(b, log_alpha.size(0), log_alpha.size(-1), "b", log_alpha.device, log_alpha.dtype)
    log_a = torch.log(a) if a is not None else None
    log_b = torch.log(b) if b is not None else None

    tau_value = float(tau.detach().item()) if torch.is_tensor(tau) else float(tau)
    if isinstance(n_iters, str):
        if n_iters != "auto":
            raise ValueError("n_iters must be an int or 'auto'")
        predicted_iters = spectral_preflight(log_alpha.detach(), tau_value, tol=tol)
        n_iters = int(predicted_iters.max().item()) if torch.is_tensor(predicted_iters) else int(predicted_iters)

    if backward_iters is None:
        backward_iters = n_iters

    use_cuda_dual = dual and log_alpha.is_cuda and backward_mode in {'implicit', 'unrolled'}
    needs_grad = (
        log_alpha.requires_grad
        or (torch.is_tensor(tau) and tau.requires_grad)
        or (a is not None and a.requires_grad)
        or (b is not None and b.requires_grad)
    )

    if use_cuda_dual:
        if return_log:
            result, _, _ = _ops.sinkhorn_dual_forward(
                log_alpha,
                tau_value,
                n_iters,
                tol,
                log_a,
                log_b,
                True,
                0,
                1.5,
                5,
                1.0,
                1.0,
                0.9,
                0.999,
                1e-8,
                True,
                -1.0,
                SCHEDULE_NONE,
            )
        elif backward_mode == 'unrolled' and needs_grad:
            result = _SinkhornCudaDualUnrolledFunction.apply(
                log_alpha,
                tau,
                n_iters,
                log_a,
                log_b,
                tol,
            )
        elif needs_grad:
            result = _SinkhornCudaDualImplicitFunction.apply(
                log_alpha,
                tau,
                n_iters,
                backward_iters,
                log_a,
                log_b,
                tol,
                0,
                1.5,
                5,
                1.0,
                1.0,
                0.9,
                0.999,
                1e-8,
                True,
                -1.0,
                SCHEDULE_NONE,
            )
        else:
            result, _, _ = _ops.sinkhorn_dual_forward(
                log_alpha,
                tau_value,
                n_iters,
                tol,
                log_a,
                log_b,
                False,
                0,
                1.5,
                5,
                1.0,
                1.0,
                0.9,
                0.999,
                1e-8,
                True,
                -1.0,
                SCHEDULE_NONE,
            )
    elif return_log:
        # No autograd for log version (use for inference only)
        result = _ops.sinkhorn_log(log_alpha, tau, n_iters, log_a, log_b)
    elif backward_mode == 'unrolled':
        result = _SinkhornUnrolledFunction.apply(log_alpha, tau, n_iters, log_a, log_b)
    else:
        result = _SinkhornImplicitFunction.apply(log_alpha, tau, n_iters, backward_iters, log_a, log_b)

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
            cost_matrix: Cost matrix of shape (B, n, m) or (n, m)

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
