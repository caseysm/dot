/**
 * @file registry.cpp
 * @brief Schema definitions for Sinkhorn optimal transport operators
 *
 * Defines the operator schemas for Sinkhorn algorithm.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * API:
 *   - sinkhorn(log_alpha, tau, n_iters) -> P
 *   - sinkhorn_log(log_alpha, tau, n_iters) -> log(P)
 *   - sinkhorn_dual_forward(log_alpha, tau, n_iters, tol, ...) -> [P_or_logP, n_iters_used, converged]
 *   - sinkhorn_spectral_preflight(log_alpha, tau, n_power) -> tau_estimates
 *   - sinkhorn_with_grads_unrolled(log_alpha, grad_P, tau, n_iters) -> [P, grad_log_alpha, grad_tau]
 *   - sinkhorn_with_grads_implicit(log_alpha, grad_P, tau, n_iters, backward_iters) -> [P, grad_log_alpha, grad_tau]
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(dot, m) {
    // =========================================================================
    // SINKHORN OPTIMAL TRANSPORT
    // =========================================================================
    //
    // Converts log-space scores to transport plan / soft permutation.
    // Input: log_alpha [B, n, m] - logits (use -D/sigma for distance matrix D)
    // Output: P [B, n, m] - normalized transport plan

    // Basic forward: compute doubly-stochastic matrix
    m.def("sinkhorn(Tensor log_alpha, float tau, int n_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor");

    // Log-space forward: return log(P) for numerical stability
    m.def("sinkhorn_log(Tensor log_alpha, float tau, int n_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor");

    // Dual-space forward with optional convergence checking and acceleration.
    // Returns [P_or_logP, n_iters_used, converged]
    m.def(
        "sinkhorn_dual_forward("
        "Tensor log_alpha, float tau, int n_iters, float tol=0.0, Tensor? log_a=None, Tensor? log_b=None, "
        "bool return_log=False, int method=0, float omega=1.5, int anderson_k=5, float mixing_beta=1.0, "
        "float lr=1.0, float beta1=0.9, float beta2=0.999, float eps_adam=1e-8, bool bias_correction=True, "
        "float reg_start=-1.0, int schedule=0"
        ") -> Tensor[]"
    );

    // Spectral preflight returns one contraction estimate per batch element.
    m.def("sinkhorn_spectral_preflight(Tensor log_alpha, float tau, int n_power=8) -> Tensor");

    // Forward + backward with unrolled differentiation through iterations
    // Returns [P, grad_log_alpha, grad_tau]
    m.def("sinkhorn_with_grads_unrolled(Tensor log_alpha, Tensor grad_P, float tau, int n_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor[]");

    // Forward + backward with implicit differentiation (memory efficient)
    // Returns [P, grad_log_alpha, grad_tau]
    m.def("sinkhorn_with_grads_implicit(Tensor log_alpha, Tensor grad_P, float tau, int n_iters, int backward_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor[]");
}

#endif // USE_TORCH_LIBRARY
