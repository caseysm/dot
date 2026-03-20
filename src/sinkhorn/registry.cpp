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

    // Forward + backward with unrolled differentiation through iterations
    // Returns [P, grad_log_alpha, grad_tau]
    m.def("sinkhorn_with_grads_unrolled(Tensor log_alpha, Tensor grad_P, float tau, int n_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor[]");

    // Forward + backward with implicit differentiation (memory efficient)
    // Returns [P, grad_log_alpha, grad_tau]
    m.def("sinkhorn_with_grads_implicit(Tensor log_alpha, Tensor grad_P, float tau, int n_iters, int backward_iters, Tensor? log_a=None, Tensor? log_b=None) -> Tensor[]");
}

#endif // USE_TORCH_LIBRARY
