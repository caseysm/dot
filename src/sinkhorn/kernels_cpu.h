/**
 * @file kernels_cpu.h
 * @brief CPU kernel declarations for Sinkhorn algorithm
 *
 * Sinkhorn-Knopp algorithm with backward passes.
 *
 * Two backward pass implementations:
 * 1. Unrolled: Differentiate through T iterations explicitly (exact for finite T)
 * 2. Implicit: Use implicit function theorem at convergence (memory efficient)
 */

#pragma once

namespace dot {
namespace sinkhorn {

/**
 * @brief Forward pass of Sinkhorn algorithm on CPU
 *
 * @param log_alpha Input logits [B, n, n]
 * @param log_P Output matrix [B, n, n]
 * @param B Batch size
 * @param n Matrix dimension
 * @param tau Temperature parameter
 * @param n_iters Number of Sinkhorn iterations
 * @param return_log If true, return log-space; else probabilities
 */
void sinkhorn_forward_cpu(
    const float* log_alpha,
    float* log_P,
    int B, int n,
    float tau,
    int n_iters,
    bool return_log
);

/**
 * @brief Forward pass with intermediate storage for unrolled backward
 *
 * @param log_alpha Input logits [B, n, n]
 * @param P Output probabilities [B, n, n]
 * @param log_X Intermediate values after column norm [B, T+1, n, n]
 * @param log_Y Intermediate values after row norm [B, T, n, n]
 * @param B Batch size
 * @param n Matrix dimension
 * @param tau Temperature parameter
 * @param n_iters Number of iterations
 */
void sinkhorn_forward_with_intermediates_cpu(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    int B, int n,
    float tau,
    int n_iters
);

/**
 * @brief Unrolled backward pass through Sinkhorn iterations
 *
 * @param log_alpha Input logits [B, n, n]
 * @param P Output from forward [B, n, n]
 * @param grad_P Upstream gradient [B, n, n]
 * @param log_X Stored intermediates [B, T+1, n, n]
 * @param log_Y Stored intermediates [B, T, n, n]
 * @param grad_log_alpha Output gradient [B, n, n]
 * @param grad_tau Output gradient w.r.t. tau [B]
 * @param B Batch size
 * @param n Matrix dimension
 * @param tau Temperature parameter
 * @param n_iters Number of iterations
 */
void sinkhorn_backward_unrolled_cpu(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    const float* log_X,
    const float* log_Y,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int n_iters
);

/**
 * @brief Implicit backward pass using implicit function theorem
 *
 * @param log_alpha Input logits [B, n, n]
 * @param P Converged output [B, n, n]
 * @param grad_P Upstream gradient [B, n, n]
 * @param grad_log_alpha Output gradient [B, n, n]
 * @param grad_tau Output gradient w.r.t. tau [B]
 * @param B Batch size
 * @param n Matrix dimension
 * @param tau Temperature parameter
 * @param max_iters Max iterations for adjoint solve
 * @param tol Convergence tolerance
 */
void sinkhorn_backward_implicit_cpu(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int max_iters,
    float tol
);

} // namespace sinkhorn
} // namespace dot
