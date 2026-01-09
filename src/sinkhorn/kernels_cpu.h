/**
 * @file kernels_cpu.h
 * @brief CPU kernel declarations for Sinkhorn algorithm
 */

#pragma once

namespace dot {
namespace sinkhorn {

/**
 * @brief Forward pass of Sinkhorn algorithm on CPU
 *
 * @param cost Cost matrix (B, M, N)
 * @param transport Output transport plan (B, M, N)
 * @param a Source distribution (B, M) or nullptr for uniform
 * @param b Target distribution (B, N) or nullptr for uniform
 * @param B Batch size
 * @param M Source size
 * @param N Target size
 * @param reg Regularization parameter
 * @param max_iter Maximum iterations
 * @param tol Convergence tolerance
 */
void sinkhorn_forward_cpu(
    const float* cost,
    float* transport,
    const float* a,
    const float* b,
    int B, int M, int N,
    float reg,
    int max_iter,
    float tol
);

/**
 * @brief Backward pass of Sinkhorn algorithm on CPU
 *
 * @param grad_output Gradient w.r.t. transport plan (B, M, N)
 * @param cost Cost matrix (B, M, N)
 * @param transport Transport plan from forward (B, M, N)
 * @param grad_cost Output gradient w.r.t. cost (B, M, N)
 * @param B Batch size
 * @param M Source size
 * @param N Target size
 * @param reg Regularization parameter
 */
void sinkhorn_backward_cpu(
    const float* grad_output,
    const float* cost,
    const float* transport,
    float* grad_cost,
    int B, int M, int N,
    float reg
);

} // namespace sinkhorn
} // namespace dot
