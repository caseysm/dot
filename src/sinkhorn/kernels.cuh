/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for Sinkhorn algorithm
 *
 * GPU-accelerated Sinkhorn-Knopp algorithm with backward passes.
 * Converts log-space scores to doubly-stochastic matrix (soft permutation).
 *
 * Two backward pass implementations:
 * 1. Unrolled: Differentiate through T iterations explicitly (exact for finite T)
 * 2. Implicit: Use implicit function theorem at convergence (memory efficient)
 */

#pragma once

#include <cuda_runtime.h>

namespace dot {
namespace sinkhorn {

constexpr int kSinkhornBlockSize = 256;
constexpr int kSinkhornLargeBlockSize = 1024;
constexpr int kSinkhornAdaptiveBlockThreshold = 512;

inline int sinkhorn_row_block_size(int m) {
    return (m > kSinkhornAdaptiveBlockThreshold && m <= kSinkhornLargeBlockSize)
        ? kSinkhornLargeBlockSize
        : kSinkhornBlockSize;
}

inline int sinkhorn_col_block_size(int n) {
    return (n > kSinkhornAdaptiveBlockThreshold && n <= kSinkhornLargeBlockSize)
        ? kSinkhornLargeBlockSize
        : kSinkhornBlockSize;
}

inline int sinkhorn_row_chunks(int m) {
    int block_size = sinkhorn_row_block_size(m);
    return (m + block_size - 1) / block_size;
}

inline int sinkhorn_col_chunks(int n) {
    int block_size = sinkhorn_col_block_size(n);
    return (n + block_size - 1) / block_size;
}

/**
 * @brief Forward pass of Sinkhorn algorithm on CUDA
 *
 * Computes doubly-stochastic matrix from log_alpha via Sinkhorn iterations.
 *
 * @param log_alpha Input logits [B, n, m]
 * @param log_P Output matrix [B, n, m] (log-space if return_log, else probabilities)
 * @param B Batch size
 * @param n Number of rows
 * @param m Number of columns
 * @param tau Temperature parameter (must be positive)
 * @param n_iters Number of Sinkhorn iterations
 * @param return_log If true, return log-space result; else return probabilities
 * @param stream CUDA stream
 */
void sinkhorn_forward_cuda(
    const float* log_alpha,
    float* log_P,
    const float* log_a,
    const float* log_b,
    float* row_partial_max,
    float* row_partial_sum,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int row_chunks,
    int col_chunks,
    float tau,
    int n_iters,
    bool return_log,
    cudaStream_t stream = 0
);

/**
 * @brief Initialize dual-space Sinkhorn state on CUDA
 *
 * Computes log_K = log_alpha / tau once and zeros log_u/log_v.
 */
void sinkhorn_dual_init_cuda(
    const float* log_alpha,
    float* log_K,
    float* log_u,
    float* log_v,
    int B, int n, int m,
    float tau,
    cudaStream_t stream = 0
);

/**
 * @brief Recompute log_K = log_alpha / tau without touching log_u/log_v
 */
void sinkhorn_dual_rescale_cuda(
    const float* log_alpha,
    float* log_K,
    int B, int n, int m,
    float tau,
    cudaStream_t stream = 0
);

/**
 * @brief Dual-space row update: log_u = log_a - logsumexp(log_K + log_v)
 */
void sinkhorn_dual_row_update_cuda(
    const float* log_K,
    const float* log_v,
    float* log_u,
    const float* log_a,
    float* row_partial_max,
    float* row_partial_sum,
    int B, int n, int m,
    int row_chunks,
    cudaStream_t stream = 0
);

/**
 * @brief Dual-space column update: log_v = log_b - logsumexp(log_K + log_u)
 */
void sinkhorn_dual_col_update_cuda(
    const float* log_K,
    const float* log_u,
    float* log_v,
    const float* log_b,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int col_chunks,
    cudaStream_t stream = 0
);

/**
 * @brief Materialize P = exp(log_u + log_K + log_v) or return log-space form
 */
void sinkhorn_dual_materialize_cuda(
    const float* log_K,
    const float* log_u,
    const float* log_v,
    float* output,
    int B, int n, int m,
    bool return_log,
    cudaStream_t stream = 0
);

/**
 * @brief Compute max(abs(lhs - rhs)) over a flat vector
 */
void sinkhorn_max_abs_diff_cuda(
    const float* lhs,
    const float* rhs,
    float* output,
    int total,
    cudaStream_t stream = 0
);

/**
 * @brief Dual-space forward with vector-history storage for unrolled backward
 *
 * Stores log_u/log_v after each iteration, requiring O(T * (n + m)) memory.
 */
void sinkhorn_dual_forward_with_intermediates_cuda(
    const float* log_alpha,
    float* P,
    float* log_K,
    float* log_u_hist,
    float* log_v_hist,
    const float* log_a,
    const float* log_b,
    float* row_partial_max,
    float* row_partial_sum,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int row_chunks,
    int col_chunks,
    float tau,
    int n_iters,
    cudaStream_t stream = 0
);

/**
 * @brief Estimate Sinkhorn contraction via deflated power iteration
 *
 * Computes one spectral estimate per batch element without materializing the
 * row-stochastic matrix explicitly. The kernel precomputes row logsumexp values
 * once, then applies the deflated P and P^T operators in-place on vector buffers.
 *
 * @param log_alpha Input logits [B, n, m]
 * @param tau_estimates Output contraction estimates [B]
 * @param row_lse Row logsumexp scratch buffer [B, n]
 * @param v_buf Right-vector scratch buffer [B, m]
 * @param u_buf Left-vector scratch buffer [B, n]
 * @param B Batch size
 * @param n Number of rows
 * @param m Number of columns
 * @param tau Temperature parameter (must be positive)
 * @param n_power Number of power iterations
 * @param stream CUDA stream
 */
void sinkhorn_spectral_preflight_cuda(
    const float* log_alpha,
    float* tau_estimates,
    float* row_lse,
    float* v_buf,
    float* u_buf,
    int B, int n, int m,
    float tau,
    int n_power,
    cudaStream_t stream = 0
);

/**
 * @brief Forward pass with intermediate storage for unrolled backward
 *
 * Stores all intermediate values needed for exact gradient computation.
 *
 * @param log_alpha Input logits [B, n, m]
 * @param P Output probabilities [B, n, m]
 * @param log_X Intermediate values after column norm [B, T+1, n, m]
 * @param log_Y Intermediate values after row norm [B, T, n, m]
 * @param B Batch size
 * @param n Number of rows
 * @param m Number of columns
 * @param tau Temperature parameter
 * @param n_iters Number of iterations
 * @param stream CUDA stream
 */
void sinkhorn_forward_with_intermediates_cuda(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    const float* log_a,
    const float* log_b,
    float* row_partial_max,
    float* row_partial_sum,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int row_chunks,
    int col_chunks,
    float tau,
    int n_iters,
    cudaStream_t stream = 0
);

/**
 * @brief Unrolled backward pass through Sinkhorn iterations
 *
 * Computes exact gradients by backpropagating through all iterations.
 * Requires intermediate values from forward_with_intermediates.
 *
 * @param log_alpha Input logits [B, n, m]
 * @param P Output from forward [B, n, m]
 * @param grad_P Upstream gradient [B, n, m]
 * @param log_X Stored intermediates [B, T+1, n, m]
 * @param log_Y Stored intermediates [B, T, n, m]
 * @param grad_log_alpha Output gradient [B, n, m]
 * @param grad_tau Output gradient w.r.t. tau [B]
 * @param B Batch size
 * @param n Number of rows
 * @param m Number of columns
 * @param tau Temperature parameter
 * @param n_iters Number of iterations
 * @param stream CUDA stream
 */
void sinkhorn_backward_unrolled_cuda(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    const float* log_X,
    const float* log_Y,
    const float* log_a,
    const float* log_b,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n, int m,
    float tau,
    int n_iters,
    cudaStream_t stream = 0
);

/**
 * @brief Unrolled backward pass using dual-state histories
 *
 * Reconstructs the per-iteration normalized matrices on the fly from stored
 * log_u/log_v histories, keeping storage at O(T * (n + m)).
 */
void sinkhorn_backward_unrolled_dual_cuda(
    const float* log_alpha,
    const float* log_K,
    const float* P,
    const float* grad_P,
    const float* log_u_hist,
    const float* log_v_hist,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n, int m,
    float tau,
    int n_iters,
    cudaStream_t stream = 0
);

/**
 * @brief Implicit backward pass using implicit function theorem
 *
 * Memory-efficient gradient computation that doesn't require storing intermediates.
 * Uses fixed-point iteration to solve the adjoint system at convergence.
 *
 * @param log_alpha Input logits [B, n, m]
 * @param P Converged output [B, n, m]
 * @param grad_P Upstream gradient [B, n, m]
 * @param grad_log_alpha Output gradient [B, n, m]
 * @param grad_tau Output gradient w.r.t. tau [B]
 * @param B Batch size
 * @param n Number of rows
 * @param m Number of columns
 * @param tau Temperature parameter
 * @param max_iters Max iterations for adjoint solve
 * @param stream CUDA stream
 */
void sinkhorn_backward_implicit_cuda(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    const float* log_a,
    const float* log_b,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n, int m,
    float tau,
    int max_iters,
    cudaStream_t stream = 0
);

} // namespace sinkhorn
} // namespace dot
