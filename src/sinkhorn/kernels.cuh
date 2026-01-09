/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for Sinkhorn algorithm
 */

#pragma once

#include <cuda_runtime.h>

namespace dot {
namespace sinkhorn {

/**
 * @brief Forward pass of Sinkhorn algorithm on CUDA
 */
void sinkhorn_forward_cuda(
    const float* cost,
    float* transport,
    const float* a,
    const float* b,
    int B, int M, int N,
    float reg,
    int max_iter,
    float tol,
    cudaStream_t stream = 0
);

/**
 * @brief Backward pass of Sinkhorn algorithm on CUDA
 */
void sinkhorn_backward_cuda(
    const float* grad_output,
    const float* cost,
    const float* transport,
    float* grad_cost,
    int B, int M, int N,
    float reg,
    cudaStream_t stream = 0
);

} // namespace sinkhorn
} // namespace dot
