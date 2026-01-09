/**
 * @file kernels.cu
 * @brief Sinkhorn CUDA Kernels
 *
 * GPU-accelerated Sinkhorn-Knopp algorithm with backward passes.
 * Converts log-space scores to doubly-stochastic matrix (soft permutation).
 *
 * Input:  log_alpha [B, n, n] - logits (use -D/sigma for distance matrix D)
 * Output: P [B, n, n] - doubly-stochastic soft permutation matrix
 *
 * Two backward pass implementations:
 * 1. Unrolled: Differentiate through T iterations explicitly (exact for finite T)
 * 2. Implicit: Use implicit function theorem at convergence (memory efficient)
 */

#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace dot {
namespace sinkhorn {

// =============================================================================
// Constants
// =============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__ float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return expf(x);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);

    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// =============================================================================
// Forward Kernels
// =============================================================================

__global__ void sinkhorn_init_kernel(
    const float* __restrict__ log_alpha,
    float* __restrict__ log_P,
    int B, int n, float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n * n;

    if (idx < total) {
        log_P[idx] = log_alpha[idx] * inv_tau;
    }
}

__global__ void sinkhorn_row_norm_kernel(
    float* __restrict__ log_P,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    float* row = log_P + b * n * n + i * n;

    // Find max in row
    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    // Compute sum(exp(x - max))
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sum += safe_exp(row[j] - max_val);
    }
    __shared__ float s_lse;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = max_val + logf(block_sum);
    __syncthreads();
    float lse = s_lse;

    // Subtract logsumexp
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] -= lse;
    }
}

__global__ void sinkhorn_col_norm_kernel(
    float* __restrict__ log_P,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / n;
    int j = col_idx % n;

    if (b >= B) return;

    float* base = log_P + b * n * n + j;

    // Find max in column
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, base[i * n]);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    // Compute sum(exp(x - max))
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += safe_exp(base[i * n] - max_val);
    }
    __shared__ float s_lse;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = max_val + logf(block_sum);
    __syncthreads();
    float lse = s_lse;

    // Subtract logsumexp
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        base[i * n] -= lse;
    }
}

__global__ void sinkhorn_exp_kernel(
    float* __restrict__ log_P,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        log_P[idx] = safe_exp(log_P[idx]);
    }
}

__global__ void sinkhorn_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// Backward Kernels - Unrolled
// =============================================================================

__global__ void sinkhorn_backward_init_kernel(
    const float* __restrict__ grad_P,
    const float* __restrict__ P,
    float* __restrict__ Gamma,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        Gamma[idx] = grad_P[idx] * P[idx];
    }
}

__global__ void sinkhorn_backward_col_kernel(
    const float* __restrict__ Gamma_in,
    const float* __restrict__ log_X,
    float* __restrict__ Gamma_out,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / n;
    int k = col_idx % n;

    if (b >= B) return;

    int n2 = n * n;
    const float* Gamma_b = Gamma_in + b * n2;
    const float* X_b = log_X + b * n2;
    float* out_b = Gamma_out + b * n2;

    // Compute s_k = sum_i Gamma_{i,k}
    float s_k = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_k += Gamma_b[i * n + k];
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(s_k, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    s_k = s_sum;

    // Gamma_Y_{i,k} = Gamma_{i,k} - exp(X_{i,k}) * s_k
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float q_ik = safe_exp(X_b[i * n + k]);
        out_b[i * n + k] = Gamma_b[i * n + k] - q_ik * s_k;
    }
}

__global__ void sinkhorn_backward_row_kernel(
    const float* __restrict__ Gamma_Y,
    const float* __restrict__ log_Y,
    float* __restrict__ Gamma_out,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    int n2 = n * n;
    const float* Gamma_Y_b = Gamma_Y + b * n2;
    const float* Y_b = log_Y + b * n2;
    float* out_b = Gamma_out + b * n2;

    // Compute t_i = sum_k Gamma_Y_{i,k}
    float t_i = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        t_i += Gamma_Y_b[i * n + k];
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(t_i, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    t_i = s_sum;

    // Gamma^(t-1)_{i,k} = Gamma_Y_{i,k} - exp(Y_{i,k}) * t_i
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float p_ik = safe_exp(Y_b[i * n + k]);
        out_b[i * n + k] = Gamma_Y_b[i * n + k] - p_ik * t_i;
    }
}

__global__ void sinkhorn_backward_final_kernel(
    const float* __restrict__ Gamma,
    float* __restrict__ grad_log_alpha,
    float inv_tau,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad_log_alpha[idx] = Gamma[idx] * inv_tau;
    }
}

__global__ void sinkhorn_backward_tau_kernel(
    const float* __restrict__ Gamma,
    const float* __restrict__ log_alpha,
    float* __restrict__ grad_tau,
    int B, int n,
    float neg_inv_tau2
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int b = blockIdx.x;
    if (b >= B) return;

    int n2 = n * n;
    const float* Gamma_b = Gamma + b * n2;
    const float* alpha_b = log_alpha + b * n2;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < n2; idx += blockDim.x) {
        sum += Gamma_b[idx] * alpha_b[idx];
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        grad_tau[b] = block_sum * neg_inv_tau2;
    }
}

// =============================================================================
// Backward Kernels - Implicit
// =============================================================================

__global__ void sinkhorn_implicit_update_lambda_kernel(
    const float* __restrict__ P,
    const float* __restrict__ G,
    const float* __restrict__ mu,
    float* __restrict__ lambda_,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    int n2 = n * n;
    const float* P_b = P + b * n2;
    const float* G_b = G + b * n2;
    const float* mu_b = mu + b * n;
    float* lambda_b = lambda_ + b * n;

    float sum_PG = 0.0f;
    float sum_P_mu = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float p_ik = P_b[i * n + k];
        sum_PG += p_ik * G_b[i * n + k];
        sum_P_mu += p_ik * mu_b[k];
    }

    float block_sum_PG = block_reduce_sum(sum_PG, shared);
    __syncthreads();
    float block_sum_P_mu = block_reduce_sum(sum_P_mu, shared);

    if (threadIdx.x == 0) {
        lambda_b[i] = block_sum_PG - block_sum_P_mu;
    }
}

__global__ void sinkhorn_implicit_update_mu_kernel(
    const float* __restrict__ P,
    const float* __restrict__ G,
    const float* __restrict__ lambda_,
    float* __restrict__ mu,
    int B, int n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / n;
    int k = col_idx % n;

    if (b >= B) return;

    int n2 = n * n;
    const float* P_b = P + b * n2;
    const float* G_b = G + b * n2;
    const float* lambda_b = lambda_ + b * n;
    float* mu_b = mu + b * n;

    float sum_PG = 0.0f;
    float sum_P_lambda = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p_ik = P_b[i * n + k];
        sum_PG += p_ik * G_b[i * n + k];
        sum_P_lambda += p_ik * lambda_b[i];
    }

    float block_sum_PG = block_reduce_sum(sum_PG, shared);
    __syncthreads();
    float block_sum_P_lambda = block_reduce_sum(sum_P_lambda, shared);

    if (threadIdx.x == 0) {
        mu_b[k] = block_sum_PG - block_sum_P_lambda;
    }
}

__global__ void sinkhorn_implicit_final_kernel(
    const float* __restrict__ P,
    const float* __restrict__ G,
    const float* __restrict__ lambda_,
    const float* __restrict__ mu,
    float* __restrict__ grad_log_alpha,
    int B, int n,
    float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n * n;
    int n2 = n * n;

    if (idx < total) {
        int b = idx / n2;
        int rem = idx % n2;
        int i = rem / n;
        int k = rem % n;

        float centered = G[idx] - lambda_[b * n + i] - mu[b * n + k];
        grad_log_alpha[idx] = P[idx] * centered * inv_tau;
    }
}

__global__ void sinkhorn_implicit_tau_kernel(
    const float* __restrict__ grad_log_alpha,
    const float* __restrict__ log_alpha,
    float* __restrict__ grad_tau,
    int B, int n,
    float inv_tau
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int b = blockIdx.x;
    if (b >= B) return;

    int n2 = n * n;
    const float* grad_b = grad_log_alpha + b * n2;
    const float* alpha_b = log_alpha + b * n2;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < n2; idx += blockDim.x) {
        sum += grad_b[idx] * alpha_b[idx];
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        grad_tau[b] = -block_sum * inv_tau;
    }
}

// =============================================================================
// Host Functions
// =============================================================================

void sinkhorn_forward_cuda(
    const float* log_alpha,
    float* log_P,
    int B, int n,
    float tau,
    int n_iters,
    bool return_log,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * n;
    int num_rows = B * n;
    float inv_tau = 1.0f / tau;

    int blocks_init = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_init_kernel<<<blocks_init, BLOCK_SIZE, 0, stream>>>(log_alpha, log_P, B, n, inv_tau);

    for (int iter = 0; iter < n_iters; ++iter) {
        sinkhorn_row_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(log_P, B, n);
        sinkhorn_col_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(log_P, B, n);
    }

    if (!return_log) {
        int blocks_exp = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sinkhorn_exp_kernel<<<blocks_exp, BLOCK_SIZE, 0, stream>>>(log_P, total);
    }
}

void sinkhorn_forward_with_intermediates_cuda(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    int B, int n,
    float tau,
    int n_iters,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * n;
    int num_rows = B * n;
    float inv_tau = 1.0f / tau;

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_alpha, log_X, B, n, inv_tau);

    for (int t = 0; t < n_iters; ++t) {
        float* X_t = log_X + t * total;
        float* Y_t = log_Y + t * total;
        float* X_t1 = log_X + (t + 1) * total;

        sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(X_t, Y_t, total);
        sinkhorn_row_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Y_t, B, n);
        sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(Y_t, X_t1, total);
        sinkhorn_col_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(X_t1, B, n);
    }

    float* X_T = log_X + n_iters * total;
    sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(X_T, P, total);
    sinkhorn_exp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(P, total);
}

void sinkhorn_backward_unrolled_cuda(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    const float* log_X,
    const float* log_Y,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int n_iters,
    cudaStream_t stream
) {
    int total = B * n * n;
    int num_rows = B * n;
    float inv_tau = 1.0f / tau;
    float neg_inv_tau2 = -1.0f / (tau * tau);

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* Gamma;
    float* Gamma_Y;
    cudaMalloc(&Gamma, total * sizeof(float));
    cudaMalloc(&Gamma_Y, total * sizeof(float));

    sinkhorn_backward_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_P, P, Gamma, total);

    for (int t = n_iters - 1; t >= 0; --t) {
        const float* X_t1 = log_X + (t + 1) * total;
        const float* Y_t = log_Y + t * total;

        sinkhorn_backward_col_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Gamma, X_t1, Gamma_Y, B, n);
        sinkhorn_backward_row_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Gamma_Y, Y_t, Gamma, B, n);
    }

    sinkhorn_backward_final_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(Gamma, grad_log_alpha, inv_tau, total);
    sinkhorn_backward_tau_kernel<<<B, BLOCK_SIZE, 0, stream>>>(Gamma, log_alpha, grad_tau, B, n, neg_inv_tau2);

    cudaFree(Gamma);
    cudaFree(Gamma_Y);
}

void sinkhorn_backward_implicit_cuda(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int max_iters,
    cudaStream_t stream
) {
    int total = B * n * n;
    int num_rows = B * n;
    float inv_tau = 1.0f / tau;

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* G;
    float* lambda_;
    float* mu;
    cudaMalloc(&G, total * sizeof(float));
    cudaMalloc(&lambda_, B * n * sizeof(float));
    cudaMalloc(&mu, B * n * sizeof(float));

    sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_P, G, total);
    cudaMemsetAsync(lambda_, 0, B * n * sizeof(float), stream);
    cudaMemsetAsync(mu, 0, B * n * sizeof(float), stream);

    for (int iter = 0; iter < max_iters; ++iter) {
        sinkhorn_implicit_update_lambda_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(P, G, mu, lambda_, B, n);
        sinkhorn_implicit_update_mu_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(P, G, lambda_, mu, B, n);
    }

    sinkhorn_implicit_final_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(P, G, lambda_, mu, grad_log_alpha, B, n, inv_tau);
    sinkhorn_implicit_tau_kernel<<<B, BLOCK_SIZE, 0, stream>>>(grad_log_alpha, log_alpha, grad_tau, B, n, inv_tau);

    cudaFree(G);
    cudaFree(lambda_);
    cudaFree(mu);
}

} // namespace sinkhorn
} // namespace dot
