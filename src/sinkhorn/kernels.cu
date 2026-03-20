/**
 * @file kernels.cu
 * @brief Sinkhorn CUDA Kernels
 */

#include "kernels.cuh"
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

namespace dot {
namespace sinkhorn {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

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

__global__ void sinkhorn_init_kernel(
    const float* __restrict__ log_alpha,
    float* __restrict__ log_P,
    int B, int n, int m, float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n * m;

    if (idx < total) {
        log_P[idx] = log_alpha[idx] * inv_tau;
    }
}

__global__ void sinkhorn_row_norm_kernel(
    float* __restrict__ log_P,
    const float* __restrict__ log_a,
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    float* row = log_P + b * n * m + i * m;

    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        sum += safe_exp(row[j] - max_val);
    }
    __shared__ float s_lse;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = max_val + logf(block_sum);
    __syncthreads();
    float lse = s_lse;
    float target = log_a != nullptr ? log_a[b * n + i] : -logf(static_cast<float>(n));

    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        row[j] -= (lse - target);
    }
}

__global__ void sinkhorn_col_norm_kernel(
    float* __restrict__ log_P,
    const float* __restrict__ log_b,
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    float* base = log_P + b * n * m + j;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, base[i * m]);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += safe_exp(base[i * m] - max_val);
    }
    __shared__ float s_lse;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = max_val + logf(block_sum);
    __syncthreads();
    float lse = s_lse;
    float target = log_b != nullptr ? log_b[b * m + j] : -logf(static_cast<float>(m));

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        base[i * m] -= (lse - target);
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
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int k = col_idx % m;

    if (b >= B) return;

    int nm = n * m;
    const float* Gamma_b = Gamma_in + b * nm;
    const float* X_b = log_X + b * nm;
    float* out_b = Gamma_out + b * nm;

    float s_k = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_k += Gamma_b[i * m + k];
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(s_k, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    s_k = s_sum;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float q_ik = safe_exp(X_b[i * m + k]);
        out_b[i * m + k] = Gamma_b[i * m + k] - q_ik * s_k;
    }
}

__global__ void sinkhorn_backward_row_kernel(
    const float* __restrict__ Gamma_Y,
    const float* __restrict__ log_Y,
    float* __restrict__ Gamma_out,
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    int nm = n * m;
    const float* Gamma_Y_b = Gamma_Y + b * nm;
    const float* Y_b = log_Y + b * nm;
    float* out_b = Gamma_out + b * nm;

    float t_i = 0.0f;
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
        t_i += Gamma_Y_b[i * m + k];
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(t_i, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    t_i = s_sum;

    for (int k = threadIdx.x; k < m; k += blockDim.x) {
        float p_ik = safe_exp(Y_b[i * m + k]);
        out_b[i * m + k] = Gamma_Y_b[i * m + k] - p_ik * t_i;
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
    int B, int n, int m,
    float neg_inv_tau2
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int b = blockIdx.x;
    if (b >= B) return;

    int nm = n * m;
    const float* Gamma_b = Gamma + b * nm;
    const float* alpha_b = log_alpha + b * nm;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < nm; idx += blockDim.x) {
        sum += Gamma_b[idx] * alpha_b[idx];
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        grad_tau[b] = block_sum * neg_inv_tau2;
    }
}

__global__ void sinkhorn_implicit_update_lambda_kernel(
    const float* __restrict__ P,
    const float* __restrict__ G,
    const float* __restrict__ mu,
    float* __restrict__ lambda_,
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    int nm = n * m;
    const float* P_b = P + b * nm;
    const float* G_b = G + b * nm;
    const float* mu_b = mu + b * m;
    float* lambda_b = lambda_ + b * n;

    float sum_PG = 0.0f;
    float sum_P_mu = 0.0f;
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
        float p_ik = P_b[i * m + k];
        sum_PG += p_ik * G_b[i * m + k];
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
    int B, int n, int m
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int k = col_idx % m;

    if (b >= B) return;

    int nm = n * m;
    const float* P_b = P + b * nm;
    const float* G_b = G + b * nm;
    const float* lambda_b = lambda_ + b * n;
    float* mu_b = mu + b * m;

    float sum_PG = 0.0f;
    float sum_P_lambda = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p_ik = P_b[i * m + k];
        sum_PG += p_ik * G_b[i * m + k];
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
    int B, int n, int m,
    float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n * m;
    int nm = n * m;

    if (idx < total) {
        int b = idx / nm;
        int rem = idx % nm;
        int i = rem / m;
        int k = rem % m;

        float centered = G[idx] - lambda_[b * n + i] - mu[b * m + k];
        grad_log_alpha[idx] = P[idx] * centered * inv_tau;
    }
}

__global__ void sinkhorn_implicit_tau_kernel(
    const float* __restrict__ grad_log_alpha,
    const float* __restrict__ log_alpha,
    float* __restrict__ grad_tau,
    int B, int n, int m,
    float inv_tau
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int b = blockIdx.x;
    if (b >= B) return;

    int nm = n * m;
    const float* grad_b = grad_log_alpha + b * nm;
    const float* alpha_b = log_alpha + b * nm;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < nm; idx += blockDim.x) {
        sum += grad_b[idx] * alpha_b[idx];
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        grad_tau[b] = -block_sum * inv_tau;
    }
}

void sinkhorn_forward_cuda(
    const float* log_alpha,
    float* log_P,
    const float* log_a,
    const float* log_b,
    int B, int n, int m,
    float tau,
    int n_iters,
    bool return_log,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    int num_rows = B * n;
    int num_cols = B * m;
    float inv_tau = 1.0f / tau;

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_alpha, log_P, B, n, m, inv_tau);

    for (int iter = 0; iter < n_iters; ++iter) {
        sinkhorn_row_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(log_P, log_a, B, n, m);
        sinkhorn_col_norm_kernel<<<num_cols, BLOCK_SIZE, 0, stream>>>(log_P, log_b, B, n, m);
    }

    if (!return_log) {
        sinkhorn_exp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_P, total);
    }
}

void sinkhorn_forward_with_intermediates_cuda(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    const float* log_a,
    const float* log_b,
    int B, int n, int m,
    float tau,
    int n_iters,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    int num_rows = B * n;
    int num_cols = B * m;
    float inv_tau = 1.0f / tau;

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_alpha, log_X, B, n, m, inv_tau);

    for (int t = 0; t < n_iters; ++t) {
        float* X_t = log_X + t * total;
        float* Y_t = log_Y + t * total;
        float* X_t1 = log_X + (t + 1) * total;

        sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(X_t, Y_t, total);
        sinkhorn_row_norm_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Y_t, log_a, B, n, m);
        sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(Y_t, X_t1, total);
        sinkhorn_col_norm_kernel<<<num_cols, BLOCK_SIZE, 0, stream>>>(X_t1, log_b, B, n, m);
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
    const float* log_a,
    const float* log_b,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n, int m,
    float tau,
    int n_iters,
    cudaStream_t stream
) {
    int total = B * n * m;
    int num_rows = B * n;
    int num_cols = B * m;
    float inv_tau = 1.0f / tau;
    float neg_inv_tau2 = -1.0f / (tau * tau);

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* Gamma;
    float* Gamma_Y;
    cudaMalloc(&Gamma, total * sizeof(float));
    cudaMalloc(&Gamma_Y, total * sizeof(float));

    (void)log_a;
    (void)log_b;
    sinkhorn_backward_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_P, P, Gamma, total);

    for (int t = n_iters - 1; t >= 0; --t) {
        const float* X_t1 = log_X + (t + 1) * total;
        const float* Y_t = log_Y + t * total;

        sinkhorn_backward_col_kernel<<<num_cols, BLOCK_SIZE, 0, stream>>>(Gamma, X_t1, Gamma_Y, B, n, m);
        sinkhorn_backward_row_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Gamma_Y, Y_t, Gamma, B, n, m);
    }

    sinkhorn_backward_final_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(Gamma, grad_log_alpha, inv_tau, total);
    sinkhorn_backward_tau_kernel<<<B, BLOCK_SIZE, 0, stream>>>(Gamma, log_alpha, grad_tau, B, n, m, neg_inv_tau2);

    cudaFree(Gamma);
    cudaFree(Gamma_Y);
}

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
    cudaStream_t stream
) {
    int total = B * n * m;
    int num_rows = B * n;
    int num_cols = B * m;
    float inv_tau = 1.0f / tau;

    (void)log_a;
    (void)log_b;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* G;
    float* lambda_;
    float* mu;
    cudaMalloc(&G, total * sizeof(float));
    cudaMalloc(&lambda_, B * n * sizeof(float));
    cudaMalloc(&mu, B * m * sizeof(float));

    sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_P, G, total);
    cudaMemsetAsync(lambda_, 0, B * n * sizeof(float), stream);
    cudaMemsetAsync(mu, 0, B * m * sizeof(float), stream);

    for (int iter = 0; iter < max_iters; ++iter) {
        sinkhorn_implicit_update_lambda_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(P, G, mu, lambda_, B, n, m);
        sinkhorn_implicit_update_mu_kernel<<<num_cols, BLOCK_SIZE, 0, stream>>>(P, G, lambda_, mu, B, n, m);
    }

    sinkhorn_implicit_final_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(P, G, lambda_, mu, grad_log_alpha, B, n, m, inv_tau);
    sinkhorn_implicit_tau_kernel<<<B, BLOCK_SIZE, 0, stream>>>(grad_log_alpha, log_alpha, grad_tau, B, n, m, inv_tau);

    cudaFree(G);
    cudaFree(lambda_);
    cudaFree(mu);
}

} // namespace sinkhorn
} // namespace dot
