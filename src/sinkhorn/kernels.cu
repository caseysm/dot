/**
 * @file kernels.cu
 * @brief Sinkhorn CUDA Kernels
 */

#include "kernels.cuh"
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace dot {
namespace sinkhorn {

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = kSinkhornBlockSize;
constexpr int SPECTRAL_BLOCK_SIZE = kSinkhornLargeBlockSize;
constexpr int MAX_REDUCTION_BLOCK_SIZE = 1024;
constexpr int MAX_REDUCTION_WARPS = MAX_REDUCTION_BLOCK_SIZE / WARP_SIZE;

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

__device__ __forceinline__ void center_vector(
    float* vec,
    int len,
    float* shared,
    float* scalar
) {
    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        sum += vec[idx];
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        *scalar = block_sum / static_cast<float>(len);
    }
    __syncthreads();

    float mean = *scalar;
    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        vec[idx] -= mean;
    }
    __syncthreads();
}

__device__ __forceinline__ float vector_norm(
    float* vec,
    int len,
    float* shared,
    float* scalar
) {
    float norm_sq = 0.0f;
    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        float val = vec[idx];
        norm_sq += val * val;
    }
    float block_norm_sq = block_reduce_sum(norm_sq, shared);
    if (threadIdx.x == 0) {
        *scalar = sqrtf(fmaxf(block_norm_sq, 1.0e-12f));
    }
    __syncthreads();
    return *scalar;
}

__device__ __forceinline__ float center_and_normalize(
    float* vec,
    int len,
    float* shared,
    float* scalar
) {
    center_vector(vec, len, shared, scalar);
    float norm = vector_norm(vec, len, shared, scalar);
    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        vec[idx] /= norm;
    }
    __syncthreads();
    return norm;
}

__global__ void sinkhorn_init_kernel(
    const float* __restrict__ log_alpha,
    float* __restrict__ log_P,
    int total,
    float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        log_P[idx] = log_alpha[idx] * inv_tau;
    }
}

__global__ void sinkhorn_spectral_preflight_kernel(
    const float* __restrict__ log_alpha,
    float* __restrict__ tau_estimates,
    float* __restrict__ row_lse,
    float* __restrict__ v_buf,
    float* __restrict__ u_buf,
    int B, int n, int m,
    float inv_tau,
    int n_power
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float scalar;

    int b = blockIdx.x;
    if (b >= B) return;

    const float* log_alpha_b = log_alpha + b * n * m;
    float* row_lse_b = row_lse + b * n;
    float* v = v_buf + b * m;
    float* u = u_buf + b * n;

    // Precompute row logsumexp values once; each subsequent mat-vec reuses them.
    for (int i = 0; i < n; ++i) {
        const float* row = log_alpha_b + i * m;
        float max_val = -FLT_MAX;
        for (int j = threadIdx.x; j < m; j += blockDim.x) {
            max_val = fmaxf(max_val, row[j] * inv_tau);
        }
        float row_max = block_reduce_max(max_val, shared);
        if (threadIdx.x == 0) {
            scalar = row_max;
        }
        __syncthreads();

        float sum = 0.0f;
        for (int j = threadIdx.x; j < m; j += blockDim.x) {
            sum += safe_exp(row[j] * inv_tau - scalar);
        }
        float row_sum = block_reduce_sum(sum, shared);
        if (threadIdx.x == 0) {
            row_lse_b[i] = scalar + logf(fmaxf(row_sum, 1.0e-12f));
        }
        __syncthreads();
    }

    // Deterministic mean-zero initialization avoids host RNG setup.
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        v[j] = sinf(
            static_cast<float>(j + 1) * 12.9898f +
            static_cast<float>(b + 1) * 78.233f
        );
    }
    __syncthreads();
    center_and_normalize(v, m, shared, &scalar);

    for (int power_iter = 0; power_iter < n_power; ++power_iter) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float* row = log_alpha_b + i * m;
            float lse = row_lse_b[i];
            float dot = 0.0f;
            for (int j = 0; j < m; ++j) {
                float weight = safe_exp(row[j] * inv_tau - lse);
                dot += weight * v[j];
            }
            u[i] = dot;
        }
        __syncthreads();
        center_vector(u, n, shared, &scalar);

        for (int j = threadIdx.x; j < m; j += blockDim.x) {
            float dot = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float* row = log_alpha_b + i * m;
                float weight = safe_exp(row[j] * inv_tau - row_lse_b[i]);
                dot += weight * u[i];
            }
            v[j] = dot;
        }
        __syncthreads();
        center_and_normalize(v, m, shared, &scalar);
    }

    float tau_est = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float* row = log_alpha_b + i * m;
        float lse = row_lse_b[i];
        float dot = 0.0f;
        for (int j = 0; j < m; ++j) {
            float weight = safe_exp(row[j] * inv_tau - lse);
            dot += weight * v[j];
        }
        u[i] = dot;
    }
    __syncthreads();
    center_vector(u, n, shared, &scalar);
    tau_est = vector_norm(u, n, shared, &scalar);

    if (threadIdx.x == 0) {
        tau_estimates[b] = fmaxf(0.0f, fminf(tau_est, 0.999999f));
    }
}

__global__ void sinkhorn_init_vec4_kernel(
    const float4* __restrict__ log_alpha,
    float4* __restrict__ log_P,
    int total_vec4,
    float inv_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec4) {
        float4 a = log_alpha[idx];
        log_P[idx] = make_float4(
            a.x * inv_tau,
            a.y * inv_tau,
            a.z * inv_tau,
            a.w * inv_tau
        );
    }
}

__global__ void sinkhorn_row_norm_kernel(
    float* __restrict__ log_P,
    const float* __restrict__ log_a,
    int B, int n, int m
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];

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

__global__ void sinkhorn_row_norm_phase1_kernel(
    const float* __restrict__ log_P,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_row
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;

    int row_idx = blockIdx.x / chunks_per_row;
    int chunk_idx = blockIdx.x % chunks_per_row;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    const float* row = log_P + b * n * m + i * m;
    int j = chunk_idx * blockDim.x + threadIdx.x;

    float max_val = (j < m) ? row[j] : -FLT_MAX;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = (j < m) ? safe_exp(row[j] - s_max) : 0.0f;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        int partial_idx = row_idx * chunks_per_row + chunk_idx;
        partial_max[partial_idx] = s_max;
        partial_sum[partial_idx] = block_sum;
    }
}

__global__ void sinkhorn_row_norm_phase2_kernel(
    float* __restrict__ log_P,
    const float* __restrict__ log_a,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_row
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_global_max;
    __shared__ float s_lse;

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    const float* row_partials_max = partial_max + row_idx * chunks_per_row;
    const float* row_partials_sum = partial_sum + row_idx * chunks_per_row;
    float* row = log_P + b * n * m + i * m;

    float max_val = -FLT_MAX;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += blockDim.x) {
        max_val = fmaxf(max_val, row_partials_max[chunk]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_global_max = block_max;
    __syncthreads();

    float global_max = s_global_max;
    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += blockDim.x) {
        sum += row_partials_sum[chunk] * safe_exp(row_partials_max[chunk] - global_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = global_max + logf(block_sum);
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
    __shared__ float shared[MAX_REDUCTION_WARPS];

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

__global__ void sinkhorn_col_norm_phase1_kernel(
    const float* __restrict__ log_P,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_col
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;

    int col_idx = blockIdx.x / chunks_per_col;
    int chunk_idx = blockIdx.x % chunks_per_col;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    const float* base = log_P + b * n * m + j;
    int i = chunk_idx * blockDim.x + threadIdx.x;

    float max_val = (i < n) ? base[i * m] : -FLT_MAX;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = (i < n) ? safe_exp(base[i * m] - s_max) : 0.0f;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        int partial_idx = col_idx * chunks_per_col + chunk_idx;
        partial_max[partial_idx] = s_max;
        partial_sum[partial_idx] = block_sum;
    }
}

__global__ void sinkhorn_col_norm_phase2_kernel(
    float* __restrict__ log_P,
    const float* __restrict__ log_b,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_col
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_global_max;
    __shared__ float s_lse;

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    const float* col_partials_max = partial_max + col_idx * chunks_per_col;
    const float* col_partials_sum = partial_sum + col_idx * chunks_per_col;
    float* base = log_P + b * n * m + j;

    float max_val = -FLT_MAX;
    for (int chunk = threadIdx.x; chunk < chunks_per_col; chunk += blockDim.x) {
        max_val = fmaxf(max_val, col_partials_max[chunk]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_global_max = block_max;
    __syncthreads();

    float global_max = s_global_max;
    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < chunks_per_col; chunk += blockDim.x) {
        sum += col_partials_sum[chunk] * safe_exp(col_partials_max[chunk] - global_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_lse = global_max + logf(block_sum);
    __syncthreads();

    float lse = s_lse;
    float target = log_b != nullptr ? log_b[b * m + j] : -logf(static_cast<float>(m));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        base[i * m] -= (lse - target);
    }
}

__global__ void sinkhorn_dual_row_update_kernel(
    const float* __restrict__ log_K,
    const float* __restrict__ log_v,
    float* __restrict__ log_u,
    const float* __restrict__ log_a,
    int B, int n, int m
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;
    __shared__ float s_lse;

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    const float* row = log_K + b * n * m + i * m;
    const float* v = log_v + b * m;

    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j] + v[j]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        sum += safe_exp(row[j] + v[j] - s_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        float target = log_a != nullptr ? log_a[b * n + i] : -logf(static_cast<float>(n));
        s_lse = target - (s_max + logf(fmaxf(block_sum, 1.0e-12f)));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        log_u[b * n + i] = s_lse;
    }
}

__global__ void sinkhorn_dual_row_update_phase1_kernel(
    const float* __restrict__ log_K,
    const float* __restrict__ log_v,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_row
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;

    int row_idx = blockIdx.x / chunks_per_row;
    int chunk_idx = blockIdx.x % chunks_per_row;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    const float* row = log_K + b * n * m + i * m;
    const float* v = log_v + b * m;
    int j = chunk_idx * blockDim.x + threadIdx.x;

    float max_val = (j < m) ? row[j] + v[j] : -FLT_MAX;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = (j < m) ? safe_exp(row[j] + v[j] - s_max) : 0.0f;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        int partial_idx = row_idx * chunks_per_row + chunk_idx;
        partial_max[partial_idx] = s_max;
        partial_sum[partial_idx] = block_sum;
    }
}

__global__ void sinkhorn_dual_row_update_phase2_kernel(
    const float* __restrict__ log_a,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    float* __restrict__ log_u,
    int B, int n,
    int chunks_per_row
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_global_max;
    __shared__ float s_value;

    int row_idx = blockIdx.x;
    int b = row_idx / n;
    int i = row_idx % n;

    if (b >= B) return;

    const float* row_partials_max = partial_max + row_idx * chunks_per_row;
    const float* row_partials_sum = partial_sum + row_idx * chunks_per_row;

    float max_val = -FLT_MAX;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += blockDim.x) {
        max_val = fmaxf(max_val, row_partials_max[chunk]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_global_max = block_max;
    __syncthreads();

    float global_max = s_global_max;
    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += blockDim.x) {
        sum += row_partials_sum[chunk] * safe_exp(row_partials_max[chunk] - global_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        float target = log_a != nullptr ? log_a[b * n + i] : -logf(static_cast<float>(n));
        s_value = target - (global_max + logf(fmaxf(block_sum, 1.0e-12f)));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        log_u[row_idx] = s_value;
    }
}

__global__ void sinkhorn_dual_col_update_kernel(
    const float* __restrict__ log_K,
    const float* __restrict__ log_u,
    float* __restrict__ log_v,
    const float* __restrict__ log_b,
    int B, int n, int m
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;
    __shared__ float s_lse;

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    const float* base = log_K + b * n * m + j;
    const float* u = log_u + b * n;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, base[i * m] + u[i]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += safe_exp(base[i * m] + u[i] - s_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        float target = log_b != nullptr ? log_b[b * m + j] : -logf(static_cast<float>(m));
        s_lse = target - (s_max + logf(fmaxf(block_sum, 1.0e-12f)));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        log_v[b * m + j] = s_lse;
    }
}

__global__ void sinkhorn_dual_col_update_phase1_kernel(
    const float* __restrict__ log_K,
    const float* __restrict__ log_u,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int B, int n, int m,
    int chunks_per_col
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_max;

    int col_idx = blockIdx.x / chunks_per_col;
    int chunk_idx = blockIdx.x % chunks_per_col;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    const float* base = log_K + b * n * m + j;
    const float* u = log_u + b * n;
    int i = chunk_idx * blockDim.x + threadIdx.x;

    float max_val = (i < n) ? base[i * m] + u[i] : -FLT_MAX;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    float sum = (i < n) ? safe_exp(base[i * m] + u[i] - s_max) : 0.0f;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        int partial_idx = col_idx * chunks_per_col + chunk_idx;
        partial_max[partial_idx] = s_max;
        partial_sum[partial_idx] = block_sum;
    }
}

__global__ void sinkhorn_dual_col_update_phase2_kernel(
    const float* __restrict__ log_b,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    float* __restrict__ log_v,
    int B, int m,
    int chunks_per_col
) {
    __shared__ float shared[MAX_REDUCTION_WARPS];
    __shared__ float s_global_max;
    __shared__ float s_value;

    int col_idx = blockIdx.x;
    int b = col_idx / m;
    int j = col_idx % m;

    if (b >= B) return;

    const float* col_partials_max = partial_max + col_idx * chunks_per_col;
    const float* col_partials_sum = partial_sum + col_idx * chunks_per_col;

    float max_val = -FLT_MAX;
    for (int chunk = threadIdx.x; chunk < chunks_per_col; chunk += blockDim.x) {
        max_val = fmaxf(max_val, col_partials_max[chunk]);
    }
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_global_max = block_max;
    __syncthreads();

    float global_max = s_global_max;
    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < chunks_per_col; chunk += blockDim.x) {
        sum += col_partials_sum[chunk] * safe_exp(col_partials_max[chunk] - global_max);
    }
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        float target = log_b != nullptr ? log_b[b * m + j] : -logf(static_cast<float>(m));
        s_value = target - (global_max + logf(fmaxf(block_sum, 1.0e-12f)));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        log_v[col_idx] = s_value;
    }
}

__global__ void sinkhorn_dual_materialize_kernel(
    const float* __restrict__ log_K,
    const float* __restrict__ log_u,
    const float* __restrict__ log_v,
    float* __restrict__ output,
    int B, int n, int m,
    bool return_log
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n * m;
    if (idx >= total) return;

    int nm = n * m;
    int b = idx / nm;
    int rem = idx % nm;
    int i = rem / m;
    int j = rem % m;

    float log_p = log_K[idx] + log_u[b * n + i] + log_v[b * m + j];
    output[idx] = return_log ? log_p : safe_exp(log_p);
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

__global__ void sinkhorn_exp_vec4_kernel(
    float4* __restrict__ log_P,
    int total_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec4) {
        float4 v = log_P[idx];
        log_P[idx] = make_float4(
            safe_exp(v.x),
            safe_exp(v.y),
            safe_exp(v.z),
            safe_exp(v.w)
        );
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

__global__ void sinkhorn_copy_vec4_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int total_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec4) {
        dst[idx] = src[idx];
    }
}

__global__ void sinkhorn_max_abs_diff_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    int total
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    float local_max = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, fabsf(lhs[idx] - rhs[idx]));
    }

    float block_max = block_reduce_max(local_max, shared);
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(output), __float_as_uint(block_max));
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
    float sum_P = 0.0f;
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
        float p_ik = P_b[i * m + k];
        sum_PG += p_ik * G_b[i * m + k];
        sum_P_mu += p_ik * mu_b[k];
        sum_P += p_ik;
    }

    float block_sum_PG = block_reduce_sum(sum_PG, shared);
    __syncthreads();
    float block_sum_P_mu = block_reduce_sum(sum_P_mu, shared);
    __syncthreads();
    float block_sum_P = block_reduce_sum(sum_P, shared);

    if (threadIdx.x == 0) {
        lambda_b[i] = (block_sum_PG - block_sum_P_mu) / fmaxf(block_sum_P, 1.0e-12f);
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
    float sum_P = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p_ik = P_b[i * m + k];
        sum_PG += p_ik * G_b[i * m + k];
        sum_P_lambda += p_ik * lambda_b[i];
        sum_P += p_ik;
    }

    float block_sum_PG = block_reduce_sum(sum_PG, shared);
    __syncthreads();
    float block_sum_P_lambda = block_reduce_sum(sum_P_lambda, shared);
    __syncthreads();
    float block_sum_P = block_reduce_sum(sum_P, shared);

    if (threadIdx.x == 0) {
        mu_b[k] = (block_sum_PG - block_sum_P_lambda) / fmaxf(block_sum_P, 1.0e-12f);
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

inline bool is_vec4_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignof(float4)) == 0;
}

inline void launch_init(
    const float* log_alpha,
    float* log_P,
    int total,
    float inv_tau,
    cudaStream_t stream
) {
    if (is_vec4_aligned(log_alpha) && is_vec4_aligned(log_P)) {
        int total_vec4 = total / 4;
        if (total_vec4 > 0) {
            int vec_blocks = (total_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_init_vec4_kernel<<<vec_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const float4*>(log_alpha),
                reinterpret_cast<float4*>(log_P),
                total_vec4,
                inv_tau
            );
        }
        int tail = total - total_vec4 * 4;
        if (tail > 0) {
            int blocks = (tail + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                log_alpha + total_vec4 * 4,
                log_P + total_vec4 * 4,
                tail,
                inv_tau
            );
        }
        return;
    }

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_alpha, log_P, total, inv_tau);
}

inline void launch_copy(
    const float* src,
    float* dst,
    int total,
    cudaStream_t stream
) {
    if (is_vec4_aligned(src) && is_vec4_aligned(dst)) {
        int total_vec4 = total / 4;
        if (total_vec4 > 0) {
            int vec_blocks = (total_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_copy_vec4_kernel<<<vec_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const float4*>(src),
                reinterpret_cast<float4*>(dst),
                total_vec4
            );
        }
        int tail = total - total_vec4 * 4;
        if (tail > 0) {
            int blocks = (tail + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                src + total_vec4 * 4,
                dst + total_vec4 * 4,
                tail
            );
        }
        return;
    }

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, total);
}

inline void launch_exp(
    float* log_P,
    int total,
    cudaStream_t stream
) {
    if (is_vec4_aligned(log_P)) {
        int total_vec4 = total / 4;
        if (total_vec4 > 0) {
            int vec_blocks = (total_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_exp_vec4_kernel<<<vec_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<float4*>(log_P),
                total_vec4
            );
        }
        int tail = total - total_vec4 * 4;
        if (tail > 0) {
            int blocks = (tail + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sinkhorn_exp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                log_P + total_vec4 * 4,
                tail
            );
        }
        return;
    }

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_exp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(log_P, total);
}

inline void launch_row_norm(
    float* log_P,
    const float* log_a,
    float* row_partial_max,
    float* row_partial_sum,
    int B, int n, int m,
    int row_chunks,
    cudaStream_t stream
) {
    int num_rows = B * n;
    int row_block_size = sinkhorn_row_block_size(m);
    if (row_chunks > 1) {
        sinkhorn_row_norm_phase1_kernel<<<num_rows * row_chunks, row_block_size, 0, stream>>>(
            log_P, row_partial_max, row_partial_sum, B, n, m, row_chunks
        );
        sinkhorn_row_norm_phase2_kernel<<<num_rows, row_block_size, 0, stream>>>(
            log_P, log_a, row_partial_max, row_partial_sum, B, n, m, row_chunks
        );
        return;
    }

    sinkhorn_row_norm_kernel<<<num_rows, row_block_size, 0, stream>>>(log_P, log_a, B, n, m);
}

inline void launch_col_norm(
    float* log_P,
    const float* log_b,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int col_chunks,
    cudaStream_t stream
) {
    int num_cols = B * m;
    int col_block_size = sinkhorn_col_block_size(n);
    if (col_chunks > 1) {
        sinkhorn_col_norm_phase1_kernel<<<num_cols * col_chunks, col_block_size, 0, stream>>>(
            log_P, col_partial_max, col_partial_sum, B, n, m, col_chunks
        );
        sinkhorn_col_norm_phase2_kernel<<<num_cols, col_block_size, 0, stream>>>(
            log_P, log_b, col_partial_max, col_partial_sum, B, n, m, col_chunks
        );
        return;
    }

    sinkhorn_col_norm_kernel<<<num_cols, col_block_size, 0, stream>>>(log_P, log_b, B, n, m);
}

inline void launch_dual_row_update(
    const float* log_K,
    const float* log_v,
    float* log_u,
    const float* log_a,
    float* row_partial_max,
    float* row_partial_sum,
    int B, int n, int m,
    int row_chunks,
    cudaStream_t stream
) {
    int num_rows = B * n;
    int row_block_size = sinkhorn_row_block_size(m);
    if (row_chunks > 1) {
        sinkhorn_dual_row_update_phase1_kernel<<<num_rows * row_chunks, row_block_size, 0, stream>>>(
            log_K, log_v, row_partial_max, row_partial_sum, B, n, m, row_chunks
        );
        sinkhorn_dual_row_update_phase2_kernel<<<num_rows, row_block_size, 0, stream>>>(
            log_a, row_partial_max, row_partial_sum, log_u, B, n, row_chunks
        );
        return;
    }

    sinkhorn_dual_row_update_kernel<<<num_rows, row_block_size, 0, stream>>>(
        log_K, log_v, log_u, log_a, B, n, m
    );
}

inline void launch_dual_col_update(
    const float* log_K,
    const float* log_u,
    float* log_v,
    const float* log_b,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int col_chunks,
    cudaStream_t stream
) {
    int num_cols = B * m;
    int col_block_size = sinkhorn_col_block_size(n);
    if (col_chunks > 1) {
        sinkhorn_dual_col_update_phase1_kernel<<<num_cols * col_chunks, col_block_size, 0, stream>>>(
            log_K, log_u, col_partial_max, col_partial_sum, B, n, m, col_chunks
        );
        sinkhorn_dual_col_update_phase2_kernel<<<num_cols, col_block_size, 0, stream>>>(
            log_b, col_partial_max, col_partial_sum, log_v, B, m, col_chunks
        );
        return;
    }

    sinkhorn_dual_col_update_kernel<<<num_cols, col_block_size, 0, stream>>>(
        log_K, log_u, log_v, log_b, B, n, m
    );
}

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
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    float inv_tau = 1.0f / tau;

    launch_init(log_alpha, log_P, total, inv_tau, stream);

    for (int iter = 0; iter < n_iters; ++iter) {
        launch_row_norm(log_P, log_a, row_partial_max, row_partial_sum, B, n, m, row_chunks, stream);
        launch_col_norm(log_P, log_b, col_partial_max, col_partial_sum, B, n, m, col_chunks, stream);
    }

    if (!return_log) {
        launch_exp(log_P, total, stream);
    }
}

void sinkhorn_dual_init_cuda(
    const float* log_alpha,
    float* log_K,
    float* log_u,
    float* log_v,
    int B, int n, int m,
    float tau,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    launch_init(log_alpha, log_K, total, 1.0f / tau, stream);
    cudaMemsetAsync(log_u, 0, B * n * sizeof(float), stream);
    cudaMemsetAsync(log_v, 0, B * m * sizeof(float), stream);
}

void sinkhorn_dual_rescale_cuda(
    const float* log_alpha,
    float* log_K,
    int B, int n, int m,
    float tau,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    launch_init(log_alpha, log_K, total, 1.0f / tau, stream);
}

void sinkhorn_dual_row_update_cuda(
    const float* log_K,
    const float* log_v,
    float* log_u,
    const float* log_a,
    float* row_partial_max,
    float* row_partial_sum,
    int B, int n, int m,
    int row_chunks,
    cudaStream_t stream
) {
    launch_dual_row_update(
        log_K,
        log_v,
        log_u,
        log_a,
        row_partial_max,
        row_partial_sum,
        B, n, m,
        row_chunks,
        stream
    );
}

void sinkhorn_dual_col_update_cuda(
    const float* log_K,
    const float* log_u,
    float* log_v,
    const float* log_b,
    float* col_partial_max,
    float* col_partial_sum,
    int B, int n, int m,
    int col_chunks,
    cudaStream_t stream
) {
    launch_dual_col_update(
        log_K,
        log_u,
        log_v,
        log_b,
        col_partial_max,
        col_partial_sum,
        B, n, m,
        col_chunks,
        stream
    );
}

void sinkhorn_dual_materialize_cuda(
    const float* log_K,
    const float* log_u,
    const float* log_v,
    float* output,
    int B, int n, int m,
    bool return_log,
    cudaStream_t stream
) {
    int total = B * n * m;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sinkhorn_dual_materialize_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        log_K, log_u, log_v, output, B, n, m, return_log
    );
}

void sinkhorn_max_abs_diff_cuda(
    const float* lhs,
    const float* rhs,
    float* output,
    int total,
    cudaStream_t stream
) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = blocks > 0 ? blocks : 1;
    sinkhorn_max_abs_diff_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(lhs, rhs, output, total);
}

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
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total_rows = B * n;
    int total_cols = B * m;
    float* log_u;
    float* log_v;
    cudaMalloc(&log_u, total_rows * sizeof(float));
    cudaMalloc(&log_v, total_cols * sizeof(float));

    sinkhorn_dual_init_cuda(log_alpha, log_K, log_u, log_v, B, n, m, tau, stream);

    for (int t = 0; t < n_iters; ++t) {
        sinkhorn_dual_row_update_cuda(
            log_K,
            log_v,
            log_u,
            log_a,
            row_partial_max,
            row_partial_sum,
            B, n, m,
            row_chunks,
            stream
        );
        launch_copy(log_u, log_u_hist + t * total_rows, total_rows, stream);

        sinkhorn_dual_col_update_cuda(
            log_K,
            log_u,
            log_v,
            log_b,
            col_partial_max,
            col_partial_sum,
            B, n, m,
            col_chunks,
            stream
        );
        launch_copy(log_v, log_v_hist + t * total_cols, total_cols, stream);
    }

    sinkhorn_dual_materialize_cuda(log_K, log_u, log_v, P, B, n, m, /*return_log=*/false, stream);

    cudaFree(log_u);
    cudaFree(log_v);
}

void sinkhorn_spectral_preflight_cuda(
    const float* log_alpha,
    float* tau_estimates,
    float* row_lse,
    float* v_buf,
    float* u_buf,
    int B, int n, int m,
    float tau,
    int n_power,
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    sinkhorn_spectral_preflight_kernel<<<B, SPECTRAL_BLOCK_SIZE, 0, stream>>>(
        log_alpha,
        tau_estimates,
        row_lse,
        v_buf,
        u_buf,
        B, n, m,
        1.0f / tau,
        n_power
    );
}

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
    cudaStream_t stream
) {
    if (tau <= 0.0f) return;

    int total = B * n * m;
    float inv_tau = 1.0f / tau;

    launch_init(log_alpha, log_X, total, inv_tau, stream);

    for (int t = 0; t < n_iters; ++t) {
        float* X_t = log_X + t * total;
        float* Y_t = log_Y + t * total;
        float* X_t1 = log_X + (t + 1) * total;

        launch_copy(X_t, Y_t, total, stream);
        launch_row_norm(Y_t, log_a, row_partial_max, row_partial_sum, B, n, m, row_chunks, stream);
        launch_copy(Y_t, X_t1, total, stream);
        launch_col_norm(X_t1, log_b, col_partial_max, col_partial_sum, B, n, m, col_chunks, stream);
    }

    float* X_T = log_X + n_iters * total;
    launch_copy(X_T, P, total, stream);
    launch_exp(P, total, stream);
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
    cudaStream_t stream
) {
    int total = B * n * m;
    int num_rows = B * n;
    int num_cols = B * m;
    int total_rows = B * n;
    int total_cols = B * m;
    float inv_tau = 1.0f / tau;
    float neg_inv_tau2 = -1.0f / (tau * tau);

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* Gamma;
    float* Gamma_Y;
    float* log_X;
    float* log_Y;
    float* zero_v;
    cudaMalloc(&Gamma, total * sizeof(float));
    cudaMalloc(&Gamma_Y, total * sizeof(float));
    cudaMalloc(&log_X, total * sizeof(float));
    cudaMalloc(&log_Y, total * sizeof(float));
    cudaMalloc(&zero_v, total_cols * sizeof(float));
    cudaMemsetAsync(zero_v, 0, total_cols * sizeof(float), stream);

    sinkhorn_backward_init_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_P, P, Gamma, total);

    for (int t = n_iters - 1; t >= 0; --t) {
        const float* log_u_t = log_u_hist + t * total_rows;
        const float* log_v_t = log_v_hist + t * total_cols;
        const float* log_v_prev = (t > 0) ? (log_v_hist + (t - 1) * total_cols) : zero_v;

        sinkhorn_dual_materialize_cuda(log_K, log_u_t, log_v_t, log_X, B, n, m, /*return_log=*/true, stream);
        sinkhorn_dual_materialize_cuda(log_K, log_u_t, log_v_prev, log_Y, B, n, m, /*return_log=*/true, stream);

        sinkhorn_backward_col_kernel<<<num_cols, BLOCK_SIZE, 0, stream>>>(Gamma, log_X, Gamma_Y, B, n, m);
        sinkhorn_backward_row_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(Gamma_Y, log_Y, Gamma, B, n, m);
    }

    sinkhorn_backward_final_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(Gamma, grad_log_alpha, inv_tau, total);
    sinkhorn_backward_tau_kernel<<<B, BLOCK_SIZE, 0, stream>>>(Gamma, log_alpha, grad_tau, B, n, m, neg_inv_tau2);

    cudaFree(Gamma);
    cudaFree(Gamma_Y);
    cudaFree(log_X);
    cudaFree(log_Y);
    cudaFree(zero_v);
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

    launch_copy(grad_P, G, total, stream);
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
