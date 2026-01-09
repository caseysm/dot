/**
 * @file kernels.cu
 * @brief Bidirectional Softmax CUDA kernels
 *
 * GPU-accelerated bidirectional softmax.
 * Computes soft matching without monotonic constraint:
 *   out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))
 *
 * Useful for unordered sequence matching where:
 * - No sequential constraint is needed
 * - Geometric mean of row and column softmax provides soft assignment
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include "bidirectional_softmax/kernels.cuh"

namespace dot {
namespace bidirectional_softmax {

// =============================================================================
// Constants
// =============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define EPS 1e-9f

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

// Row-wise softmax: each block handles one row
__global__ void row_softmax_kernel(
    const float* __restrict__ sim_matrix,
    float* __restrict__ row_softmax,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float inv_T
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / max_L1;
    int i = row_idx % max_L1;

    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Skip if row is outside valid range
    if (i >= L1) {
        for (int j = threadIdx.x; j < max_L2; j += blockDim.x) {
            row_softmax[b * max_L1 * max_L2 + i * max_L2 + j] = 0.0f;
        }
        return;
    }

    const float* row_in = sim_matrix + b * max_L1 * max_L2 + i * max_L2;
    float* row_out = row_softmax + b * max_L1 * max_L2 + i * max_L2;

    // Find max in valid columns
    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < L2; j += blockDim.x) {
        max_val = fmaxf(max_val, row_in[j] * inv_T);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    // Compute sum(exp(x - max))
    float sum = 0.0f;
    for (int j = threadIdx.x; j < L2; j += blockDim.x) {
        sum += safe_exp(row_in[j] * inv_T - max_val);
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    float total_sum = s_sum;

    // Compute softmax and write output
    float inv_sum = (total_sum > 0.0f) ? 1.0f / total_sum : 0.0f;
    for (int j = threadIdx.x; j < max_L2; j += blockDim.x) {
        if (j < L2) {
            row_out[j] = expf(row_in[j] * inv_T - max_val) * inv_sum;
        } else {
            row_out[j] = 0.0f;
        }
    }
}

// Column-wise softmax: each block handles one column
__global__ void col_softmax_kernel(
    const float* __restrict__ sim_matrix,
    float* __restrict__ col_softmax,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float inv_T
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / max_L2;
    int j = col_idx % max_L2;

    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Skip if column is outside valid range
    if (j >= L2) {
        for (int i = threadIdx.x; i < max_L1; i += blockDim.x) {
            col_softmax[b * max_L1 * max_L2 + i * max_L2 + j] = 0.0f;
        }
        return;
    }

    const float* base_in = sim_matrix + b * max_L1 * max_L2 + j;
    float* base_out = col_softmax + b * max_L1 * max_L2 + j;

    // Find max in valid rows
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < L1; i += blockDim.x) {
        max_val = fmaxf(max_val, base_in[i * max_L2] * inv_T);
    }
    __shared__ float s_max;
    float block_max = block_reduce_max(max_val, shared);
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    max_val = s_max;

    // Compute sum(exp(x - max))
    float sum = 0.0f;
    for (int i = threadIdx.x; i < L1; i += blockDim.x) {
        sum += expf(base_in[i * max_L2] * inv_T - max_val);
    }
    __shared__ float s_sum;
    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) s_sum = block_sum;
    __syncthreads();
    float total_sum = s_sum;

    // Compute softmax and write output
    float inv_sum = (total_sum > 0.0f) ? 1.0f / total_sum : 0.0f;
    for (int i = threadIdx.x; i < max_L1; i += blockDim.x) {
        if (i < L1) {
            base_out[i * max_L2] = expf(base_in[i * max_L2] * inv_T - max_val) * inv_sum;
        } else {
            base_out[i * max_L2] = 0.0f;
        }
    }
}

// Combine: out = sqrt(eps + row * col)
__global__ void combine_softmax_kernel(
    const float* __restrict__ row_softmax,
    const float* __restrict__ col_softmax,
    float* __restrict__ output,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * max_L1 * max_L2;

    if (idx >= total) return;

    int b = idx / (max_L1 * max_L2);
    int rem = idx % (max_L1 * max_L2);
    int i = rem / max_L2;
    int j = rem % max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i < L1 && j < L2) {
        float product = row_softmax[idx] * col_softmax[idx];
        output[idx] = sqrtf(EPS + product);
    } else {
        output[idx] = 0.0f;
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// d_product = grad_output * 0.5 / output
__global__ void backward_sqrt_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ d_product,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * max_L1 * max_L2;

    if (idx >= total) return;

    int b = idx / (max_L1 * max_L2);
    int rem = idx % (max_L1 * max_L2);
    int i = rem / max_L2;
    int j = rem % max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i < L1 && j < L2 && output[idx] > EPS) {
        d_product[idx] = grad_output[idx] * 0.5f / output[idx];
    } else {
        d_product[idx] = 0.0f;
    }
}

// Gradients for row softmax
__global__ void backward_row_softmax_kernel(
    const float* __restrict__ d_row_softmax,
    const float* __restrict__ row_softmax,
    float* __restrict__ grad_sim,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float inv_T
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    int b = row_idx / max_L1;
    int i = row_idx % max_L1;

    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i >= L1) return;

    const float* d_row = d_row_softmax + b * max_L1 * max_L2 + i * max_L2;
    const float* row = row_softmax + b * max_L1 * max_L2 + i * max_L2;
    float* g_row = grad_sim + b * max_L1 * max_L2 + i * max_L2;

    // Compute dot product: sum(d_row_sm * row_sm)
    float dot = 0.0f;
    for (int j = threadIdx.x; j < L2; j += blockDim.x) {
        dot += d_row[j] * row[j];
    }
    __shared__ float s_dot;
    float block_dot = block_reduce_sum(dot, shared);
    if (threadIdx.x == 0) s_dot = block_dot;
    __syncthreads();
    dot = s_dot;

    // Compute gradient contribution
    for (int j = threadIdx.x; j < L2; j += blockDim.x) {
        atomicAdd(&g_row[j], (d_row[j] - dot) * row[j] * inv_T);
    }
}

// Gradients for column softmax
__global__ void backward_col_softmax_kernel(
    const float* __restrict__ d_col_softmax,
    const float* __restrict__ col_softmax,
    float* __restrict__ grad_sim,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float inv_T
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int col_idx = blockIdx.x;
    int b = col_idx / max_L2;
    int j = col_idx % max_L2;

    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (j >= L2) return;

    const float* d_col_base = d_col_softmax + b * max_L1 * max_L2 + j;
    const float* col_base = col_softmax + b * max_L1 * max_L2 + j;
    float* g_col_base = grad_sim + b * max_L1 * max_L2 + j;

    // Compute dot product: sum(d_col_sm * col_sm)
    float dot = 0.0f;
    for (int i = threadIdx.x; i < L1; i += blockDim.x) {
        dot += d_col_base[i * max_L2] * col_base[i * max_L2];
    }
    __shared__ float s_dot;
    float block_dot = block_reduce_sum(dot, shared);
    if (threadIdx.x == 0) s_dot = block_dot;
    __syncthreads();
    dot = s_dot;

    // Compute gradient contribution
    for (int i = threadIdx.x; i < L1; i += blockDim.x) {
        atomicAdd(&g_col_base[i * max_L2], (d_col_base[i * max_L2] - dot) * col_base[i * max_L2] * inv_T);
    }
}

// d_row_softmax = d_product * col_softmax, d_col_softmax = d_product * row_softmax
__global__ void backward_product_kernel(
    const float* __restrict__ d_product,
    const float* __restrict__ row_softmax,
    const float* __restrict__ col_softmax,
    float* __restrict__ d_row_softmax,
    float* __restrict__ d_col_softmax,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    d_row_softmax[idx] = d_product[idx] * col_softmax[idx];
    d_col_softmax[idx] = d_product[idx] * row_softmax[idx];
}

// Temperature gradient
__global__ void backward_temperature_kernel(
    const float* __restrict__ grad_sim,
    const float* __restrict__ sim_matrix,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float inv_T
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int b = blockIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* g_sim = grad_sim + b * max_L1 * max_L2;
    const float* sim = sim_matrix + b * max_L1 * max_L2;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < L1 * max_L2; idx += blockDim.x) {
        int i = idx / max_L2;
        int j = idx % max_L2;
        if (i < L1 && j < L2) {
            sum -= g_sim[i * max_L2 + j] * sim[i * max_L2 + j];
        }
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) {
        grad_T[b] = block_sum * inv_T;
    }
}

// =============================================================================
// Entry Points
// =============================================================================

void bidirectional_softmax_forward_cuda(
    const float* sim_matrix,
    float* output,
    float* row_softmax_buf,
    float* col_softmax_buf,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    float inv_T = 1.0f / T;

    // Row softmax: one block per row
    int num_rows = B * max_L1;
    row_softmax_kernel<<<num_rows, BLOCK_SIZE>>>(
        sim_matrix, row_softmax_buf, lengths, B, max_L1, max_L2, inv_T
    );

    // Column softmax: one block per column
    int num_cols = B * max_L2;
    col_softmax_kernel<<<num_cols, BLOCK_SIZE>>>(
        sim_matrix, col_softmax_buf, lengths, B, max_L1, max_L2, inv_T
    );

    // Combine: out = sqrt(eps + row * col)
    int total = B * max_L1 * max_L2;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    combine_softmax_kernel<<<num_blocks, BLOCK_SIZE>>>(
        row_softmax_buf, col_softmax_buf, output, lengths, B, max_L1, max_L2
    );

    cudaDeviceSynchronize();
}

void bidirectional_softmax_backward_cuda(
    const float* sim_matrix,
    const float* output,
    const float* grad_output,
    const float* row_softmax_buf,
    const float* col_softmax_buf,
    float* grad_sim,
    float* grad_T,
    float* d_product,
    float* d_row_softmax,
    float* d_col_softmax,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    float inv_T = 1.0f / T;
    int total = B * max_L1 * max_L2;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Initialize grad_sim to zero
    cudaMemset(grad_sim, 0, total * sizeof(float));

    // Step 1: d_product = grad_output * 0.5 / output
    backward_sqrt_kernel<<<num_blocks, BLOCK_SIZE>>>(
        grad_output, output, d_product, lengths, B, max_L1, max_L2
    );

    // Step 2: d_row_softmax = d_product * col_softmax, d_col_softmax = d_product * row_softmax
    backward_product_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_product, row_softmax_buf, col_softmax_buf,
        d_row_softmax, d_col_softmax, total
    );

    // Step 3: Backward through row softmax
    int num_rows = B * max_L1;
    backward_row_softmax_kernel<<<num_rows, BLOCK_SIZE>>>(
        d_row_softmax, row_softmax_buf, grad_sim, lengths, B, max_L1, max_L2, inv_T
    );

    // Step 4: Backward through column softmax
    int num_cols = B * max_L2;
    backward_col_softmax_kernel<<<num_cols, BLOCK_SIZE>>>(
        d_col_softmax, col_softmax_buf, grad_sim, lengths, B, max_L1, max_L2, inv_T
    );

    // Step 5: Temperature gradient
    backward_temperature_kernel<<<B, BLOCK_SIZE>>>(
        grad_sim, sim_matrix, grad_T, lengths, B, max_L1, max_L2, inv_T
    );

    cudaDeviceSynchronize();
}

} // namespace bidirectional_softmax
} // namespace dot
