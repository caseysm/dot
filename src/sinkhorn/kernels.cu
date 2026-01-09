/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for Sinkhorn algorithm
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cmath>
#include <vector>

namespace dot {
namespace sinkhorn {

// Kernel to compute K = exp(-C / reg)
__global__ void compute_kernel_matrix(
    const float* __restrict__ cost,
    float* __restrict__ K,
    int M, int N,
    float inv_reg
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        K[idx] = expf(-cost[idx] * inv_reg);
    }
}

// Kernel to compute Kv = K @ v (matrix-vector product along rows)
__global__ void compute_Kv(
    const float* __restrict__ K,
    const float* __restrict__ v,
    float* __restrict__ Kv,
    int M, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += K[i * N + j] * v[j];
        }
        Kv[i] = sum;
    }
}

// Kernel to compute Ktu = K^T @ u (matrix-vector product along columns)
__global__ void compute_Ktu(
    const float* __restrict__ K,
    const float* __restrict__ u,
    float* __restrict__ Ktu,
    int M, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += K[i * N + j] * u[i];
        }
        Ktu[j] = sum;
    }
}

// Kernel to update u = a / Kv
__global__ void update_u(
    const float* __restrict__ a,
    const float* __restrict__ Kv,
    float* __restrict__ u,
    int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        u[i] = a[i] / (Kv[i] + 1e-10f);
    }
}

// Kernel to update v = b / Ktu
__global__ void update_v(
    const float* __restrict__ b,
    const float* __restrict__ Ktu,
    float* __restrict__ v,
    int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N) {
        v[j] = b[j] / (Ktu[j] + 1e-10f);
    }
}

// Kernel to compute transport plan P = diag(u) @ K @ diag(v)
__global__ void compute_transport(
    const float* __restrict__ u,
    const float* __restrict__ K,
    const float* __restrict__ v,
    float* __restrict__ P,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int i = idx / N;
        int j = idx % N;
        P[idx] = u[i] * K[idx] * v[j];
    }
}

// Kernel for backward pass gradient
__global__ void compute_grad_cost(
    const float* __restrict__ grad_output,
    const float* __restrict__ transport,
    float* __restrict__ grad_cost,
    int total,
    float inv_reg
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        grad_cost[idx] = -transport[idx] * grad_output[idx] * inv_reg;
    }
}

void sinkhorn_forward_cuda(
    const float* cost,
    float* transport,
    const float* a,
    const float* b,
    int B, int M, int N,
    float reg,
    int max_iter,
    float tol,
    cudaStream_t stream
) {
    float inv_reg = 1.0f / reg;
    int block_size = 256;

    // Allocate temporary buffers (simplified: per-batch processing)
    float* K;
    float* u;
    float* v;
    float* Kv;
    float* Ktu;
    float* a_uniform;
    float* b_uniform;

    cudaMalloc(&K, M * N * sizeof(float));
    cudaMalloc(&u, M * sizeof(float));
    cudaMalloc(&v, N * sizeof(float));
    cudaMalloc(&Kv, M * sizeof(float));
    cudaMalloc(&Ktu, N * sizeof(float));

    // Create uniform distributions if needed
    bool need_uniform_a = (a == nullptr);
    bool need_uniform_b = (b == nullptr);

    if (need_uniform_a) {
        cudaMalloc(&a_uniform, M * sizeof(float));
        float uniform_val = 1.0f / M;
        // Initialize with uniform value
        std::vector<float> a_host(M, uniform_val);
        cudaMemcpy(a_uniform, a_host.data(), M * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (need_uniform_b) {
        cudaMalloc(&b_uniform, N * sizeof(float));
        float uniform_val = 1.0f / N;
        std::vector<float> b_host(N, uniform_val);
        cudaMemcpy(b_uniform, b_host.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (int batch = 0; batch < B; batch++) {
        const float* C = cost + batch * M * N;
        float* P = transport + batch * M * N;
        const float* a_ptr = need_uniform_a ? a_uniform : (a + batch * M);
        const float* b_ptr = need_uniform_b ? b_uniform : (b + batch * N);

        // Compute K = exp(-C / reg)
        int num_blocks = common::get_num_blocks(M * N, block_size);
        compute_kernel_matrix<<<num_blocks, block_size, 0, stream>>>(C, K, M, N, inv_reg);

        // Initialize u and v to ones
        std::vector<float> ones_m(M, 1.0f);
        std::vector<float> ones_n(N, 1.0f);
        cudaMemcpy(u, ones_m.data(), M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v, ones_n.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Sinkhorn iterations
        for (int iter = 0; iter < max_iter; iter++) {
            // Kv = K @ v
            int blocks_m = common::get_num_blocks(M, block_size);
            compute_Kv<<<blocks_m, block_size, 0, stream>>>(K, v, Kv, M, N);

            // u = a / Kv
            update_u<<<blocks_m, block_size, 0, stream>>>(a_ptr, Kv, u, M);

            // Ktu = K^T @ u
            int blocks_n = common::get_num_blocks(N, block_size);
            compute_Ktu<<<blocks_n, block_size, 0, stream>>>(K, u, Ktu, M, N);

            // v = b / Ktu
            update_v<<<blocks_n, block_size, 0, stream>>>(b_ptr, Ktu, v, N);
        }

        // Compute transport plan
        compute_transport<<<num_blocks, block_size, 0, stream>>>(u, K, v, P, M, N);
    }

    // Free temporary buffers
    cudaFree(K);
    cudaFree(u);
    cudaFree(v);
    cudaFree(Kv);
    cudaFree(Ktu);

    if (need_uniform_a) cudaFree(a_uniform);
    if (need_uniform_b) cudaFree(b_uniform);
}

void sinkhorn_backward_cuda(
    const float* grad_output,
    const float* cost,
    const float* transport,
    float* grad_cost,
    int B, int M, int N,
    float reg,
    cudaStream_t stream
) {
    float inv_reg = 1.0f / reg;
    int total = B * M * N;
    int block_size = 256;
    int num_blocks = common::get_num_blocks(total, block_size);

    compute_grad_cost<<<num_blocks, block_size, 0, stream>>>(
        grad_output, transport, grad_cost, total, inv_reg
    );
}

} // namespace sinkhorn
} // namespace dot
