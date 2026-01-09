/**
 * @file kernels_cpu.cpp
 * @brief CPU kernel implementations for Sinkhorn algorithm
 */

#include "kernels_cpu.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace dot {
namespace sinkhorn {

void sinkhorn_forward_cpu(
    const float* cost,
    float* transport,
    const float* a,
    const float* b,
    int B, int M, int N,
    float reg,
    int max_iter,
    float tol
) {
    // Sinkhorn-Knopp algorithm
    // K = exp(-C / reg)
    // u = a / (K @ v)
    // v = b / (K^T @ u)
    // P = diag(u) @ K @ diag(v)

    float inv_reg = 1.0f / reg;

    for (int batch = 0; batch < B; batch++) {
        const float* C = cost + batch * M * N;
        float* P = transport + batch * M * N;

        // Initialize K = exp(-C / reg)
        std::vector<float> K(M * N);
        for (int i = 0; i < M * N; i++) {
            K[i] = std::exp(-C[i] * inv_reg);
        }

        // Initialize u and v
        std::vector<float> u(M, 1.0f);
        std::vector<float> v(N, 1.0f);

        // Source and target distributions
        std::vector<float> a_vec(M);
        std::vector<float> b_vec(N);

        if (a != nullptr) {
            const float* a_batch = a + batch * M;
            for (int i = 0; i < M; i++) a_vec[i] = a_batch[i];
        } else {
            float uniform_a = 1.0f / M;
            for (int i = 0; i < M; i++) a_vec[i] = uniform_a;
        }

        if (b != nullptr) {
            const float* b_batch = b + batch * N;
            for (int j = 0; j < N; j++) b_vec[j] = b_batch[j];
        } else {
            float uniform_b = 1.0f / N;
            for (int j = 0; j < N; j++) b_vec[j] = uniform_b;
        }

        // Sinkhorn iterations
        for (int iter = 0; iter < max_iter; iter++) {
            // u = a / (K @ v)
            std::vector<float> Kv(M, 0.0f);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    Kv[i] += K[i * N + j] * v[j];
                }
                u[i] = a_vec[i] / (Kv[i] + 1e-10f);
            }

            // v = b / (K^T @ u)
            std::vector<float> Ktu(N, 0.0f);
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < M; i++) {
                    Ktu[j] += K[i * N + j] * u[i];
                }
                v[j] = b_vec[j] / (Ktu[j] + 1e-10f);
            }

            // Check convergence (optional)
            // Could check ||P @ 1 - a|| + ||P^T @ 1 - b|| < tol
        }

        // Compute transport plan: P = diag(u) @ K @ diag(v)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                P[i * N + j] = u[i] * K[i * N + j] * v[j];
            }
        }
    }
}

void sinkhorn_backward_cpu(
    const float* grad_output,
    const float* cost,
    const float* transport,
    float* grad_cost,
    int B, int M, int N,
    float reg
) {
    // Gradient of transport plan w.r.t. cost matrix
    // Using the implicit function theorem on the optimality conditions
    //
    // Simplified gradient: grad_cost = -transport * grad_output / reg
    // (This is an approximation; full gradient requires solving linear system)

    float inv_reg = 1.0f / reg;

    for (int batch = 0; batch < B; batch++) {
        const float* grad_out = grad_output + batch * M * N;
        const float* P = transport + batch * M * N;
        float* grad_C = grad_cost + batch * M * N;

        for (int i = 0; i < M * N; i++) {
            grad_C[i] = -P[i] * grad_out[i] * inv_reg;
        }
    }
}

} // namespace sinkhorn
} // namespace dot
