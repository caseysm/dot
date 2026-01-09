/**
 * @file kernels_cpu.cpp
 * @brief Bidirectional Softmax CPU kernels
 *
 * CPU implementation of bidirectional softmax.
 * Computes soft matching without monotonic constraint:
 *   out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))
 */

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

#include "bidirectional_softmax/kernels_cpu.h"

namespace dot {
namespace bidirectional_softmax {

// =============================================================================
// Constants
// =============================================================================

constexpr float NINF = -1e30f;
constexpr float EPS = 1e-9f;

// =============================================================================
// Helpers
// =============================================================================

inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

// =============================================================================
// Forward Pass
// =============================================================================

void bidirectional_softmax_forward_cpu(
    const float* sim_matrix,
    float* output,
    float* row_softmax_buf,
    float* col_softmax_buf,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t matrix_stride = static_cast<size_t>(max_L1) * max_L2;
    float inv_T = 1.0f / T;

    for (int b = 0; b < B; b++) {
        const float* sim = sim_matrix + b * matrix_stride;
        float* out = output + b * matrix_stride;
        float* row_sm = row_softmax_buf + b * matrix_stride;
        float* col_sm = col_softmax_buf + b * matrix_stride;

        int L1 = lengths[b * 2];
        int L2 = lengths[b * 2 + 1];

        // Initialize buffers to zero
        std::memset(row_sm, 0, matrix_stride * sizeof(float));
        std::memset(col_sm, 0, matrix_stride * sizeof(float));

        // Row softmax (for each row, softmax over columns)
        for (int i = 0; i < L1; i++) {
            // Find max
            float max_val = NINF;
            for (int j = 0; j < L2; j++) {
                float val = sim[i * max_L2 + j] * inv_T;
                if (val > max_val) max_val = val;
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < L2; j++) {
                float val = sim[i * max_L2 + j] * inv_T;
                float exp_val = safe_exp(val - max_val);
                row_sm[i * max_L2 + j] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int j = 0; j < L2; j++) {
                    row_sm[i * max_L2 + j] *= inv_sum;
                }
            }
        }

        // Column softmax (for each column, softmax over rows)
        for (int j = 0; j < L2; j++) {
            // Find max
            float max_val = NINF;
            for (int i = 0; i < L1; i++) {
                float val = sim[i * max_L2 + j] * inv_T;
                if (val > max_val) max_val = val;
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int i = 0; i < L1; i++) {
                float val = sim[i * max_L2 + j] * inv_T;
                float exp_val = safe_exp(val - max_val);
                col_sm[i * max_L2 + j] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int i = 0; i < L1; i++) {
                    col_sm[i * max_L2 + j] *= inv_sum;
                }
            }
        }

        // Compute output: sqrt(eps + row_softmax * col_softmax)
        for (int i = 0; i < max_L1; i++) {
            for (int j = 0; j < max_L2; j++) {
                int idx = i * max_L2 + j;
                if (i < L1 && j < L2) {
                    float product = row_sm[idx] * col_sm[idx];
                    out[idx] = std::sqrt(EPS + product);
                } else {
                    out[idx] = 0.0f;
                }
            }
        }
    }
}

// =============================================================================
// Backward Pass
// =============================================================================

void bidirectional_softmax_backward_cpu(
    const float* sim_matrix,
    const float* output,
    const float* grad_output,
    const float* row_softmax_buf,
    const float* col_softmax_buf,
    float* grad_sim,
    float* grad_T,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t matrix_stride = static_cast<size_t>(max_L1) * max_L2;
    float inv_T = 1.0f / T;

    for (int b = 0; b < B; b++) {
        const float* sim = sim_matrix + b * matrix_stride;
        const float* out = output + b * matrix_stride;
        const float* g_out = grad_output + b * matrix_stride;
        const float* row_sm = row_softmax_buf + b * matrix_stride;
        const float* col_sm = col_softmax_buf + b * matrix_stride;
        float* g_sim = grad_sim + b * matrix_stride;

        int L1 = lengths[b * 2];
        int L2 = lengths[b * 2 + 1];

        // Initialize gradient to zero
        std::memset(g_sim, 0, matrix_stride * sizeof(float));

        // Allocate workspace
        float* d_product = new float[matrix_stride]();
        float* d_row_sm = new float[matrix_stride]();
        float* d_col_sm = new float[matrix_stride]();

        // Step 1: d_product = grad_output * 0.5 / output
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                int idx = i * max_L2 + j;
                if (out[idx] > EPS) {
                    d_product[idx] = g_out[idx] * 0.5f / out[idx];
                }
            }
        }

        // Step 2: d_row_sm = d_product * col_sm, d_col_sm = d_product * row_sm
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                int idx = i * max_L2 + j;
                d_row_sm[idx] = d_product[idx] * col_sm[idx];
                d_col_sm[idx] = d_product[idx] * row_sm[idx];
            }
        }

        // Step 3: Backward through row softmax
        for (int i = 0; i < L1; i++) {
            float dot = 0.0f;
            for (int j = 0; j < L2; j++) {
                int idx = i * max_L2 + j;
                dot += d_row_sm[idx] * row_sm[idx];
            }
            for (int j = 0; j < L2; j++) {
                int idx = i * max_L2 + j;
                g_sim[idx] += (d_row_sm[idx] - dot) * row_sm[idx] * inv_T;
            }
        }

        // Step 4: Backward through column softmax
        for (int j = 0; j < L2; j++) {
            float dot = 0.0f;
            for (int i = 0; i < L1; i++) {
                int idx = i * max_L2 + j;
                dot += d_col_sm[idx] * col_sm[idx];
            }
            for (int i = 0; i < L1; i++) {
                int idx = i * max_L2 + j;
                g_sim[idx] += (d_col_sm[idx] - dot) * col_sm[idx] * inv_T;
            }
        }

        // Step 5: Temperature gradient
        float g_T_batch = 0.0f;
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                int idx = i * max_L2 + j;
                g_T_batch -= g_sim[idx] * sim[idx] * inv_T;
            }
        }
        grad_T[b] = g_T_batch;

        delete[] d_product;
        delete[] d_row_sm;
        delete[] d_col_sm;
    }
}

} // namespace bidirectional_softmax
} // namespace dot
