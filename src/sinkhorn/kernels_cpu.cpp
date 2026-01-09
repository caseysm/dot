/**
 * @file kernels_cpu.cpp
 * @brief Sinkhorn CPU Kernels
 *
 * Raw C++ implementation of Sinkhorn-Knopp algorithm with backward passes.
 * Converts log-space scores to doubly-stochastic matrix (soft permutation).
 *
 * Input:  log_alpha [B, n, n] - logits (use -D/sigma for distance matrix D)
 * Output: P [B, n, n] - doubly-stochastic soft permutation matrix
 *
 * Two backward pass implementations:
 * 1. Unrolled: Differentiate through T iterations explicitly (exact for finite T)
 * 2. Implicit: Use implicit function theorem at convergence (memory efficient)
 */

#include "kernels_cpu.h"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstring>
#include <vector>

namespace dot {
namespace sinkhorn {

// =============================================================================
// Numerical Stability Helpers
// =============================================================================

inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

// Kahan compensated summation for better precision
struct KahanAccumulator {
    float sum = 0.0f;
    float c = 0.0f;

    void add(float value) {
        float y = value - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    float result() const { return sum; }
    void reset() { sum = 0.0f; c = 0.0f; }
};

// =============================================================================
// Helper: logsumexp for a row/column with corner case handling
// =============================================================================

static inline float logsumexp(const float* x, int n, int stride) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        max_val = std::max(max_val, x[i * stride]);
    }

    if (max_val == -FLT_MAX) {
        return -FLT_MAX;
    }

    KahanAccumulator sum;
    for (int i = 0; i < n; ++i) {
        sum.add(safe_exp(x[i * stride] - max_val));
    }

    if (sum.result() == 0.0f) {
        return -FLT_MAX;
    }

    return max_val + std::log(sum.result());
}

// =============================================================================
// Sinkhorn Forward (CPU) - Basic version without storing intermediates
// =============================================================================

void sinkhorn_forward_cpu(
    const float* log_alpha,
    float* log_P,
    int B, int n,
    float tau,
    int n_iters,
    bool return_log
) {
    if (tau <= 0.0f) return;

    const int n2 = n * n;

    for (int b = 0; b < B; ++b) {
        const float* in_b = log_alpha + b * n2;
        float* out_b = log_P + b * n2;

        // Initialize: scale by temperature
        for (int i = 0; i < n2; ++i) {
            out_b[i] = in_b[i] / tau;
        }

        // Sinkhorn iterations
        for (int iter = 0; iter < n_iters; ++iter) {
            // Row normalization
            for (int i = 0; i < n; ++i) {
                float* row = out_b + i * n;
                float lse = logsumexp(row, n, 1);
                for (int j = 0; j < n; ++j) {
                    row[j] -= lse;
                }
            }

            // Column normalization
            for (int j = 0; j < n; ++j) {
                float lse = logsumexp(out_b + j, n, n);
                for (int i = 0; i < n; ++i) {
                    out_b[i * n + j] -= lse;
                }
            }
        }

        // Convert to probability if needed
        if (!return_log) {
            for (int i = 0; i < n2; ++i) {
                out_b[i] = safe_exp(out_b[i]);
            }
        }
    }
}

// =============================================================================
// Sinkhorn Forward with Intermediates (for Unrolled Backward)
// =============================================================================

void sinkhorn_forward_with_intermediates_cpu(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    int B, int n,
    float tau,
    int n_iters
) {
    if (tau <= 0.0f) return;

    const int n2 = n * n;

    for (int b = 0; b < B; ++b) {
        const float* in_b = log_alpha + b * n2;
        float* P_b = P + b * n2;
        float* log_X_b = log_X + b * (n_iters + 1) * n2;
        float* log_Y_b = log_Y + b * n_iters * n2;

        // Initialize X^(0) = log_alpha / tau
        float* X0 = log_X_b;
        for (int i = 0; i < n2; ++i) {
            X0[i] = in_b[i] / tau;
        }

        // Sinkhorn iterations
        for (int t = 0; t < n_iters; ++t) {
            float* X_t = log_X_b + t * n2;
            float* Y_t = log_Y_b + t * n2;
            float* X_t1 = log_X_b + (t + 1) * n2;

            std::memcpy(Y_t, X_t, n2 * sizeof(float));

            // Row normalization
            for (int i = 0; i < n; ++i) {
                float* row = Y_t + i * n;
                float lse = logsumexp(row, n, 1);
                for (int j = 0; j < n; ++j) {
                    row[j] -= lse;
                }
            }

            std::memcpy(X_t1, Y_t, n2 * sizeof(float));

            // Column normalization
            for (int j = 0; j < n; ++j) {
                float lse = logsumexp(X_t1 + j, n, n);
                for (int i = 0; i < n; ++i) {
                    X_t1[i * n + j] -= lse;
                }
            }
        }

        // P = exp(X^(T))
        float* X_T = log_X_b + n_iters * n2;
        for (int i = 0; i < n2; ++i) {
            P_b[i] = safe_exp(X_T[i]);
        }
    }
}

// =============================================================================
// Sinkhorn Unrolled Backward (CPU)
// =============================================================================

void sinkhorn_backward_unrolled_cpu(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    const float* log_X,
    const float* log_Y,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int n_iters
) {
    const int n2 = n * n;

    std::vector<float> Gamma(n2);
    std::vector<float> Gamma_Y(n2);
    std::vector<float> q(n2);
    std::vector<float> p(n2);

    for (int b = 0; b < B; ++b) {
        const float* log_alpha_b = log_alpha + b * n2;
        const float* P_b = P + b * n2;
        const float* grad_P_b = grad_P + b * n2;
        const float* log_X_b = log_X + b * (n_iters + 1) * n2;
        const float* log_Y_b = log_Y + b * n_iters * n2;
        float* grad_out_b = grad_log_alpha + b * n2;

        // Initialize: Gamma^(T) = grad_P * P
        for (int i = 0; i < n2; ++i) {
            Gamma[i] = grad_P_b[i] * P_b[i];
        }

        // Backprop through iterations
        for (int t = n_iters - 1; t >= 0; --t) {
            const float* X_t1 = log_X_b + (t + 1) * n2;
            const float* Y_t = log_Y_b + t * n2;

            // Column backward
            for (int i = 0; i < n2; ++i) {
                q[i] = safe_exp(X_t1[i]);
            }

            for (int k = 0; k < n; ++k) {
                float s_k = 0.0f;
                for (int i = 0; i < n; ++i) {
                    s_k += Gamma[i * n + k];
                }
                for (int i = 0; i < n; ++i) {
                    Gamma_Y[i * n + k] = Gamma[i * n + k] - q[i * n + k] * s_k;
                }
            }

            // Row backward
            for (int i = 0; i < n2; ++i) {
                p[i] = safe_exp(Y_t[i]);
            }

            for (int i = 0; i < n; ++i) {
                float t_i = 0.0f;
                for (int k = 0; k < n; ++k) {
                    t_i += Gamma_Y[i * n + k];
                }
                for (int k = 0; k < n; ++k) {
                    Gamma[i * n + k] = Gamma_Y[i * n + k] - p[i * n + k] * t_i;
                }
            }
        }

        // Gradient w.r.t. log_alpha
        for (int i = 0; i < n2; ++i) {
            grad_out_b[i] = Gamma[i] / tau;
        }

        // Gradient w.r.t. tau
        float grad_tau_b = 0.0f;
        for (int i = 0; i < n2; ++i) {
            grad_tau_b += Gamma[i] * log_alpha_b[i];
        }
        grad_tau[b] = -grad_tau_b / (tau * tau);
    }
}

// =============================================================================
// Sinkhorn Implicit Backward (CPU)
// =============================================================================

void sinkhorn_backward_implicit_cpu(
    const float* log_alpha,
    const float* P,
    const float* grad_P,
    float* grad_log_alpha,
    float* grad_tau,
    int B, int n,
    float tau,
    int max_iters,
    float tol
) {
    const int n2 = n * n;

    std::vector<float> G(n2);
    std::vector<float> lambda_(n);
    std::vector<float> mu(n);

    for (int b = 0; b < B; ++b) {
        const float* log_alpha_b = log_alpha + b * n2;
        const float* P_b = P + b * n2;
        const float* grad_P_b = grad_P + b * n2;
        float* grad_out_b = grad_log_alpha + b * n2;

        std::memcpy(G.data(), grad_P_b, n2 * sizeof(float));
        std::fill(lambda_.begin(), lambda_.end(), 0.0f);
        std::fill(mu.begin(), mu.end(), 0.0f);

        // Iterate to solve adjoint system
        for (int iter = 0; iter < max_iters; ++iter) {
            float max_change = 0.0f;

            // Update lambda
            for (int i = 0; i < n; ++i) {
                float sum_PG = 0.0f;
                float sum_P_mu = 0.0f;
                for (int k = 0; k < n; ++k) {
                    float p_ik = P_b[i * n + k];
                    sum_PG += p_ik * G[i * n + k];
                    sum_P_mu += p_ik * mu[k];
                }
                float new_lambda = sum_PG - sum_P_mu;
                max_change = std::max(max_change, std::abs(new_lambda - lambda_[i]));
                lambda_[i] = new_lambda;
            }

            // Update mu
            for (int k = 0; k < n; ++k) {
                float sum_PG = 0.0f;
                float sum_P_lambda = 0.0f;
                for (int i = 0; i < n; ++i) {
                    float p_ik = P_b[i * n + k];
                    sum_PG += p_ik * G[i * n + k];
                    sum_P_lambda += p_ik * lambda_[i];
                }
                float new_mu = sum_PG - sum_P_lambda;
                max_change = std::max(max_change, std::abs(new_mu - mu[k]));
                mu[k] = new_mu;
            }

            if (max_change < tol) {
                break;
            }
        }

        // Compute gradient
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                float centered = G[i * n + k] - lambda_[i] - mu[k];
                grad_out_b[i * n + k] = P_b[i * n + k] * centered / tau;
            }
        }

        // Gradient w.r.t. tau
        float grad_tau_b = 0.0f;
        for (int i = 0; i < n2; ++i) {
            grad_tau_b += grad_out_b[i] * log_alpha_b[i];
        }
        grad_tau[b] = -grad_tau_b / tau;
    }
}

} // namespace sinkhorn
} // namespace dot
