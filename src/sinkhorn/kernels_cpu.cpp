/**
 * @file kernels_cpu.cpp
 * @brief Sinkhorn CPU Kernels
 *
 * Raw C++ implementation of Sinkhorn-Knopp algorithm with backward passes.
 */

#include "kernels_cpu.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace dot {
namespace sinkhorn {

inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

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
};

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

void sinkhorn_forward_cpu(
    const float* log_alpha,
    float* log_P,
    const float* log_a,
    const float* log_b,
    int B, int n, int m,
    float tau,
    int n_iters,
    bool return_log
) {
    if (tau <= 0.0f) return;

    const int nm = n * m;

    for (int b = 0; b < B; ++b) {
        const float* in_b = log_alpha + b * nm;
        float* out_b = log_P + b * nm;

        for (int idx = 0; idx < nm; ++idx) {
            out_b[idx] = in_b[idx] / tau;
        }

        for (int iter = 0; iter < n_iters; ++iter) {
            for (int i = 0; i < n; ++i) {
                float* row = out_b + i * m;
                float lse = logsumexp(row, m, 1);
                float target = log_a != nullptr ? log_a[b * n + i] : -std::log(static_cast<float>(n));
                for (int j = 0; j < m; ++j) {
                    row[j] -= (lse - target);
                }
            }

            for (int j = 0; j < m; ++j) {
                float lse = logsumexp(out_b + j, n, m);
                float target = log_b != nullptr ? log_b[b * m + j] : -std::log(static_cast<float>(m));
                for (int i = 0; i < n; ++i) {
                    out_b[i * m + j] -= (lse - target);
                }
            }
        }

        if (!return_log) {
            for (int idx = 0; idx < nm; ++idx) {
                out_b[idx] = safe_exp(out_b[idx]);
            }
        }
    }
}

void sinkhorn_forward_with_intermediates_cpu(
    const float* log_alpha,
    float* P,
    float* log_X,
    float* log_Y,
    const float* log_a,
    const float* log_b,
    int B, int n, int m,
    float tau,
    int n_iters
) {
    if (tau <= 0.0f) return;

    const int nm = n * m;

    for (int b = 0; b < B; ++b) {
        const float* in_b = log_alpha + b * nm;
        float* P_b = P + b * nm;
        float* log_X_b = log_X + b * (n_iters + 1) * nm;
        float* log_Y_b = log_Y + b * n_iters * nm;

        float* X0 = log_X_b;
        for (int idx = 0; idx < nm; ++idx) {
            X0[idx] = in_b[idx] / tau;
        }

        for (int t = 0; t < n_iters; ++t) {
            float* X_t = log_X_b + t * nm;
            float* Y_t = log_Y_b + t * nm;
            float* X_t1 = log_X_b + (t + 1) * nm;

            std::memcpy(Y_t, X_t, nm * sizeof(float));

            for (int i = 0; i < n; ++i) {
                float* row = Y_t + i * m;
                float lse = logsumexp(row, m, 1);
                float target = log_a != nullptr ? log_a[b * n + i] : -std::log(static_cast<float>(n));
                for (int j = 0; j < m; ++j) {
                    row[j] -= (lse - target);
                }
            }

            std::memcpy(X_t1, Y_t, nm * sizeof(float));

            for (int j = 0; j < m; ++j) {
                float lse = logsumexp(X_t1 + j, n, m);
                float target = log_b != nullptr ? log_b[b * m + j] : -std::log(static_cast<float>(m));
                for (int i = 0; i < n; ++i) {
                    X_t1[i * m + j] -= (lse - target);
                }
            }
        }

        float* X_T = log_X_b + n_iters * nm;
        for (int idx = 0; idx < nm; ++idx) {
            P_b[idx] = safe_exp(X_T[idx]);
        }
    }
}

void sinkhorn_backward_unrolled_cpu(
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
    int n_iters
) {
    const int nm = n * m;

    std::vector<float> Gamma(nm);
    std::vector<float> Gamma_Y(nm);
    std::vector<float> q(nm);
    std::vector<float> p(nm);

    for (int b = 0; b < B; ++b) {
        const float* log_alpha_b = log_alpha + b * nm;
        const float* P_b = P + b * nm;
        const float* grad_P_b = grad_P + b * nm;
        const float* log_X_b = log_X + b * (n_iters + 1) * nm;
        const float* log_Y_b = log_Y + b * n_iters * nm;
        float* grad_out_b = grad_log_alpha + b * nm;

        for (int idx = 0; idx < nm; ++idx) {
            Gamma[idx] = grad_P_b[idx] * P_b[idx];
        }

        for (int t = n_iters - 1; t >= 0; --t) {
            const float* X_t1 = log_X_b + (t + 1) * nm;
            const float* Y_t = log_Y_b + t * nm;

            for (int idx = 0; idx < nm; ++idx) {
                q[idx] = safe_exp(X_t1[idx]);
            }

            for (int k = 0; k < m; ++k) {
                float s_k = 0.0f;
                for (int i = 0; i < n; ++i) {
                    s_k += Gamma[i * m + k];
                }
                for (int i = 0; i < n; ++i) {
                    Gamma_Y[i * m + k] = Gamma[i * m + k] - q[i * m + k] * s_k;
                }
            }

            for (int idx = 0; idx < nm; ++idx) {
                p[idx] = safe_exp(Y_t[idx]);
            }

            for (int i = 0; i < n; ++i) {
                float t_i = 0.0f;
                for (int k = 0; k < m; ++k) {
                    t_i += Gamma_Y[i * m + k];
                }
                for (int k = 0; k < m; ++k) {
                    Gamma[i * m + k] = Gamma_Y[i * m + k] - p[i * m + k] * t_i;
                }
            }
        }

        for (int idx = 0; idx < nm; ++idx) {
            grad_out_b[idx] = Gamma[idx] / tau;
        }

        float grad_tau_b = 0.0f;
        for (int idx = 0; idx < nm; ++idx) {
            grad_tau_b += Gamma[idx] * log_alpha_b[idx];
        }
        grad_tau[b] = -grad_tau_b / (tau * tau);
    }
}

void sinkhorn_backward_implicit_cpu(
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
    float tol
) {
    const int nm = n * m;

    std::vector<float> G(nm);
    std::vector<float> lambda_(n);
    std::vector<float> mu(m);

    for (int b = 0; b < B; ++b) {
        const float* log_alpha_b = log_alpha + b * nm;
        const float* P_b = P + b * nm;
        const float* grad_P_b = grad_P + b * nm;
        float* grad_out_b = grad_log_alpha + b * nm;

        std::memcpy(G.data(), grad_P_b, nm * sizeof(float));
        std::fill(lambda_.begin(), lambda_.end(), 0.0f);
        std::fill(mu.begin(), mu.end(), 0.0f);

        for (int iter = 0; iter < max_iters; ++iter) {
            float max_change = 0.0f;

            for (int i = 0; i < n; ++i) {
                float sum_PG = 0.0f;
                float sum_P_mu = 0.0f;
                float sum_P = 0.0f;
                for (int k = 0; k < m; ++k) {
                    float p_ik = P_b[i * m + k];
                    sum_PG += p_ik * G[i * m + k];
                    sum_P_mu += p_ik * mu[k];
                    sum_P += p_ik;
                }
                float new_lambda = (sum_PG - sum_P_mu) / std::max(sum_P, 1.0e-12f);
                max_change = std::max(max_change, std::abs(new_lambda - lambda_[i]));
                lambda_[i] = new_lambda;
            }

            for (int k = 0; k < m; ++k) {
                float sum_PG = 0.0f;
                float sum_P_lambda = 0.0f;
                float sum_P = 0.0f;
                for (int i = 0; i < n; ++i) {
                    float p_ik = P_b[i * m + k];
                    sum_PG += p_ik * G[i * m + k];
                    sum_P_lambda += p_ik * lambda_[i];
                    sum_P += p_ik;
                }
                float new_mu = (sum_PG - sum_P_lambda) / std::max(sum_P, 1.0e-12f);
                max_change = std::max(max_change, std::abs(new_mu - mu[k]));
                mu[k] = new_mu;
            }

            if (max_change < tol) {
                break;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < m; ++k) {
                float centered = G[i * m + k] - lambda_[i] - mu[k];
                grad_out_b[i * m + k] = P_b[i * m + k] * centered / tau;
            }
        }

        float grad_tau_b = 0.0f;
        for (int idx = 0; idx < nm; ++idx) {
            grad_tau_b += grad_out_b[idx] * log_alpha_b[idx];
        }
        grad_tau[b] = -grad_tau_b / tau;
    }
}

} // namespace sinkhorn
} // namespace dot
