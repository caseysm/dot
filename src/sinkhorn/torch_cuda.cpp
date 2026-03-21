/**
 * @file torch_cuda.cpp
 * @brief Sinkhorn CUDA Extension with PyTorch Autograd
 *
 * CUDA implementations registered via TORCH_LIBRARY_IMPL for automatic dispatch.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/linalg_solve.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "common/torch_utils.h"
#include "sinkhorn/kernels.cuh"

using namespace dot::common;

namespace {

constexpr int kMethodVanilla = 0;
constexpr int kMethodMomentum = 1;
constexpr int kMethodAnderson = 2;
constexpr int kMethodAdam = 3;

constexpr int kScheduleNone = 0;
constexpr int kScheduleLinear = 1;
constexpr int kScheduleExponential = 2;
constexpr int kScheduleCosine = 3;
constexpr float kPi = 3.14159265358979323846f;

torch::Tensor default_log_marginal_cuda(
    c10::optional<torch::Tensor> maybe_log_marginal,
    int batch,
    int size,
    const torch::TensorOptions& options
) {
    if (maybe_log_marginal.has_value()) {
        return maybe_log_marginal->to(torch::kFloat32).contiguous();
    }
    return torch::full({batch, size}, -std::log(static_cast<float>(size)), options);
}

float max_abs_diff_cuda(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    torch::Tensor& delta_tensor,
    cudaStream_t stream
) {
    dot::sinkhorn::sinkhorn_max_abs_diff_cuda(
        lhs.data_ptr<float>(),
        rhs.data_ptr<float>(),
        delta_tensor.data_ptr<float>(),
        static_cast<int>(lhs.numel()),
        stream
    );
    return delta_tensor.item<float>();
}

float scheduled_tau_cuda(
    int64_t step,
    int64_t total_steps,
    float tau_target,
    float reg_start,
    int64_t schedule_code
) {
    if (schedule_code == kScheduleNone || total_steps <= 1) {
        return tau_target;
    }

    float start = reg_start > 0.0f ? reg_start : 10.0f * tau_target;
    float frac = static_cast<float>(step) / static_cast<float>(total_steps - 1);
    if (schedule_code == kScheduleLinear) {
        return start * (1.0f - frac) + tau_target * frac;
    }
    if (schedule_code == kScheduleCosine) {
        return tau_target + 0.5f * (start - tau_target) * (1.0f + std::cos(kPi * frac));
    }
    if (schedule_code == kScheduleExponential) {
        return tau_target * std::pow(start / tau_target, 1.0f - frac);
    }
    return tau_target;
}

bool candidate_improves_cuda(
    const torch::Tensor& candidate,
    const torch::Tensor& target,
    const torch::Tensor& current,
    torch::Tensor& delta_tensor,
    cudaStream_t stream
) {
    float candidate_gap = max_abs_diff_cuda(candidate, target, delta_tensor, stream);
    float current_gap = max_abs_diff_cuda(current, target, delta_tensor, stream);
    return candidate_gap < current_gap;
}

torch::Tensor dual_fixed_point_map_cuda(
    const torch::Tensor& state,
    const torch::Tensor& log_K,
    const torch::Tensor& log_a,
    const torch::Tensor& log_b,
    torch::Tensor& log_u_next,
    torch::Tensor& log_v_next,
    torch::Tensor& row_partial_max,
    torch::Tensor& row_partial_sum,
    torch::Tensor& col_partial_max,
    torch::Tensor& col_partial_sum,
    int64_t n,
    int64_t m,
    int row_chunks,
    int col_chunks,
    cudaStream_t stream
) {
    int B = log_K.size(0);
    auto log_u = state.narrow(1, 0, n);
    auto log_v = state.narrow(1, n, m);
    dot::sinkhorn::sinkhorn_dual_row_update_cuda(
        log_K.data_ptr<float>(),
        log_v.data_ptr<float>(),
        log_u_next.data_ptr<float>(),
        log_a.data_ptr<float>(),
        row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
        B,
        static_cast<int>(n),
        static_cast<int>(m),
        row_chunks,
        stream
    );
    dot::sinkhorn::sinkhorn_dual_col_update_cuda(
        log_K.data_ptr<float>(),
        log_u_next.data_ptr<float>(),
        log_v_next.data_ptr<float>(),
        log_b.data_ptr<float>(),
        col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
        B,
        static_cast<int>(n),
        static_cast<int>(m),
        col_chunks,
        stream
    );
    return torch::cat({log_u_next, log_v_next}, 1);
}

std::vector<torch::Tensor> sinkhorn_dual_vanilla_cuda(
    const torch::Tensor& log_alpha_f,
    float tau,
    int64_t n_iters,
    double tol,
    const torch::Tensor& log_a_f,
    const torch::Tensor& log_b_f,
    bool return_log,
    float reg_start,
    int64_t schedule_code
) {
    int B = log_alpha_f.size(0);
    int n = log_alpha_f.size(1);
    int m = log_alpha_f.size(2);
    auto options = log_alpha_f.options().dtype(torch::kFloat32);
    auto output = torch::empty({B, n, m}, options);
    auto log_K = torch::empty({B, n, m}, options);
    auto log_u = torch::zeros({B, n}, options);
    auto log_v = torch::zeros({B, m}, options);
    auto log_u_prev = tol > 0.0 ? torch::empty_like(log_u) : torch::Tensor();
    auto delta_tensor = torch::zeros({1}, options);
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    auto row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    auto row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    auto col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    auto col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float initial_tau = scheduled_tau_cuda(0, n_iters, tau, reg_start, schedule_code);
    dot::sinkhorn::sinkhorn_dual_init_cuda(
        log_alpha_f.data_ptr<float>(),
        log_K.data_ptr<float>(),
        log_u.data_ptr<float>(),
        log_v.data_ptr<float>(),
        B, n, m,
        initial_tau,
        stream
    );

    int64_t used_iters = 0;
    bool converged = false;
    for (int64_t iter = 0; iter < n_iters; ++iter) {
        if (schedule_code != kScheduleNone) {
            float tau_current = scheduled_tau_cuda(iter, n_iters, tau, reg_start, schedule_code);
            dot::sinkhorn::sinkhorn_dual_rescale_cuda(
                log_alpha_f.data_ptr<float>(),
                log_K.data_ptr<float>(),
                B, n, m,
                tau_current,
                stream
            );
        }
        if (tol > 0.0) {
            log_u_prev.copy_(log_u);
        }
        dot::sinkhorn::sinkhorn_dual_row_update_cuda(
            log_K.data_ptr<float>(),
            log_v.data_ptr<float>(),
            log_u.data_ptr<float>(),
            log_a_f.data_ptr<float>(),
            row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
            row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
            B, n, m,
            row_chunks,
            stream
        );
        dot::sinkhorn::sinkhorn_dual_col_update_cuda(
            log_K.data_ptr<float>(),
            log_u.data_ptr<float>(),
            log_v.data_ptr<float>(),
            log_b_f.data_ptr<float>(),
            col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
            col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
            B, n, m,
            col_chunks,
            stream
        );
        used_iters = iter + 1;
        if (tol > 0.0) {
            dot::sinkhorn::sinkhorn_max_abs_diff_cuda(
                log_u.data_ptr<float>(),
                log_u_prev.data_ptr<float>(),
                delta_tensor.data_ptr<float>(),
                B * n,
                stream
            );
            float delta = delta_tensor.item<float>();
            if (delta < static_cast<float>(tol)) {
                converged = true;
                break;
            }
        }
    }

    if (schedule_code != kScheduleNone) {
        dot::sinkhorn::sinkhorn_dual_rescale_cuda(
            log_alpha_f.data_ptr<float>(),
            log_K.data_ptr<float>(),
            B, n, m,
            tau,
            stream
        );
    }

    dot::sinkhorn::sinkhorn_dual_materialize_cuda(
        log_K.data_ptr<float>(),
        log_u.data_ptr<float>(),
        log_v.data_ptr<float>(),
        output.data_ptr<float>(),
        B, n, m,
        return_log,
        stream
    );

    auto used_tensor = torch::tensor(used_iters, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto converged_tensor = torch::tensor(static_cast<int64_t>(converged ? 1 : 0), torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return {output, used_tensor, converged_tensor};
}

std::vector<torch::Tensor> sinkhorn_dual_forward_cuda_impl(
    torch::Tensor log_alpha,
    double tau,
    int64_t n_iters,
    double tol,
    c10::optional<torch::Tensor> log_a,
    c10::optional<torch::Tensor> log_b,
    bool return_log,
    int64_t method,
    double omega,
    int64_t anderson_k,
    double mixing_beta,
    double lr,
    double beta1,
    double beta2,
    double eps_adam,
    bool bias_correction,
    double reg_start,
    int64_t schedule
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_iters >= 0, "n_iters must be >= 0");
    TORCH_CHECK(anderson_k >= 1, "anderson_k must be >= 1");

    int64_t B = log_alpha.size(0);
    int64_t n = log_alpha.size(1);
    int64_t m = log_alpha.size(2);
    if (log_a.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_a);
        TORCH_CHECK(log_a->dim() == 2 && log_a->size(0) == B && log_a->size(1) == n, "log_a must have shape [B, n]");
    }
    if (log_b.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_b);
        TORCH_CHECK(log_b->dim() == 2 && log_b->size(0) == B && log_b->size(1) == m, "log_b must have shape [B, m]");
    }

    auto options = log_alpha.options().dtype(torch::kFloat32);
    auto log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    auto log_a_f = default_log_marginal_cuda(log_a, static_cast<int>(B), static_cast<int>(n), options);
    auto log_b_f = default_log_marginal_cuda(log_b, static_cast<int>(B), static_cast<int>(m), options);

    if (method == kMethodVanilla) {
        return sinkhorn_dual_vanilla_cuda(
            log_alpha_f,
            static_cast<float>(tau),
            n_iters,
            tol,
            log_a_f,
            log_b_f,
            return_log,
            static_cast<float>(reg_start),
            schedule
        );
    }

    auto log_K = torch::empty({B, n, m}, options);
    auto log_u = torch::zeros({B, n}, options);
    auto log_v = torch::zeros({B, m}, options);
    auto log_u_target = torch::zeros_like(log_u);
    auto log_v_target = torch::zeros_like(log_v);
    auto log_u_next = torch::zeros_like(log_u);
    auto log_v_next = torch::zeros_like(log_v);
    auto delta_tensor = torch::zeros({1}, options);
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(static_cast<int>(m));
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(static_cast<int>(n));
    auto row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    auto row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    auto col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    auto col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float reg_start_f = static_cast<float>(reg_start);
    float tau_f = static_cast<float>(tau);
    float initial_tau = scheduled_tau_cuda(0, n_iters, tau_f, reg_start_f, schedule);

    dot::sinkhorn::sinkhorn_dual_init_cuda(
        log_alpha_f.data_ptr<float>(),
        log_K.data_ptr<float>(),
        log_u.data_ptr<float>(),
        log_v.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(n),
        static_cast<int>(m),
        initial_tau,
        stream
    );

    auto m_u = torch::Tensor();
    auto v_u = torch::Tensor();
    auto m_v = torch::Tensor();
    auto v_v = torch::Tensor();
    if (method == kMethodAdam) {
        m_u = torch::zeros_like(log_u);
        v_u = torch::zeros_like(log_u);
        m_v = torch::zeros_like(log_v);
        v_v = torch::zeros_like(log_v);
    }

    auto state = torch::Tensor();
    auto candidate_state = torch::Tensor();
    auto history_x = torch::Tensor();
    auto history_f = torch::Tensor();
    auto eye = torch::eye(anderson_k, options);
    if (method == kMethodAnderson) {
        state = torch::zeros({B, n + m}, options);
        candidate_state = torch::zeros_like(state);
        history_x = torch::empty({B, n + m, anderson_k}, options);
        history_f = torch::empty({B, n + m, anderson_k}, options);
    }

    int64_t used_iters = 0;
    bool converged = false;
    int64_t history_size = 0;

    for (int64_t iter = 0; iter < n_iters; ++iter) {
        if (schedule != kScheduleNone) {
            float tau_current = scheduled_tau_cuda(iter, n_iters, tau_f, reg_start_f, schedule);
            dot::sinkhorn::sinkhorn_dual_rescale_cuda(
                log_alpha_f.data_ptr<float>(),
                log_K.data_ptr<float>(),
                static_cast<int>(B),
                static_cast<int>(n),
                static_cast<int>(m),
                tau_current,
                stream
            );
        }

        if (method == kMethodAnderson) {
            auto x_new = dual_fixed_point_map_cuda(
                state,
                log_K,
                log_a_f,
                log_b_f,
                log_u_next,
                log_v_next,
                row_partial_max,
                row_partial_sum,
                col_partial_max,
                col_partial_sum,
                n,
                m,
                row_chunks,
                col_chunks,
                stream
            );
            auto residual = x_new - state;
            float delta = tol > 0.0 ? max_abs_diff_cuda(x_new, state, delta_tensor, stream) : 0.0f;

            if (history_size < anderson_k) {
                history_x.select(2, history_size).copy_(state);
                history_f.select(2, history_size).copy_(residual);
                history_size += 1;
            } else {
                history_x.narrow(2, 0, anderson_k - 1).copy_(history_x.narrow(2, 1, anderson_k - 1));
                history_f.narrow(2, 0, anderson_k - 1).copy_(history_f.narrow(2, 1, anderson_k - 1));
                history_x.select(2, anderson_k - 1).copy_(state);
                history_f.select(2, anderson_k - 1).copy_(residual);
            }

            if (history_size >= 2) {
                int64_t h = history_size;
                auto X = history_x.narrow(2, 0, h);
                auto F = history_f.narrow(2, 0, h);
                auto gram = torch::bmm(F.transpose(1, 2), F);
                auto eye_h = eye.narrow(0, 0, h).narrow(1, 0, h).unsqueeze(0);
                gram = gram + 1.0e-8f * eye_h;
                try {
                    auto rhs = torch::ones({B, h, 1}, options);
                    auto alpha = at::linalg_solve(gram, rhs).squeeze(-1);
                    auto denom = alpha.sum(1, true);
                    if (denom.abs().min().item<float>() >= 1.0e-12f) {
                        alpha = alpha / denom;
                        auto candidates = X + static_cast<float>(mixing_beta) * F;
                        candidate_state.copy_(torch::bmm(candidates, alpha.unsqueeze(-1)).squeeze(-1));
                        state.copy_(candidate_state);
                    } else {
                        state.copy_(x_new);
                    }
                } catch (const c10::Error&) {
                    state.copy_(x_new);
                }
            } else {
                state.copy_(x_new);
            }
            used_iters = iter + 1;
            if (tol > 0.0 && delta < static_cast<float>(tol)) {
                converged = true;
                break;
            }
            continue;
        }

        dot::sinkhorn::sinkhorn_dual_row_update_cuda(
            log_K.data_ptr<float>(),
            log_v.data_ptr<float>(),
            log_u_target.data_ptr<float>(),
            log_a_f.data_ptr<float>(),
            row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
            row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
            static_cast<int>(B),
            static_cast<int>(n),
            static_cast<int>(m),
            row_chunks,
            stream
        );
        if (method == kMethodMomentum) {
            log_u_next.copy_((1.0f - static_cast<float>(omega)) * log_u + static_cast<float>(omega) * log_u_target);
            if (candidate_improves_cuda(log_u_next, log_u_target, log_u, delta_tensor, stream)) {
                log_u_target.copy_(log_u_next);
            }
        } else if (method == kMethodAdam) {
            auto g_u = log_u_target - log_u;
            m_u = static_cast<float>(beta1) * m_u + (1.0f - static_cast<float>(beta1)) * g_u;
            v_u = static_cast<float>(beta2) * v_u + (1.0f - static_cast<float>(beta2)) * (g_u * g_u);
            auto m_u_hat = bias_correction ? m_u / (1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(iter + 1))) : m_u;
            auto v_u_hat = bias_correction ? v_u / (1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(iter + 1))) : v_u;
            log_u_next.copy_(log_u + static_cast<float>(lr) * m_u_hat / (torch::sqrt(v_u_hat) + static_cast<float>(eps_adam)));
            if (candidate_improves_cuda(log_u_next, log_u_target, log_u, delta_tensor, stream)) {
                log_u_target.copy_(log_u_next);
            }
        }

        dot::sinkhorn::sinkhorn_dual_col_update_cuda(
            log_K.data_ptr<float>(),
            log_u_target.data_ptr<float>(),
            log_v_target.data_ptr<float>(),
            log_b_f.data_ptr<float>(),
            col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
            col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
            static_cast<int>(B),
            static_cast<int>(n),
            static_cast<int>(m),
            col_chunks,
            stream
        );
        if (method == kMethodMomentum) {
            log_v_next.copy_((1.0f - static_cast<float>(omega)) * log_v + static_cast<float>(omega) * log_v_target);
            if (candidate_improves_cuda(log_v_next, log_v_target, log_v, delta_tensor, stream)) {
                log_v_target.copy_(log_v_next);
            }
        } else if (method == kMethodAdam) {
            auto g_v = log_v_target - log_v;
            m_v = static_cast<float>(beta1) * m_v + (1.0f - static_cast<float>(beta1)) * g_v;
            v_v = static_cast<float>(beta2) * v_v + (1.0f - static_cast<float>(beta2)) * (g_v * g_v);
            auto m_v_hat = bias_correction ? m_v / (1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(iter + 1))) : m_v;
            auto v_v_hat = bias_correction ? v_v / (1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(iter + 1))) : v_v;
            log_v_next.copy_(log_v + static_cast<float>(lr) * m_v_hat / (torch::sqrt(v_v_hat) + static_cast<float>(eps_adam)));
            if (candidate_improves_cuda(log_v_next, log_v_target, log_v, delta_tensor, stream)) {
                log_v_target.copy_(log_v_next);
            }
        }

        float delta = 0.0f;
        if (tol > 0.0) {
            float delta_u = max_abs_diff_cuda(log_u_target, log_u, delta_tensor, stream);
            float delta_v = max_abs_diff_cuda(log_v_target, log_v, delta_tensor, stream);
            delta = std::max(delta_u, delta_v);
        }
        log_u.copy_(log_u_target);
        log_v.copy_(log_v_target);
        used_iters = iter + 1;
        if (tol > 0.0 && delta < static_cast<float>(tol)) {
            converged = true;
            break;
        }
    }

    if (method == kMethodAnderson) {
        log_u = state.narrow(1, 0, n);
        log_v = state.narrow(1, n, m);
    }

    if (schedule != kScheduleNone) {
        dot::sinkhorn::sinkhorn_dual_rescale_cuda(
            log_alpha_f.data_ptr<float>(),
            log_K.data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(n),
            static_cast<int>(m),
            tau_f,
            stream
        );
    }

    auto log_P = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2);
    auto output = return_log ? log_P : log_P.exp();
    auto used_tensor = torch::tensor(used_iters, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto converged_tensor = torch::tensor(static_cast<int64_t>(converged ? 1 : 0), torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return {output, used_tensor, converged_tensor};
}

torch::Tensor sinkhorn_spectral_preflight_cuda_impl(
    torch::Tensor log_alpha,
    double tau,
    int64_t n_power
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_power >= 1, "n_power must be >= 1");

    int B = log_alpha.size(0);
    int n = log_alpha.size(1);
    int m = log_alpha.size(2);
    auto options = log_alpha.options().dtype(torch::kFloat32);
    auto log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    auto tau_est = torch::empty({B}, options);
    auto row_lse = torch::empty({B, n}, options);
    auto v_buf = torch::empty({B, m}, options);
    auto u_buf = torch::empty({B, n}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dot::sinkhorn::sinkhorn_spectral_preflight_cuda(
        log_alpha_f.data_ptr<float>(),
        tau_est.data_ptr<float>(),
        row_lse.data_ptr<float>(),
        v_buf.data_ptr<float>(),
        u_buf.data_ptr<float>(),
        B, n, m,
        static_cast<float>(tau),
        static_cast<int>(n_power),
        stream
    );

    return tau_est;
}

// =============================================================================
// CUDA Implementation Functions
// =============================================================================

torch::Tensor sinkhorn_cuda_impl(
    torch::Tensor log_alpha,
    double tau,
    int64_t n_iters,
    c10::optional<torch::Tensor> log_a,
    c10::optional<torch::Tensor> log_b
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_iters >= 0, "n_iters must be >= 0");

    int B = log_alpha.size(0);
    int n = log_alpha.size(1);
    int m = log_alpha.size(2);
    if (log_a.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_a);
        TORCH_CHECK(log_a->dim() == 2 && log_a->size(0) == B && log_a->size(1) == n, "log_a must have shape [B, n]");
    }
    if (log_b.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_b);
        TORCH_CHECK(log_b->dim() == 2 && log_b->size(0) == B && log_b->size(1) == m, "log_b must have shape [B, m]");
    }

    auto options = log_alpha.options().dtype(torch::kFloat32);
    torch::Tensor P = torch::empty({B, n, m}, options);

    torch::Tensor log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    torch::Tensor log_a_f = log_a.has_value() ? log_a->to(torch::kFloat32).contiguous() : torch::Tensor();
    torch::Tensor log_b_f = log_b.has_value() ? log_b->to(torch::kFloat32).contiguous() : torch::Tensor();
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    torch::Tensor row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dot::sinkhorn::sinkhorn_forward_cuda(
        log_alpha_f.data_ptr<float>(),
        P.data_ptr<float>(),
        log_a.has_value() ? log_a_f.data_ptr<float>() : nullptr,
        log_b.has_value() ? log_b_f.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
        B, n, m,
        row_chunks,
        col_chunks,
        static_cast<float>(tau),
        static_cast<int>(n_iters),
        /*return_log=*/false,
        stream
    );

    return P;
}

torch::Tensor sinkhorn_log_cuda_impl(
    torch::Tensor log_alpha,
    double tau,
    int64_t n_iters,
    c10::optional<torch::Tensor> log_a,
    c10::optional<torch::Tensor> log_b
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_iters >= 0, "n_iters must be >= 0");

    int B = log_alpha.size(0);
    int n = log_alpha.size(1);
    int m = log_alpha.size(2);
    if (log_a.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_a);
        TORCH_CHECK(log_a->dim() == 2 && log_a->size(0) == B && log_a->size(1) == n, "log_a must have shape [B, n]");
    }
    if (log_b.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_b);
        TORCH_CHECK(log_b->dim() == 2 && log_b->size(0) == B && log_b->size(1) == m, "log_b must have shape [B, m]");
    }

    auto options = log_alpha.options().dtype(torch::kFloat32);
    torch::Tensor log_P = torch::empty({B, n, m}, options);

    torch::Tensor log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    torch::Tensor log_a_f = log_a.has_value() ? log_a->to(torch::kFloat32).contiguous() : torch::Tensor();
    torch::Tensor log_b_f = log_b.has_value() ? log_b->to(torch::kFloat32).contiguous() : torch::Tensor();
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    torch::Tensor row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dot::sinkhorn::sinkhorn_forward_cuda(
        log_alpha_f.data_ptr<float>(),
        log_P.data_ptr<float>(),
        log_a.has_value() ? log_a_f.data_ptr<float>() : nullptr,
        log_b.has_value() ? log_b_f.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
        B, n, m,
        row_chunks,
        col_chunks,
        static_cast<float>(tau),
        static_cast<int>(n_iters),
        /*return_log=*/true,
        stream
    );

    return log_P;
}

std::vector<torch::Tensor> sinkhorn_with_grads_unrolled_cuda_impl(
    torch::Tensor log_alpha,
    torch::Tensor grad_P,
    double tau,
    int64_t n_iters,
    c10::optional<torch::Tensor> log_a,
    c10::optional<torch::Tensor> log_b
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    DOT_CHECK_INPUT_CUDA(grad_P);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(grad_P.sizes() == log_alpha.sizes(), "grad_P must match log_alpha shape");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_iters >= 0, "n_iters must be >= 0");

    int B = log_alpha.size(0);
    int n = log_alpha.size(1);
    int m = log_alpha.size(2);
    if (log_a.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_a);
        TORCH_CHECK(log_a->dim() == 2 && log_a->size(0) == B && log_a->size(1) == n, "log_a must have shape [B, n]");
    }
    if (log_b.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_b);
        TORCH_CHECK(log_b->dim() == 2 && log_b->size(0) == B && log_b->size(1) == m, "log_b must have shape [B, m]");
    }

    auto options = log_alpha.options().dtype(torch::kFloat32);
    torch::Tensor P = torch::empty({B, n, m}, options);
    torch::Tensor grad_log_alpha = torch::empty({B, n, m}, options);
    torch::Tensor grad_tau_tensor = torch::empty({B}, options);

    torch::Tensor log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    torch::Tensor grad_P_f = grad_P.to(torch::kFloat32).contiguous();
    torch::Tensor log_a_f = default_log_marginal_cuda(log_a, B, n, options);
    torch::Tensor log_b_f = default_log_marginal_cuda(log_b, B, m, options);
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    torch::Tensor row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor log_K = torch::empty({B, n, m}, options);
    torch::Tensor log_u_hist = torch::empty({n_iters, B, n}, options);
    torch::Tensor log_v_hist = torch::empty({n_iters, B, m}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dot::sinkhorn::sinkhorn_dual_forward_with_intermediates_cuda(
        log_alpha_f.data_ptr<float>(),
        P.data_ptr<float>(),
        log_K.data_ptr<float>(),
        log_u_hist.data_ptr<float>(),
        log_v_hist.data_ptr<float>(),
        log_a_f.data_ptr<float>(),
        log_b_f.data_ptr<float>(),
        row_chunks > 1 ? row_partial_max.data_ptr<float>() : nullptr,
        row_chunks > 1 ? row_partial_sum.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_max.data_ptr<float>() : nullptr,
        col_chunks > 1 ? col_partial_sum.data_ptr<float>() : nullptr,
        B, n, m,
        row_chunks,
        col_chunks,
        static_cast<float>(tau),
        static_cast<int>(n_iters),
        stream
    );

    dot::sinkhorn::sinkhorn_backward_unrolled_dual_cuda(
        log_alpha_f.data_ptr<float>(),
        log_K.data_ptr<float>(),
        P.data_ptr<float>(),
        grad_P_f.data_ptr<float>(),
        log_u_hist.data_ptr<float>(),
        log_v_hist.data_ptr<float>(),
        grad_log_alpha.data_ptr<float>(),
        grad_tau_tensor.data_ptr<float>(),
        B, n, m,
        static_cast<float>(tau),
        static_cast<int>(n_iters),
        stream
    );

    return {P, grad_log_alpha, grad_tau_tensor};
}

std::vector<torch::Tensor> sinkhorn_with_grads_implicit_cuda_impl(
    torch::Tensor log_alpha,
    torch::Tensor grad_P,
    double tau,
    int64_t n_iters,
    int64_t backward_iters,
    c10::optional<torch::Tensor> log_a,
    c10::optional<torch::Tensor> log_b
) {
    DOT_CHECK_INPUT_CUDA(log_alpha);
    DOT_CHECK_INPUT_CUDA(grad_P);
    TORCH_CHECK(log_alpha.dim() == 3, "log_alpha must be 3D [B, n, m]");
    TORCH_CHECK(grad_P.sizes() == log_alpha.sizes(), "grad_P must match log_alpha shape");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(n_iters >= 0, "n_iters must be >= 0");
    TORCH_CHECK(backward_iters >= 0, "backward_iters must be >= 0");

    int B = log_alpha.size(0);
    int n = log_alpha.size(1);
    int m = log_alpha.size(2);
    if (log_a.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_a);
        TORCH_CHECK(log_a->dim() == 2 && log_a->size(0) == B && log_a->size(1) == n, "log_a must have shape [B, n]");
    }
    if (log_b.has_value()) {
        DOT_CHECK_INPUT_CUDA(*log_b);
        TORCH_CHECK(log_b->dim() == 2 && log_b->size(0) == B && log_b->size(1) == m, "log_b must have shape [B, m]");
    }

    auto options = log_alpha.options().dtype(torch::kFloat32);
    torch::Tensor P = torch::empty({B, n, m}, options);
    torch::Tensor grad_log_alpha = torch::empty({B, n, m}, options);
    torch::Tensor grad_tau_tensor = torch::empty({B}, options);

    torch::Tensor log_alpha_f = log_alpha.to(torch::kFloat32).contiguous();
    torch::Tensor grad_P_f = grad_P.to(torch::kFloat32).contiguous();
    torch::Tensor log_a_f = log_a.has_value() ? log_a->to(torch::kFloat32).contiguous() : torch::Tensor();
    torch::Tensor log_b_f = log_b.has_value() ? log_b->to(torch::kFloat32).contiguous() : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto dual_forward = sinkhorn_dual_vanilla_cuda(
        log_alpha_f,
        static_cast<float>(tau),
        n_iters,
        0.0,
        log_a.has_value() ? log_a_f : default_log_marginal_cuda(log_a, B, n, options),
        log_b.has_value() ? log_b_f : default_log_marginal_cuda(log_b, B, m, options),
        false,
        -1.0f,
        kScheduleNone
    );
    P.copy_(dual_forward[0]);

    // Backward (implicit)
    dot::sinkhorn::sinkhorn_backward_implicit_cuda(
        log_alpha_f.data_ptr<float>(),
        P.data_ptr<float>(),
        grad_P_f.data_ptr<float>(),
        log_a.has_value() ? log_a_f.data_ptr<float>() : nullptr,
        log_b.has_value() ? log_b_f.data_ptr<float>() : nullptr,
        grad_log_alpha.data_ptr<float>(),
        grad_tau_tensor.data_ptr<float>(),
        B, n, m,
        static_cast<float>(tau),
        static_cast<int>(backward_iters),
        stream
    );

    return {P, grad_log_alpha, grad_tau_tensor};
}

} // anonymous namespace

// =============================================================================
// Register CUDA Implementations
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(dot, CUDA, m) {
    m.impl("sinkhorn", sinkhorn_cuda_impl);
    m.impl("sinkhorn_log", sinkhorn_log_cuda_impl);
    m.impl("sinkhorn_dual_forward", sinkhorn_dual_forward_cuda_impl);
    m.impl("sinkhorn_spectral_preflight", sinkhorn_spectral_preflight_cuda_impl);
    m.impl("sinkhorn_with_grads_unrolled", sinkhorn_with_grads_unrolled_cuda_impl);
    m.impl("sinkhorn_with_grads_implicit", sinkhorn_with_grads_implicit_cuda_impl);
}

#endif // USE_TORCH_LIBRARY
