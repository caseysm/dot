/**
 * @file torch_cuda.cpp
 * @brief Sinkhorn CUDA Extension with PyTorch Autograd
 *
 * CUDA implementations registered via TORCH_LIBRARY_IMPL for automatic dispatch.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#include "common/torch_utils.h"
#include "sinkhorn/kernels.cuh"

using namespace dot::common;

namespace {

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
    torch::Tensor log_a_f = log_a.has_value() ? log_a->to(torch::kFloat32).contiguous() : torch::Tensor();
    torch::Tensor log_b_f = log_b.has_value() ? log_b->to(torch::kFloat32).contiguous() : torch::Tensor();
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    torch::Tensor row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();

    // Allocate intermediates
    torch::Tensor log_X = torch::empty({B, n_iters + 1, n, m}, options);
    torch::Tensor log_Y = torch::empty({B, n_iters, n, m}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Forward with intermediates
    dot::sinkhorn::sinkhorn_forward_with_intermediates_cuda(
        log_alpha_f.data_ptr<float>(),
        P.data_ptr<float>(),
        log_X.data_ptr<float>(),
        log_Y.data_ptr<float>(),
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
        stream
    );

    // Backward (unrolled)
    dot::sinkhorn::sinkhorn_backward_unrolled_cuda(
        log_alpha_f.data_ptr<float>(),
        P.data_ptr<float>(),
        grad_P_f.data_ptr<float>(),
        log_X.data_ptr<float>(),
        log_Y.data_ptr<float>(),
        log_a.has_value() ? log_a_f.data_ptr<float>() : nullptr,
        log_b.has_value() ? log_b_f.data_ptr<float>() : nullptr,
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
    int row_chunks = dot::sinkhorn::sinkhorn_row_chunks(m);
    int col_chunks = dot::sinkhorn::sinkhorn_col_chunks(n);
    torch::Tensor row_partial_max = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor row_partial_sum = row_chunks > 1 ? torch::empty({B * n, row_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_max = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();
    torch::Tensor col_partial_sum = col_chunks > 1 ? torch::empty({B * m, col_chunks}, options) : torch::Tensor();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Forward
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
    m.impl("sinkhorn_with_grads_unrolled", sinkhorn_with_grads_unrolled_cuda_impl);
    m.impl("sinkhorn_with_grads_implicit", sinkhorn_with_grads_implicit_cuda_impl);
}

#endif // USE_TORCH_LIBRARY
