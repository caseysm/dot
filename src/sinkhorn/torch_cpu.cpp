/**
 * @file torch_cpu.cpp
 * @brief Sinkhorn CPU Extension with PyTorch Autograd
 *
 * CPU implementations registered via TORCH_LIBRARY_IMPL for automatic dispatch.
 */

#include <torch/extension.h>
#include <vector>

#include "common/torch_utils.h"
#include "sinkhorn/kernels_cpu.h"

using namespace dot::common;

namespace {

// =============================================================================
// CPU Implementation Functions
// =============================================================================

std::vector<torch::Tensor> sinkhorn_forward_cpu_impl(
    torch::Tensor cost,
    double reg,
    int64_t max_iter,
    double tol,
    std::optional<torch::Tensor> a,
    std::optional<torch::Tensor> b
) {
    DOT_CHECK_INPUT_CPU(cost);
    TORCH_CHECK(cost.dim() == 3, "cost must be 3D (B, M, N)");
    TORCH_CHECK(cost.dtype() == torch::kFloat32, "cost must be float32");

    int B = cost.size(0);
    int M = cost.size(1);
    int N = cost.size(2);

    const float* a_ptr = nullptr;
    const float* b_ptr = nullptr;

    if (a.has_value()) {
        DOT_CHECK_INPUT_CPU(a.value());
        TORCH_CHECK(a.value().dim() == 2 && a.value().size(0) == B && a.value().size(1) == M);
        a_ptr = a.value().data_ptr<float>();
    }

    if (b.has_value()) {
        DOT_CHECK_INPUT_CPU(b.value());
        TORCH_CHECK(b.value().dim() == 2 && b.value().size(0) == B && b.value().size(1) == N);
        b_ptr = b.value().data_ptr<float>();
    }

    auto options = cost.options();
    torch::Tensor transport = torch::zeros({B, M, N}, options);

    dot::sinkhorn::sinkhorn_forward_cpu(
        cost.data_ptr<float>(),
        transport.data_ptr<float>(),
        a_ptr,
        b_ptr,
        B, M, N,
        static_cast<float>(reg),
        static_cast<int>(max_iter),
        static_cast<float>(tol)
    );

    return {transport};
}

torch::Tensor sinkhorn_backward_cpu_impl(
    torch::Tensor grad_output,
    torch::Tensor cost,
    torch::Tensor transport,
    double reg
) {
    DOT_CHECK_INPUT_CPU(grad_output);
    DOT_CHECK_INPUT_CPU(cost);
    DOT_CHECK_INPUT_CPU(transport);

    int B = cost.size(0);
    int M = cost.size(1);
    int N = cost.size(2);

    auto options = cost.options();
    torch::Tensor grad_cost = torch::zeros({B, M, N}, options);

    dot::sinkhorn::sinkhorn_backward_cpu(
        grad_output.data_ptr<float>(),
        cost.data_ptr<float>(),
        transport.data_ptr<float>(),
        grad_cost.data_ptr<float>(),
        B, M, N,
        static_cast<float>(reg)
    );

    return grad_cost;
}

} // anonymous namespace

// =============================================================================
// Register CPU Implementations
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(dot, CPU, m) {
    m.impl("sinkhorn_forward", sinkhorn_forward_cpu_impl);
    m.impl("sinkhorn_backward", sinkhorn_backward_cpu_impl);
}

#endif // USE_TORCH_LIBRARY
