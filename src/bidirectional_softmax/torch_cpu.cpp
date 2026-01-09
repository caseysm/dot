/**
 * @file torch_cpu.cpp
 * @brief Bidirectional Softmax CPU Extension with PyTorch bindings
 *
 * CPU implementations registered via TORCH_LIBRARY_IMPL for automatic dispatch.
 */

#include <torch/extension.h>
#include <vector>

#include "common/torch_utils.h"
#include "bidirectional_softmax/kernels_cpu.h"

using namespace dot::common;

namespace {

// =============================================================================
// CPU Implementation Functions
// =============================================================================

std::vector<torch::Tensor> bidirectional_softmax_cpu_impl(
    torch::Tensor sim_matrix,
    double tau,
    torch::Tensor lengths
) {
    DOT_CHECK_INPUT_CPU(sim_matrix);
    DOT_CHECK_INPUT_CPU(lengths);
    TORCH_CHECK(sim_matrix.dim() == 3, "sim_matrix must be 3D [B, L1, L2]");
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(1) == 2, "lengths must be [B, 2]");
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");

    int B = sim_matrix.size(0);
    int max_L1 = sim_matrix.size(1);
    int max_L2 = sim_matrix.size(2);

    auto options = sim_matrix.options().dtype(torch::kFloat32);
    torch::Tensor output = torch::empty({B, max_L1, max_L2}, options);
    torch::Tensor row_softmax = torch::empty({B, max_L1, max_L2}, options);
    torch::Tensor col_softmax = torch::empty({B, max_L1, max_L2}, options);

    torch::Tensor sim_f = sim_matrix.to(torch::kFloat32).contiguous();

    dot::bidirectional_softmax::bidirectional_softmax_forward_cpu(
        sim_f.data_ptr<float>(),
        output.data_ptr<float>(),
        row_softmax.data_ptr<float>(),
        col_softmax.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        B, max_L1, max_L2,
        static_cast<float>(tau)
    );

    return {output, row_softmax, col_softmax};
}

std::vector<torch::Tensor> bidirectional_softmax_backward_cpu_impl(
    torch::Tensor sim_matrix,
    torch::Tensor output,
    torch::Tensor grad_output,
    torch::Tensor row_softmax,
    torch::Tensor col_softmax,
    double tau,
    torch::Tensor lengths
) {
    DOT_CHECK_INPUT_CPU(sim_matrix);
    DOT_CHECK_INPUT_CPU(output);
    DOT_CHECK_INPUT_CPU(grad_output);
    DOT_CHECK_INPUT_CPU(row_softmax);
    DOT_CHECK_INPUT_CPU(col_softmax);
    DOT_CHECK_INPUT_CPU(lengths);

    TORCH_CHECK(sim_matrix.dim() == 3, "sim_matrix must be 3D [B, L1, L2]");
    TORCH_CHECK(output.sizes() == sim_matrix.sizes(), "output must match sim_matrix shape");
    TORCH_CHECK(grad_output.sizes() == sim_matrix.sizes(), "grad_output must match sim_matrix shape");
    TORCH_CHECK(row_softmax.sizes() == sim_matrix.sizes(), "row_softmax must match sim_matrix shape");
    TORCH_CHECK(col_softmax.sizes() == sim_matrix.sizes(), "col_softmax must match sim_matrix shape");
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(1) == 2, "lengths must be [B, 2]");
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");

    int B = sim_matrix.size(0);
    int max_L1 = sim_matrix.size(1);
    int max_L2 = sim_matrix.size(2);

    auto options = sim_matrix.options().dtype(torch::kFloat32);
    torch::Tensor grad_sim = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_tau = torch::zeros({B}, options);

    torch::Tensor sim_f = sim_matrix.to(torch::kFloat32).contiguous();
    torch::Tensor output_f = output.to(torch::kFloat32).contiguous();
    torch::Tensor grad_output_f = grad_output.to(torch::kFloat32).contiguous();
    torch::Tensor row_sm_f = row_softmax.to(torch::kFloat32).contiguous();
    torch::Tensor col_sm_f = col_softmax.to(torch::kFloat32).contiguous();

    dot::bidirectional_softmax::bidirectional_softmax_backward_cpu(
        sim_f.data_ptr<float>(),
        output_f.data_ptr<float>(),
        grad_output_f.data_ptr<float>(),
        row_sm_f.data_ptr<float>(),
        col_sm_f.data_ptr<float>(),
        grad_sim.data_ptr<float>(),
        grad_tau.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        B, max_L1, max_L2,
        static_cast<float>(tau)
    );

    return {grad_sim, grad_tau};
}

} // anonymous namespace

// =============================================================================
// Register CPU Implementations
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(dot, CPU, m) {
    m.impl("bidirectional_softmax", bidirectional_softmax_cpu_impl);
    m.impl("bidirectional_softmax_backward", bidirectional_softmax_backward_cpu_impl);
}

#endif // USE_TORCH_LIBRARY
