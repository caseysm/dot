/**
 * @file torch_cuda.cpp
 * @brief Bidirectional Softmax CUDA Extension with PyTorch bindings
 *
 * CUDA implementations registered via TORCH_LIBRARY_IMPL for automatic dispatch.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#include "common/torch_utils.h"
#include "bidirectional_softmax/kernels.cuh"

using namespace dot::common;

namespace {

// =============================================================================
// Helper: Create default lengths tensor
// =============================================================================

torch::Tensor make_default_lengths_cuda(int B, int L1, int L2, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor lengths = torch::empty({B, 2}, options);

    // Fill with L1, L2 for each batch
    auto lengths_cpu = torch::empty({B, 2}, torch::TensorOptions().dtype(torch::kInt32));
    auto acc = lengths_cpu.accessor<int32_t, 2>();
    for (int b = 0; b < B; b++) {
        acc[b][0] = L1;
        acc[b][1] = L2;
    }
    return lengths_cpu.to(device);
}

// =============================================================================
// CUDA Implementation Functions
// =============================================================================

std::vector<torch::Tensor> bidirectional_softmax_cuda_impl(
    torch::Tensor sim_matrix,
    double tau,
    torch::Tensor lengths
) {
    DOT_CHECK_INPUT_CUDA(sim_matrix);
    DOT_CHECK_INPUT_CUDA(lengths);
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

    dot::bidirectional_softmax::bidirectional_softmax_forward_cuda(
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

std::vector<torch::Tensor> bidirectional_softmax_backward_cuda_impl(
    torch::Tensor sim_matrix,
    torch::Tensor output,
    torch::Tensor grad_output,
    torch::Tensor row_softmax,
    torch::Tensor col_softmax,
    double tau,
    torch::Tensor lengths
) {
    DOT_CHECK_INPUT_CUDA(sim_matrix);
    DOT_CHECK_INPUT_CUDA(output);
    DOT_CHECK_INPUT_CUDA(grad_output);
    DOT_CHECK_INPUT_CUDA(row_softmax);
    DOT_CHECK_INPUT_CUDA(col_softmax);
    DOT_CHECK_INPUT_CUDA(lengths);

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

    // Workspace tensors
    torch::Tensor d_product = torch::empty({B, max_L1, max_L2}, options);
    torch::Tensor d_row_softmax = torch::empty({B, max_L1, max_L2}, options);
    torch::Tensor d_col_softmax = torch::empty({B, max_L1, max_L2}, options);

    torch::Tensor sim_f = sim_matrix.to(torch::kFloat32).contiguous();
    torch::Tensor output_f = output.to(torch::kFloat32).contiguous();
    torch::Tensor grad_output_f = grad_output.to(torch::kFloat32).contiguous();
    torch::Tensor row_sm_f = row_softmax.to(torch::kFloat32).contiguous();
    torch::Tensor col_sm_f = col_softmax.to(torch::kFloat32).contiguous();

    dot::bidirectional_softmax::bidirectional_softmax_backward_cuda(
        sim_f.data_ptr<float>(),
        output_f.data_ptr<float>(),
        grad_output_f.data_ptr<float>(),
        row_sm_f.data_ptr<float>(),
        col_sm_f.data_ptr<float>(),
        grad_sim.data_ptr<float>(),
        grad_tau.data_ptr<float>(),
        d_product.data_ptr<float>(),
        d_row_softmax.data_ptr<float>(),
        d_col_softmax.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        B, max_L1, max_L2,
        static_cast<float>(tau)
    );

    return {grad_sim, grad_tau};
}

} // anonymous namespace

// =============================================================================
// Register CUDA Implementations
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(dot, CUDA, m) {
    m.impl("bidirectional_softmax", bidirectional_softmax_cuda_impl);
    m.impl("bidirectional_softmax_backward", bidirectional_softmax_backward_cuda_impl);
}

#endif // USE_TORCH_LIBRARY
