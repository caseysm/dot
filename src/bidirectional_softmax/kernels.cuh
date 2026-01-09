/**
 * @file kernels.cuh
 * @brief Bidirectional Softmax CUDA kernel declarations
 *
 * Computes soft matching without monotonic constraint:
 *   out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))
 */

#pragma once

namespace dot {
namespace bidirectional_softmax {

/**
 * Forward pass: compute bidirectional softmax.
 *
 * @param sim_matrix Input similarity scores [B, L1, L2]
 * @param output Output soft matching [B, L1, L2]
 * @param row_softmax_buf Workspace for row softmax [B, L1, L2]
 * @param col_softmax_buf Workspace for col softmax [B, L1, L2]
 * @param lengths Sequence lengths [B, 2] (L1, L2 per batch)
 * @param B Batch size
 * @param max_L1 Maximum L1 dimension
 * @param max_L2 Maximum L2 dimension
 * @param T Temperature parameter
 */
void bidirectional_softmax_forward_cuda(
    const float* sim_matrix,
    float* output,
    float* row_softmax_buf,
    float* col_softmax_buf,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * Backward pass: compute gradients for bidirectional softmax.
 *
 * @param sim_matrix Input similarity scores [B, L1, L2]
 * @param output Forward output [B, L1, L2]
 * @param grad_output Gradient of loss w.r.t. output [B, L1, L2]
 * @param row_softmax_buf Row softmax from forward [B, L1, L2]
 * @param col_softmax_buf Col softmax from forward [B, L1, L2]
 * @param grad_sim Output gradient w.r.t. sim_matrix [B, L1, L2]
 * @param grad_T Output gradient w.r.t. temperature [B]
 * @param d_product Workspace [B, L1, L2]
 * @param d_row_softmax Workspace [B, L1, L2]
 * @param d_col_softmax Workspace [B, L1, L2]
 * @param lengths Sequence lengths [B, 2]
 * @param B Batch size
 * @param max_L1 Maximum L1 dimension
 * @param max_L2 Maximum L2 dimension
 * @param T Temperature parameter
 */
void bidirectional_softmax_backward_cuda(
    const float* sim_matrix,
    const float* output,
    const float* grad_output,
    const float* row_softmax_buf,
    const float* col_softmax_buf,
    float* grad_sim,
    float* grad_T,
    float* d_product,
    float* d_row_softmax,
    float* d_col_softmax,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

} // namespace bidirectional_softmax
} // namespace dot
