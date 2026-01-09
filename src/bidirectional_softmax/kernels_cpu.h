/**
 * @file kernels_cpu.h
 * @brief Bidirectional Softmax CPU kernel declarations
 */

#pragma once

namespace dot {
namespace bidirectional_softmax {

/**
 * Forward pass: compute bidirectional softmax on CPU.
 */
void bidirectional_softmax_forward_cpu(
    const float* sim_matrix,
    float* output,
    float* row_softmax_buf,
    float* col_softmax_buf,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * Backward pass: compute gradients on CPU.
 */
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
);

} // namespace bidirectional_softmax
} // namespace dot
