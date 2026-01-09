/**
 * @file registry.cpp
 * @brief Schema definitions for Bidirectional Softmax operators
 *
 * Defines the operator schemas for bidirectional softmax.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * API:
 *   - bidirectional_softmax(sim, tau, lengths) -> [P, row_sm, col_sm]
 *   - bidirectional_softmax_backward(sim, output, grad_output, row_sm, col_sm, tau, lengths) -> [grad_sim, grad_tau]
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(dot, m) {
    // =========================================================================
    // BIDIRECTIONAL SOFTMAX
    // =========================================================================
    //
    // Computes soft matching without monotonic constraint:
    //   out[i,j] = sqrt(eps + softmax(sim/T, row) * softmax(sim/T, col))
    //
    // Input: sim [B, L1, L2] - similarity scores
    // Output: P [B, L1, L2] - soft matching posteriors
    //
    // Also returns row and column softmax buffers for efficient backward pass.

    // Forward pass: returns [P, row_softmax, col_softmax]
    m.def("bidirectional_softmax(Tensor sim, float tau, Tensor lengths) -> Tensor[]");

    // Backward pass: returns [grad_sim, grad_tau]
    m.def("bidirectional_softmax_backward(Tensor sim, Tensor output, Tensor grad_output, Tensor row_softmax, Tensor col_softmax, float tau, Tensor lengths) -> Tensor[]");
}

#endif // USE_TORCH_LIBRARY
