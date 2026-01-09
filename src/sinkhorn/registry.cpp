/**
 * @file registry.cpp
 * @brief Schema definitions for Sinkhorn optimal transport operators
 *
 * Defines the operator schemas (m.def calls) for Sinkhorn algorithm.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(dot, m) {
    // =========================================================================
    // SINKHORN OPTIMAL TRANSPORT
    // =========================================================================
    //
    // Entropy-regularized optimal transport using Sinkhorn-Knopp algorithm.
    //
    // Given cost matrix C, source distribution a, target distribution b:
    //   min_P <C, P> - reg * H(P)
    //   s.t. P @ 1 = a, P^T @ 1 = b, P >= 0

    // Forward pass: compute optimal transport plan
    m.def("sinkhorn_forward(Tensor cost, float reg, int max_iter, float tol, Tensor? a, Tensor? b) -> Tensor[]");

    // Backward pass: compute gradient w.r.t. cost matrix
    m.def("sinkhorn_backward(Tensor grad_output, Tensor cost, Tensor transport, float reg) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
