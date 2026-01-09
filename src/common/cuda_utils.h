/**
 * @file cuda_utils.h
 * @brief Common CUDA utilities
 */

#pragma once

#include <cuda_runtime.h>

namespace dot {
namespace common {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// Common CUDA constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Get number of blocks for a given number of elements and block size
inline int get_num_blocks(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

} // namespace common
} // namespace dot
