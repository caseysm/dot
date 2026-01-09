/**
 * @file torch_utils.h
 * @brief Common PyTorch utilities and macros
 */

#pragma once

#include <torch/extension.h>

namespace dot {
namespace common {

// Input validation macros
#define DOT_CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define DOT_CHECK_CPU(x) TORCH_CHECK((x).device().is_cpu(), #x " must be a CPU tensor")
#define DOT_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define DOT_CHECK_INPUT_CUDA(x) DOT_CHECK_CUDA(x); DOT_CHECK_CONTIGUOUS(x)
#define DOT_CHECK_INPUT_CPU(x) DOT_CHECK_CPU(x); DOT_CHECK_CONTIGUOUS(x)

} // namespace common
} // namespace dot
