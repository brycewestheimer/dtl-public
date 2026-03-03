// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory_space.hpp
/// @brief Vendor-agnostic device memory space types
/// @details Provides portable memory space types that resolve at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/memory/cuda_memory_space.hpp>
#elif DTL_ENABLE_HIP
#include <backends/hip/hip_memory_space.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

/// @brief Vendor-agnostic device memory space (resolves to cuda_device_memory_space)
using device_memory_space = dtl::cuda::cuda_device_memory_space;

/// @brief Vendor-agnostic unified memory space (resolves to cuda_unified_memory_space)
using unified_memory_space = dtl::cuda::cuda_unified_memory_space;

#elif DTL_ENABLE_HIP

/// @brief Vendor-agnostic device memory space (resolves to hip_memory_space)
using device_memory_space = dtl::hip::hip_memory_space;

// HIP managed memory space maps to hip_managed_memory_space
/// @brief Vendor-agnostic unified/managed memory space (resolves to hip_managed_memory_space)
using unified_memory_space = dtl::hip::hip_managed_memory_space;

#endif

}  // namespace device
}  // namespace dtl
