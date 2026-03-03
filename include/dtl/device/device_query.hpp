// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_query.hpp
/// @brief Vendor-agnostic device query functions
/// @details Provides portable device management that resolves at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/device_guard.hpp>
#elif DTL_ENABLE_HIP
#include <dtl/hip/device_guard.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

using dtl::cuda::current_device_id;
using dtl::cuda::device_count;
inline constexpr int invalid_device_id = dtl::cuda::invalid_device_id;

#elif DTL_ENABLE_HIP

using dtl::hip::current_device_id;
using dtl::hip::device_count;
inline constexpr int invalid_device_id = dtl::hip::invalid_device_id;

#else

/// @brief Get current device ID (stub — no GPU backend)
/// @return Always returns -1
[[nodiscard]] inline int current_device_id() noexcept { return -1; }

/// @brief Get device count (stub — no GPU backend)
/// @return Always returns 0
[[nodiscard]] inline int device_count() noexcept { return 0; }

/// @brief Invalid device ID sentinel
inline constexpr int invalid_device_id = -1;

#endif

}  // namespace device
}  // namespace dtl
