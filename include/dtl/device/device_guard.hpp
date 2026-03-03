// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_guard.hpp
/// @brief Vendor-agnostic RAII device scope guard
/// @details Provides portable device guard that resolves at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/device_guard.hpp>
#elif DTL_ENABLE_HIP
#include <dtl/hip/device_guard.hpp>
#elif DTL_ENABLE_SYCL
#include <dtl/sycl/device_guard.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

/// @brief Vendor-agnostic RAII device guard (resolves to dtl::cuda::device_guard)
using device_guard = dtl::cuda::device_guard;

#elif DTL_ENABLE_HIP

/// @brief Vendor-agnostic RAII device guard (resolves to dtl::hip::device_guard)
using device_guard = dtl::hip::device_guard;

#elif DTL_ENABLE_SYCL

/// @brief Vendor-agnostic RAII device guard (resolves to dtl::sycl::device_guard)
using device_guard = dtl::sycl::device_guard;

#else

/// @brief Stub device guard when no GPU backend is enabled
/// @details Static asserts on construction to provide a clear error message.
class device_guard {
public:
    explicit device_guard(int /*target_device*/) noexcept {
        static_assert(sizeof(device_guard) == 0,
                      "dtl::device::device_guard requires DTL_ENABLE_CUDA, DTL_ENABLE_HIP, or DTL_ENABLE_SYCL");
    }
};

#endif

}  // namespace device
}  // namespace dtl
