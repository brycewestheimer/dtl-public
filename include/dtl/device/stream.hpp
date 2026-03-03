// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stream.hpp
/// @brief Vendor-agnostic stream handle
/// @details Provides portable stream type that resolves at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/stream_handle.hpp>
#elif DTL_ENABLE_HIP
#include <dtl/hip/stream_handle.hpp>
#elif DTL_ENABLE_SYCL
#include <dtl/sycl/stream_handle.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

/// @brief Vendor-agnostic stream handle (resolves to dtl::cuda::stream_handle)
using stream_handle = dtl::cuda::stream_handle;

#elif DTL_ENABLE_HIP

/// @brief Vendor-agnostic stream handle (resolves to dtl::hip::stream_handle)
using stream_handle = dtl::hip::stream_handle;

#elif DTL_ENABLE_SYCL

/// @brief Vendor-agnostic stream handle (resolves to dtl::sycl::stream_handle)
using stream_handle = dtl::sycl::stream_handle;

#else

/// @brief Stub stream handle when no GPU backend is enabled
class stream_handle {
public:
    stream_handle() noexcept = default;
    explicit stream_handle(bool /*create*/) {
        // No GPU backend — no-op
    }
    bool synchronize() const { return true; }
    bool is_default() const noexcept { return true; }
};

#endif

}  // namespace device
}  // namespace dtl
