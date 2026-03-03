// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file buffer.hpp
/// @brief Vendor-agnostic device buffer
/// @details Provides portable device buffer that resolves at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/device_buffer.hpp>
#elif DTL_ENABLE_HIP
#include <dtl/hip/device_buffer.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

/// @brief Vendor-agnostic device buffer (resolves to dtl::cuda::device_buffer<T>)
template <typename T>
using device_buffer = dtl::cuda::device_buffer<T>;

#elif DTL_ENABLE_HIP

/// @brief Vendor-agnostic device buffer (resolves to dtl::hip::device_buffer<T>)
template <typename T>
using device_buffer = dtl::hip::device_buffer<T>;

#else

/// @brief Stub device buffer when no GPU backend is enabled
template <typename T>
class device_buffer {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    device_buffer() noexcept = default;
    explicit device_buffer(size_type /*size*/, int /*device_id*/ = 0) {
        // No GPU backend — cannot allocate device memory
    }

    [[nodiscard]] pointer data() noexcept { return nullptr; }
    [[nodiscard]] size_type size() const noexcept { return 0; }
    [[nodiscard]] bool empty() const noexcept { return true; }
};

#endif

}  // namespace device
}  // namespace dtl
