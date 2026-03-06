// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_fill.hpp
/// @brief Device-side fill algorithm for device_view<T>
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/views/device_view.hpp>

#include <vector>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::algorithms::device {

/// @brief Fill a device view with a value
/// @tparam T Trivially copyable element type
/// @param view Device view to fill
/// @param value Value to fill with
/// @note Uses host-staging memcpy. Future: replace with thrust::fill or custom kernel.
template <DeviceStorable T>
void fill(device_view<T> view, const T& value) {
#if DTL_ENABLE_CUDA
    if (view.empty()) return;
    std::vector<T> staging(view.size(), value);
    cudaMemcpy(view.data(), staging.data(), view.size_bytes(), cudaMemcpyHostToDevice);
#else
    (void)view;
    (void)value;
#endif
}

}  // namespace dtl::algorithms::device
