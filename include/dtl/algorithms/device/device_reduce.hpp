// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_reduce.hpp
/// @brief Device-side reduction algorithms for device_view<T>
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/views/device_view.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::algorithms::device {

/// @brief Sum-reduce a device view
/// @note Uses host-staging copy. Future: replace with thrust::reduce.
template <DeviceStorable T>
T reduce_sum(device_view<const T> view) {
#if DTL_ENABLE_CUDA
    if (view.empty()) return T{};
    std::vector<T> staging(view.size());
    cudaMemcpy(staging.data(), view.data(), view.size_bytes(), cudaMemcpyDeviceToHost);
    T sum{};
    for (const auto& v : staging) sum = sum + v;
    return sum;
#else
    (void)view;
    return T{};
#endif
}

/// @brief Min-reduce a device view
template <DeviceStorable T>
T reduce_min(device_view<const T> view) {
#if DTL_ENABLE_CUDA
    if (view.empty()) return T{};
    std::vector<T> staging(view.size());
    cudaMemcpy(staging.data(), view.data(), view.size_bytes(), cudaMemcpyDeviceToHost);
    return *std::min_element(staging.begin(), staging.end());
#else
    (void)view;
    return T{};
#endif
}

/// @brief Max-reduce a device view
template <DeviceStorable T>
T reduce_max(device_view<const T> view) {
#if DTL_ENABLE_CUDA
    if (view.empty()) return T{};
    std::vector<T> staging(view.size());
    cudaMemcpy(staging.data(), view.data(), view.size_bytes(), cudaMemcpyDeviceToHost);
    return *std::max_element(staging.begin(), staging.end());
#else
    (void)view;
    return T{};
#endif
}

}  // namespace dtl::algorithms::device
