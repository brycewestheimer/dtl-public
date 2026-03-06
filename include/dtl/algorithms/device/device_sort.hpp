// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_sort.hpp
/// @brief Device-side sort algorithm for device_view<T>
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/views/device_view.hpp>

#include <algorithm>
#include <functional>
#include <vector>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::algorithms::device {

/// @brief Sort a device view in ascending order
/// @note Uses host-staging copy. Future: replace with thrust::sort.
template <DeviceStorable T>
void sort_ascending(device_view<T> view) {
#if DTL_ENABLE_CUDA
    if (view.size() <= 1) return;
    std::vector<T> staging(view.size());
    cudaMemcpy(staging.data(), view.data(), view.size_bytes(), cudaMemcpyDeviceToHost);
    std::sort(staging.begin(), staging.end());
    cudaMemcpy(view.data(), staging.data(), view.size_bytes(), cudaMemcpyHostToDevice);
#else
    (void)view;
#endif
}

/// @brief Sort a device view in descending order
template <DeviceStorable T>
void sort_descending(device_view<T> view) {
#if DTL_ENABLE_CUDA
    if (view.size() <= 1) return;
    std::vector<T> staging(view.size());
    cudaMemcpy(staging.data(), view.data(), view.size_bytes(), cudaMemcpyDeviceToHost);
    std::sort(staging.begin(), staging.end(), std::greater<T>());
    cudaMemcpy(view.data(), staging.data(), view.size_bytes(), cudaMemcpyHostToDevice);
#else
    (void)view;
#endif
}

}  // namespace dtl::algorithms::device
