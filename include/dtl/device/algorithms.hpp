// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file algorithms.hpp
/// @brief Vendor-agnostic GPU algorithm functions
/// @details Provides portable device algorithm functions that resolve at compile time
///          based on which GPU backend is enabled.
///          Currently only CUDA has algorithm implementations; HIP parity is deferred.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/cuda_algorithms.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

// Unary operations
using dtl::cuda::for_each_device;
using dtl::cuda::transform_device;

// Reductions
using dtl::cuda::reduce_device;
using dtl::cuda::reduce_sum_device;
using dtl::cuda::reduce_min_device;
using dtl::cuda::reduce_max_device;

// Sorting
using dtl::cuda::sort_device;
using dtl::cuda::stable_sort_device;

// Fill & Copy
using dtl::cuda::fill_device;
using dtl::cuda::copy_device;

// Query operations
using dtl::cuda::count_device;
using dtl::cuda::count_if_device;
using dtl::cuda::find_device;
using dtl::cuda::find_if_device;

// Logical operations
using dtl::cuda::all_of_device;
using dtl::cuda::any_of_device;
using dtl::cuda::none_of_device;

// Synchronization
using dtl::cuda::synchronize_stream;
using dtl::cuda::synchronize_device;

#elif DTL_ENABLE_HIP

#include <dtl/hip/hip_algorithms.hpp>

// Unary operations
using dtl::hip::for_each_device;
using dtl::hip::transform_device;

// Reductions
using dtl::hip::reduce_device;
using dtl::hip::reduce_sum_device;
using dtl::hip::reduce_min_device;
using dtl::hip::reduce_max_device;

// Sorting
using dtl::hip::sort_device;
using dtl::hip::stable_sort_device;

// Fill & Copy
using dtl::hip::fill_device;
using dtl::hip::copy_device;

// Query operations
using dtl::hip::count_device;
using dtl::hip::count_if_device;
using dtl::hip::find_device;
using dtl::hip::find_if_device;

// Logical operations
using dtl::hip::all_of_device;
using dtl::hip::any_of_device;
using dtl::hip::none_of_device;

// Synchronization
using dtl::hip::synchronize_stream;
using dtl::hip::synchronize_device;

#endif

}  // namespace device
}  // namespace dtl
