// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file event.hpp
/// @brief Vendor-agnostic event types
/// @details Provides portable event types that resolve at compile time
///          based on which GPU backend is enabled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <backends/cuda/cuda_event.hpp>
#elif DTL_ENABLE_HIP
#include <backends/hip/hip_event.hpp>
#endif

namespace dtl {
namespace device {

#if DTL_ENABLE_CUDA

/// @brief Vendor-agnostic event flags (resolves to dtl::cuda::event_flags)
using event_flags = dtl::cuda::event_flags;

/// @brief Vendor-agnostic device event (resolves to dtl::cuda::cuda_event)
using device_event = dtl::cuda::cuda_event;

/// @brief Vendor-agnostic scoped timer (resolves to dtl::cuda::cuda_scoped_timer)
using scoped_timer = dtl::cuda::cuda_scoped_timer;

/// @brief Factory: create a synchronization-only event
inline device_event make_sync_event() { return dtl::cuda::make_sync_event(); }

/// @brief Factory: create a timing event
inline device_event make_timing_event() { return dtl::cuda::make_timing_event(); }

/// @brief Factory: create a blocking-sync event
inline device_event make_blocking_event() { return dtl::cuda::make_blocking_event(); }

#elif DTL_ENABLE_HIP

/// @brief Vendor-agnostic event flags (resolves to dtl::hip::event_flags)
using event_flags = dtl::hip::event_flags;

/// @brief Vendor-agnostic device event (resolves to dtl::hip::hip_event)
using device_event = dtl::hip::hip_event;

/// @brief Vendor-agnostic scoped timer (resolves to dtl::hip::hip_scoped_timer)
using scoped_timer = dtl::hip::hip_scoped_timer;

/// @brief Factory: create a synchronization-only event
inline device_event make_sync_event() { return dtl::hip::make_sync_event(); }

/// @brief Factory: create a timing event
inline device_event make_timing_event() { return dtl::hip::make_timing_event(); }

/// @brief Factory: create a blocking-sync event
inline device_event make_blocking_event() { return dtl::hip::make_blocking_event(); }

#endif

}  // namespace device
}  // namespace dtl
