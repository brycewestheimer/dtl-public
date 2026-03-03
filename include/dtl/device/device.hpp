// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device.hpp
/// @brief Master header for vendor-agnostic device abstraction
/// @details Includes all dtl::device:: headers. Users can include this single
///          header for portable GPU code that works on both NVIDIA and AMD
///          hardware without source changes.
///
/// @code
///   #include <dtl/device/device.hpp>
///
///   // Portable: works with CUDA or HIP backend
///   int n = dtl::device::device_count();
///   dtl::device::device_guard guard(0);
///   dtl::device::device_buffer<float> buf(1024);
/// @endcode
///
/// @since 0.1.0

#pragma once

#include <dtl/device/device_query.hpp>
#include <dtl/device/device_guard.hpp>
#include <dtl/device/stream.hpp>
#include <dtl/device/buffer.hpp>
#include <dtl/device/event.hpp>
#include <dtl/device/algorithms.hpp>
#include <dtl/device/memory_space.hpp>
