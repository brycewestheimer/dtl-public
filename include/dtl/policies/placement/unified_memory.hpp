// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file unified_memory.hpp
/// @brief Unified (managed) memory placement policy
/// @details Memory is accessible from both host and device with automatic migration.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {

/// @brief Unified memory placement using CUDA managed memory or equivalent
/// @details Memory is automatically migrated between host and device as needed.
///          Simplifies programming but may have performance implications due
///          to page migration overhead.
///
/// @par Characteristics:
/// - Accessible from both host and device
/// - Automatic page migration
/// - May have higher latency for first access after migration
/// - Simplifies code but requires careful performance consideration
struct unified_memory {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief Get the preferred memory location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::unified;
    }

    /// @brief Check if memory is host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return true;
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return true;
    }

    /// @brief Check if explicit copies are needed
    [[nodiscard]] static constexpr bool requires_explicit_copy() noexcept {
        return false;  // Automatic migration
    }

    /// @brief Check if prefetching is recommended
    [[nodiscard]] static constexpr bool supports_prefetch() noexcept {
        return true;
    }

    /// @brief Get device ID (applies to all devices in unified space)
    [[nodiscard]] static constexpr int device_id() noexcept {
        return -1;  // Applicable to all devices
    }

    /// @brief Hint to prefetch data to device
    /// @details Initiates an asynchronous prefetch of unified (managed)
    ///          memory to the specified GPU device. This is a performance
    ///          hint; the CUDA runtime may migrate pages proactively to
    ///          avoid demand-paging faults on first device access.
    ///          When CUDA is not available, this is a no-op.
    /// @param ptr Pointer to managed memory
    /// @param bytes Number of bytes to prefetch
    /// @param device Target device ID
    static void prefetch_to_device([[maybe_unused]] void* ptr,
                                   [[maybe_unused]] size_type bytes,
                                   [[maybe_unused]] int device = 0) {
#if DTL_ENABLE_CUDA
        cudaMemPrefetchAsync(ptr, bytes, device, nullptr);
#endif
        // No-op when CUDA is not available
    }

    /// @brief Hint to prefetch data to host
    /// @details Initiates an asynchronous prefetch of unified (managed)
    ///          memory to host (CPU) memory. This is a performance hint
    ///          that avoids demand-paging faults when the host next
    ///          accesses the data. When CUDA is not available, this is
    ///          a no-op.
    /// @param ptr Pointer to managed memory
    /// @param bytes Number of bytes to prefetch
    static void prefetch_to_host([[maybe_unused]] void* ptr,
                                 [[maybe_unused]] size_type bytes) {
#if DTL_ENABLE_CUDA
        cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, nullptr);
#endif
        // No-op when CUDA is not available
    }
};

}  // namespace dtl
