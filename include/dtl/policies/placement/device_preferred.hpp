// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_preferred.hpp
/// @brief Device-preferred placement policy with host fallback
/// @details Memory is allocated on GPU if available, falls back to host.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

namespace dtl {

/// @brief Device-preferred placement with automatic host fallback
/// @details Attempts to allocate memory on GPU device, but falls back
///          to host memory if device memory is unavailable or insufficient.
///          Useful for systems that may or may not have GPU support.
///
/// @par Fallback Semantics
/// The current implementation uses compile-time fallback based on whether
/// CUDA/HIP is enabled. Future versions will support runtime fallback on
/// allocation failure.
///
/// @par Actual Location Query
/// Containers using this policy expose `actual_location()` to query
/// where memory was actually allocated.
///
/// @par Example
/// @code
/// // On CUDA-enabled build: allocates on device
/// // On non-CUDA build: allocates on host
/// dtl::distributed_vector<float, dtl::device_preferred> vec(1000, ctx);
/// @endcode
struct device_preferred {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief Get the preferred memory location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::device;
    }

    /// @brief Get the fallback memory location
    [[nodiscard]] static constexpr memory_location fallback_location() noexcept {
        return memory_location::host;
    }

    /// @brief Check if memory is host accessible
    /// @note May be true if fallback is used
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        // If using fallback (host), it's host accessible
        // Otherwise (device), not host accessible without copy
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return false;  // Primary location is device
#else
        return true;   // Fallback to host
#endif
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return true;   // Preferred location
#else
        return false;  // Fallback to host, not device accessible
#endif
    }

    /// @brief Check if fallback to host is allowed
    [[nodiscard]] static constexpr bool allows_fallback() noexcept {
        return true;
    }

    /// @brief Get default device ID
    [[nodiscard]] static constexpr int device_id() noexcept {
        return 0;  // Default to device 0
    }

    /// @brief Runtime check if currently using fallback
    /// @note Compile-time fallback in current implementation
    /// @return true if host fallback is active
    [[nodiscard]] static bool using_fallback() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return false;
#else
        return true;
#endif
    }

    /// @brief Get actual memory location (compile-time for now)
    /// @return The memory location where data is actually stored
    [[nodiscard]] static constexpr memory_location actual_location() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return memory_location::device;
#else
        return memory_location::host;
#endif
    }
};

/// @brief Templated device-preferred with specific target device
/// @tparam DeviceId Target device ID
/// @details Like device_preferred but targets a specific device.
///          Falls back to host if device is unavailable.
template <int DeviceId = 0>
struct device_preferred_on {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief The target device ID
    static constexpr int device = DeviceId;

    /// @brief Get the preferred memory location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::device;
    }

    /// @brief Get the fallback memory location
    [[nodiscard]] static constexpr memory_location fallback_location() noexcept {
        return memory_location::host;
    }

    /// @brief Check if memory is host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return false;
#else
        return true;
#endif
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return true;
#else
        return false;
#endif
    }

    /// @brief Check if fallback to host is allowed
    [[nodiscard]] static constexpr bool allows_fallback() noexcept {
        return true;
    }

    /// @brief Get device ID
    [[nodiscard]] static constexpr int device_id() noexcept {
        return DeviceId;
    }

    /// @brief Runtime check if currently using fallback
    [[nodiscard]] static bool using_fallback() noexcept {
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
        return false;
#else
        return true;
#endif
    }
};

}  // namespace dtl
