// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file prefetch_policy.hpp
/// @brief Prefetch policy helpers for unified/managed memory
/// @details Provides prefetch policy enumerations and hint structures
///          for controlling data migration in unified memory systems.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <string_view>

namespace dtl {

// ============================================================================
// Prefetch Policy
// ============================================================================

/// @brief Policy for memory prefetching behavior
enum class prefetch_policy {
    /// @brief No prefetching
    none,

    /// @brief Prefetch data to device memory
    to_device,

    /// @brief Prefetch data to host memory
    to_host,

    /// @brief Prefetch data in both directions (bidirectional)
    bidirectional
};

// ============================================================================
// String Conversion
// ============================================================================

/// @brief Convert prefetch policy to string representation
/// @param policy The prefetch policy
/// @return String view of the policy name
[[nodiscard]] inline constexpr std::string_view to_string(prefetch_policy policy) noexcept {
    switch (policy) {
        case prefetch_policy::none:          return "none";
        case prefetch_policy::to_device:     return "to_device";
        case prefetch_policy::to_host:       return "to_host";
        case prefetch_policy::bidirectional: return "bidirectional";
        default:                             return "unknown";
    }
}

// ============================================================================
// Prefetch Hint
// ============================================================================

/// @brief Hint structure describing a prefetch operation
/// @details Contains all parameters needed to perform a prefetch operation
///          on unified/managed memory.
struct prefetch_hint {
    /// @brief The prefetch policy to apply
    prefetch_policy policy = prefetch_policy::none;

    /// @brief Target device ID for device prefetching
    int device_id = 0;

    /// @brief Byte offset within the allocation to start prefetching
    size_type offset = 0;

    /// @brief Number of bytes to prefetch (0 means entire allocation)
    size_type size = 0;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a prefetch hint for device prefetching
/// @param device_id Target device ID (default 0)
/// @return Prefetch hint configured for device prefetching
[[nodiscard]] inline constexpr prefetch_hint make_device_prefetch(int device_id = 0) noexcept {
    return prefetch_hint{
        .policy = prefetch_policy::to_device,
        .device_id = device_id,
        .offset = 0,
        .size = 0
    };
}

/// @brief Create a prefetch hint for host prefetching
/// @return Prefetch hint configured for host prefetching
[[nodiscard]] inline constexpr prefetch_hint make_host_prefetch() noexcept {
    return prefetch_hint{
        .policy = prefetch_policy::to_host,
        .device_id = 0,
        .offset = 0,
        .size = 0
    };
}

}  // namespace dtl
