// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file host_only.hpp
/// @brief Host-only (CPU) memory placement policy
/// @details Memory is allocated exclusively on the host (CPU).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

namespace dtl {

/// @brief Host-only placement allocates memory on CPU
/// @details This is the default placement policy. All memory is allocated
///          in host (CPU) memory and is not directly accessible from
///          GPU devices.
struct host_only {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief Get the preferred memory location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::host;
    }

    /// @brief Check if memory is host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return true;
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return false;
    }

    /// @brief Check if data needs to be copied to device for GPU operations
    [[nodiscard]] static constexpr bool requires_device_copy() noexcept {
        return true;
    }

    /// @brief Get device ID (not applicable for host-only)
    [[nodiscard]] static constexpr int device_id() noexcept {
        return -1;
    }
};

}  // namespace dtl
