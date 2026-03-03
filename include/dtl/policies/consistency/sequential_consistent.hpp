// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file sequential_consistent.hpp
/// @brief Sequential consistency policy
/// @details All operations globally ordered - strongest consistency guarantee.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/consistency/consistency_policy.hpp>

namespace dtl {

/// @brief Sequential consistency model
/// @details Provides the strongest consistency guarantee. All operations
///          appear to execute in some sequential order that is consistent
///          with the program order of each rank.
///
/// @par Guarantees:
/// - All ranks observe the same order of all operations
/// - Writes are immediately visible to all ranks
/// - No reordering of operations across ranks
///
/// @warning This policy has significant performance overhead due to
///          required synchronization after every operation.
struct sequential_consistent {
    /// @brief Policy category tag
    using policy_category = consistency_policy_tag;

    /// @brief Get the memory ordering for this policy
    [[nodiscard]] static constexpr memory_ordering ordering() noexcept {
        return memory_ordering::sequential;
    }

    /// @brief Check if explicit barriers are required
    [[nodiscard]] static constexpr bool requires_barrier() noexcept {
        return true;  // BSP semantics require barriers for global ordering
    }

    /// @brief Check if operations can overlap with communication
    [[nodiscard]] static constexpr bool allows_overlap() noexcept {
        return false;  // Must serialize all operations
    }

    /// @brief Check if point-to-point sync is sufficient
    [[nodiscard]] static constexpr bool needs_collective_sync() noexcept {
        return true;  // Requires global ordering
    }

    /// @brief Get the default synchronization point type
    [[nodiscard]] static constexpr sync_point default_sync() noexcept {
        return sync_point::barrier;
    }

    /// @brief Check if writes are immediately visible
    [[nodiscard]] static constexpr bool immediate_visibility() noexcept {
        return true;
    }

    /// @brief Check if reads can return stale values
    [[nodiscard]] static constexpr bool allows_stale_reads() noexcept {
        return false;
    }
};

/// @brief Specialization of consistency_traits for sequential_consistent
template <>
struct consistency_traits<sequential_consistent> {
    static constexpr bool requires_barrier = true;  // BSP semantics require barriers
    static constexpr bool allows_overlap = false;
    static constexpr memory_ordering default_ordering = memory_ordering::sequential;
};

}  // namespace dtl
