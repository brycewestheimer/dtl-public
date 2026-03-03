// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file relaxed.hpp
/// @brief Relaxed consistency policy
/// @details No ordering guarantees - maximum performance, minimum guarantees.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/consistency/consistency_policy.hpp>

namespace dtl {

/// @brief Relaxed consistency model
/// @details Provides no ordering guarantees between operations. Writes may
///          become visible in any order or not at all until explicit
///          synchronization.
///
/// @par Characteristics:
/// - No automatic synchronization
/// - Writes may be arbitrarily delayed
/// - Reads may return stale values
/// - Maximum performance, minimum guarantees
///
/// @warning Only use this policy when the algorithm does not require
///          any ordering guarantees, or when explicit synchronization
///          is managed externally.
///
/// @par Use Cases:
/// - Accumulation patterns (order doesn't matter)
/// - Read-mostly workloads
/// - Performance-critical code with external sync
struct relaxed {
    /// @brief Policy category tag
    using policy_category = consistency_policy_tag;

    /// @brief Get the memory ordering for this policy
    [[nodiscard]] static constexpr memory_ordering ordering() noexcept {
        return memory_ordering::relaxed;
    }

    /// @brief Check if explicit barriers are required
    [[nodiscard]] static constexpr bool requires_barrier() noexcept {
        return false;  // No synchronization required
    }

    /// @brief Check if operations can overlap with communication
    [[nodiscard]] static constexpr bool allows_overlap() noexcept {
        return true;  // Everything can overlap
    }

    /// @brief Check if point-to-point sync is sufficient
    [[nodiscard]] static constexpr bool needs_collective_sync() noexcept {
        return false;  // No sync needed by default
    }

    /// @brief Get the default synchronization point type
    [[nodiscard]] static constexpr sync_point default_sync() noexcept {
        return sync_point::none;
    }

    /// @brief Check if reads can return stale values
    [[nodiscard]] static constexpr bool allows_stale_reads() noexcept {
        return true;
    }

    /// @brief Check if writes can be reordered
    [[nodiscard]] static constexpr bool allows_reordering() noexcept {
        return true;
    }

    /// @brief Check if writes can be coalesced
    [[nodiscard]] static constexpr bool allows_coalescing() noexcept {
        return true;
    }

    /// @brief Optional explicit synchronization (user-controlled)
    /// @details Under relaxed consistency, no ordering guarantees are
    ///          provided by the runtime. Writes may become visible to
    ///          remote ranks in any order, and reads may return stale
    ///          values at any time. This sync() method is provided as
    ///          an opt-in escape hatch for user-managed synchronization.
    ///          Calling it has no effect; the user is responsible for
    ///          coordinating visibility through external mechanisms
    ///          (e.g., explicit barriers or application-level protocols).
    static void sync() {
        // No-op: relaxed consistency provides no ordering guarantees.
        // Users must manage synchronization externally if needed.
    }
};

/// @brief Specialization of consistency_traits for relaxed
template <>
struct consistency_traits<relaxed> {
    static constexpr bool requires_barrier = false;
    static constexpr bool allows_overlap = true;
    static constexpr memory_ordering default_ordering = memory_ordering::relaxed;
};

}  // namespace dtl
