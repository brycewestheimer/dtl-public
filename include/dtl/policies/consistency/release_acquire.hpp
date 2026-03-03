// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file release_acquire.hpp
/// @brief Release-acquire consistency policy
/// @details Synchronization on explicit release/acquire operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/consistency/consistency_policy.hpp>

namespace dtl {

/// @brief Release-acquire consistency model
/// @details Provides consistency through paired release/acquire operations.
///          A release operation makes all prior writes visible to any
///          subsequent acquire operation on the same synchronization variable.
///
/// @par Semantics:
/// - Release: All prior writes become visible
/// - Acquire: See all writes before the paired release
/// - No global ordering (more efficient than sequential)
///
/// @par Use Cases:
/// - Producer-consumer patterns
/// - Lock-based synchronization
/// - Efficient point-to-point synchronization
struct release_acquire {
    /// @brief Policy category tag
    using policy_category = consistency_policy_tag;

    /// @brief Get the memory ordering for this policy
    [[nodiscard]] static constexpr memory_ordering ordering() noexcept {
        return memory_ordering::acquire_release;
    }

    /// @brief Check if explicit barriers are required
    [[nodiscard]] static constexpr bool requires_barrier() noexcept {
        return false;  // Uses release/acquire pairs instead
    }

    /// @brief Check if operations can overlap with communication
    [[nodiscard]] static constexpr bool allows_overlap() noexcept {
        return true;
    }

    /// @brief Check if point-to-point sync is sufficient
    [[nodiscard]] static constexpr bool needs_collective_sync() noexcept {
        return false;  // Point-to-point is sufficient
    }

    /// @brief Get the default synchronization point type
    [[nodiscard]] static constexpr sync_point default_sync() noexcept {
        return sync_point::fence;
    }

    /// @brief Check if this policy supports epoch-based access
    [[nodiscard]] static constexpr bool supports_epochs() noexcept {
        return true;
    }

    /// @brief Perform release operation (make prior writes visible)
    /// @details Issues a release fence so that all memory writes performed
    ///          before this call are guaranteed to be visible to any
    ///          subsequent acquire operation on the same synchronization
    ///          variable. In a distributed context, this ensures that
    ///          outgoing messages or RMA puts carry a consistent view of
    ///          the sender's memory. The actual fence is issued by the
    ///          backend (e.g., MPI_Win_flush or std::atomic_thread_fence).
    static void release() {
        // No-op: actual release fence is issued by the backend when
        // the communicator or RMA window completes the release epoch.
    }

    /// @brief Perform acquire operation (observe prior writes)
    /// @details Issues an acquire fence so that all memory reads after
    ///          this call observe the effects of a prior paired release
    ///          operation. In a distributed context, this ensures that
    ///          incoming messages or RMA gets see a consistent view of
    ///          the remote rank's memory. The actual fence is issued by
    ///          the backend (e.g., MPI_Win_lock or std::atomic_thread_fence).
    static void acquire() {
        // No-op: actual acquire fence is issued by the backend when
        // the communicator or RMA window begins the acquire epoch.
    }

    /// @brief Perform combined release-acquire fence
    /// @details Issues a full memory fence that combines both release and
    ///          acquire semantics. All prior writes become visible and all
    ///          subsequent reads observe prior remote writes. This is
    ///          stronger than either release or acquire alone but weaker
    ///          than sequential consistency. Useful for bidirectional
    ///          synchronization such as lock/unlock patterns.
    static void fence() {
        // No-op: actual fence is issued by the backend when the
        // communicator performs a full memory synchronization.
    }
};

/// @brief Specialization of consistency_traits for release_acquire
template <>
struct consistency_traits<release_acquire> {
    static constexpr bool requires_barrier = false;
    static constexpr bool allows_overlap = true;
    static constexpr memory_ordering default_ordering = memory_ordering::acquire_release;
};

}  // namespace dtl
