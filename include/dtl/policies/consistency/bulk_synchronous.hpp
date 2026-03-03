// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bulk_synchronous.hpp
/// @brief Bulk Synchronous Parallel (BSP) consistency policy
/// @details MVP default - consistency at explicit barriers (supersteps).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/consistency/consistency_policy.hpp>

namespace dtl {

/// @brief Bulk Synchronous Parallel consistency model
/// @details Provides consistency through explicit barrier synchronization.
///          Operations between barriers can execute in any order, but
///          all operations before a barrier complete before any operations
///          after the barrier begin.
///
/// @par BSP Superstep Model:
/// 1. Local computation phase (no communication visibility)
/// 2. Communication phase (sends/receives initiated)
/// 3. Barrier synchronization (all communication completes)
/// 4. All writes become visible to all ranks
///
/// @note This is the default and recommended consistency model for DTL.
///       It provides the best balance of simplicity and performance.
struct bulk_synchronous {
    /// @brief Policy category tag
    using policy_category = consistency_policy_tag;

    /// @brief Get the memory ordering for this policy
    [[nodiscard]] static constexpr memory_ordering ordering() noexcept {
        return memory_ordering::sequential;  // After barrier, all see same state
    }

    /// @brief Check if explicit barriers are required
    [[nodiscard]] static constexpr bool requires_barrier() noexcept {
        return true;
    }

    /// @brief Check if operations can overlap with communication
    [[nodiscard]] static constexpr bool allows_overlap() noexcept {
        return true;  // Within a superstep
    }

    /// @brief Check if point-to-point sync is sufficient
    [[nodiscard]] static constexpr bool needs_collective_sync() noexcept {
        return true;  // Requires collective barrier
    }

    /// @brief Get the default synchronization point type
    [[nodiscard]] static constexpr sync_point default_sync() noexcept {
        return sync_point::barrier;
    }

    /// @brief Check if writes are buffered until barrier
    [[nodiscard]] static constexpr bool buffers_writes() noexcept {
        return true;
    }

    /// @brief Synchronization point -- delegates to backend barrier
    /// @details In the BSP model, barrier() acts as a superstep boundary.
    ///          The actual barrier is executed by the backend communicator
    ///          when the algorithm invokes the collective operation.
    ///          This static method is a policy marker; real synchronization
    ///          is performed by the communicator's barrier() call within
    ///          distributed algorithm implementations.
    static void barrier() {
        // No-op: actual synchronization is performed by the backend
        // communicator when algorithms call comm.barrier().
    }
};

/// @brief Specialization of consistency_traits for bulk_synchronous
template <>
struct consistency_traits<bulk_synchronous> {
    static constexpr bool requires_barrier = true;
    static constexpr bool allows_overlap = true;
    static constexpr memory_ordering default_ordering = memory_ordering::sequential;
};

}  // namespace dtl
