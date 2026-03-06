// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file consistency_policy.hpp
/// @brief Base consistency policy concept and interface
/// @details Defines the consistency model for distributed operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/core/fwd.hpp>

namespace dtl {

// Note: consistency_policy_tag is defined in dtl/core/traits.hpp
// Note: is_consistency_policy_v is defined in dtl/core/traits.hpp
// Note: ConsistencyPolicy concept is defined in dtl/core/concepts.hpp

/// @brief Synchronization points available in the system
enum class sync_point {
    none,       ///< No synchronization
    barrier,    ///< Full barrier synchronization
    fence,      ///< Memory fence only
    epoch       ///< Epoch-based synchronization
};

/// @brief Memory ordering for distributed operations
enum class memory_ordering {
    relaxed,            ///< No ordering guarantees
    acquire,            ///< Acquire semantics
    release,            ///< Release semantics
    acquire_release,    ///< Both acquire and release
    sequential          ///< Sequential consistency
};

/// @brief Traits for consistency policy inspection
template <typename Policy>
struct consistency_traits {
    /// @brief Check if policy requires explicit barriers
    static constexpr bool requires_barrier = false;

    /// @brief Check if policy allows overlap of computation and communication
    static constexpr bool allows_overlap = true;

    /// @brief Get the default memory ordering
    static constexpr memory_ordering default_ordering = memory_ordering::relaxed;
};

// ============================================================================
// Consistency Guarantee Matrix
// ============================================================================
// The following traits document the precise guarantees and trade-offs
// for each consistency policy. Users selecting a weaker consistency model
// should consult these traits to understand what invariants they lose.

/// @brief Detailed guarantees provided by a consistency policy
/// @details Use this to query exactly what a consistency policy guarantees.
///          This is especially important when choosing relaxed or
///          release_acquire policies, where some invariants are deliberately
///          traded for performance.
///
/// @par Guarantee dimensions:
/// - writes_visible_after_barrier:  After barrier(), all prior writes are
///                                  visible to all ranks
/// - writes_visible_after_fence:    After fence(), all prior writes are
///                                  visible to paired acquire operations
/// - reads_see_latest:              Reads always return the most recent
///                                  value written by any rank
/// - no_write_reordering:           Writes from a single rank are observed
///                                  in program order by all other ranks
/// - total_order:                   All ranks observe the same total order
///                                  of all operations
/// - safe_for_concurrent_mutation:  Multiple ranks can safely write to
///                                  overlapping regions without UB
///
/// @par Example:
/// @code
/// using traits = consistency_guarantees<relaxed>;
/// static_assert(!traits::reads_see_latest,
///     "relaxed consistency may return stale reads");
/// @endcode
template <typename Policy>
struct consistency_guarantees {
    static constexpr bool writes_visible_after_barrier = false;
    static constexpr bool writes_visible_after_fence = false;
    static constexpr bool reads_see_latest = false;
    static constexpr bool no_write_reordering = false;
    static constexpr bool total_order = false;
    static constexpr bool safe_for_concurrent_mutation = false;
};

/// @brief Guarantees for bulk_synchronous policy
/// @details BSP provides strong guarantees after barriers: all writes
///          become visible, and operations within a superstep can execute
///          freely. This is the safest and recommended default.
template <>
struct consistency_guarantees<bulk_synchronous> {
    static constexpr bool writes_visible_after_barrier = true;
    static constexpr bool writes_visible_after_fence = false;
    static constexpr bool reads_see_latest = false;  // Only after barrier
    static constexpr bool no_write_reordering = true; // Within superstep
    static constexpr bool total_order = false;        // Not between barriers
    static constexpr bool safe_for_concurrent_mutation = true; // Via barriers
};

/// @brief Guarantees for sequential_consistent policy
/// @details Sequential consistency provides the strongest guarantees:
///          all operations appear in a single global order. This comes
///          at significant performance cost due to required synchronization
///          after every operation.
template <>
struct consistency_guarantees<sequential_consistent> {
    static constexpr bool writes_visible_after_barrier = true;
    static constexpr bool writes_visible_after_fence = true;
    static constexpr bool reads_see_latest = true;
    static constexpr bool no_write_reordering = true;
    static constexpr bool total_order = true;
    static constexpr bool safe_for_concurrent_mutation = true;
};

/// @brief Guarantees for release_acquire policy
/// @details Release-acquire provides consistency through paired operations.
///          A release makes prior writes visible to a subsequent acquire.
///          No global ordering is provided — only pairwise synchronization.
///
/// @warning Without proper release/acquire pairing, reads may return
///          stale values. Only use when you understand the synchronization
///          pattern of your algorithm.
template <>
struct consistency_guarantees<release_acquire> {
    static constexpr bool writes_visible_after_barrier = true;
    static constexpr bool writes_visible_after_fence = true;
    static constexpr bool reads_see_latest = false;   // Only after acquire
    static constexpr bool no_write_reordering = true;  // Per release/acquire pair
    static constexpr bool total_order = false;
    static constexpr bool safe_for_concurrent_mutation = false; // Must use pairs
};

/// @brief Guarantees for relaxed policy
/// @details Relaxed consistency provides NO ordering guarantees.
///          Writes may become visible in any order or not at all until
///          explicit synchronization. Reads may return stale values.
///
/// @warning This policy trades all safety guarantees for maximum
///          performance. Only use when:
///          - Operations are commutative (e.g., accumulation)
///          - Data is read-only or partitioned without overlap
///          - External synchronization is managed by the application
template <>
struct consistency_guarantees<relaxed> {
    static constexpr bool writes_visible_after_barrier = false;
    static constexpr bool writes_visible_after_fence = false;
    static constexpr bool reads_see_latest = false;
    static constexpr bool no_write_reordering = false;
    static constexpr bool total_order = false;
    static constexpr bool safe_for_concurrent_mutation = false;
};

}  // namespace dtl
