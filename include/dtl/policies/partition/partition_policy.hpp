// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file partition_policy.hpp
/// @brief Base partition policy concept and interface
/// @details Defines the partition policy tag and requirements for
///          data distribution across ranks.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

namespace dtl {

// =============================================================================
// Partition Policy Tag
// =============================================================================

// partition_policy_tag is defined in core/traits.hpp

// =============================================================================
// Partition Policy Concept Requirements
// =============================================================================

/// @brief Concept defining requirements for partition policies
/// @details A partition policy must provide:
///          - policy_category type alias = partition_policy_tag
///          - owner(global_idx, global_size, num_ranks) -> rank_t
///          - local_size(global_size, num_ranks, rank) -> size_type
///          - to_local(global_idx, global_size, num_ranks, rank) -> index_t
///          - to_global(local_idx, global_size, num_ranks, rank) -> index_t
template <typename P>
concept PartitionPolicyConcept =
    std::same_as<typename P::policy_category, partition_policy_tag> &&
    requires(const P& p, index_t global_idx, index_t local_idx, size_type global_size, rank_t num_ranks, rank_t rank) {
        { p.owner(global_idx, global_size, num_ranks) } -> std::convertible_to<rank_t>;
        { p.local_size(global_size, num_ranks, rank) } -> std::convertible_to<size_type>;
        { p.to_local(global_idx, global_size, num_ranks, rank) } -> std::convertible_to<index_t>;
        { p.to_global(local_idx, global_size, num_ranks, rank) } -> std::convertible_to<index_t>;
    };

// =============================================================================
// Partition Info Structure
// =============================================================================

/// @brief Runtime partition information for a given configuration
/// @details Cached partition state to avoid repeated calculations.
struct partition_info {
    size_type global_size;      ///< Total number of elements
    rank_t num_ranks;           ///< Number of ranks
    rank_t my_rank;             ///< This rank's ID
    size_type local_size;       ///< Number of local elements
    index_t local_start;        ///< Global index of first local element
    index_t local_end;          ///< Global index past last local element

    /// @brief Check if a global index is local to this rank
    [[nodiscard]] constexpr bool is_local(index_t global_idx) const noexcept {
        return global_idx >= local_start && global_idx < local_end;
    }

    /// @brief Convert global index to local (unchecked)
    [[nodiscard]] constexpr index_t to_local_unchecked(index_t global_idx) const noexcept {
        return global_idx - local_start;
    }

    /// @brief Convert local index to global
    [[nodiscard]] constexpr index_t to_global(index_t local_idx) const noexcept {
        return local_idx + local_start;
    }
};

// =============================================================================
// Default Partition Policy
// =============================================================================

/// @brief Type alias for the default partition policy
/// @details block_partition<0> is used as the default, providing
///          contiguous chunk distribution.
template <size_type N>
struct block_partition;  // Forward declaration (default in fwd.hpp)

/// @brief Default partition policy is 1D block partition
using default_partition_policy = block_partition<0>;

}  // namespace dtl
