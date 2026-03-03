// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file replicated.hpp
/// @brief Replicated partition policy
/// @details Full copy of data on each rank (no actual partitioning).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

namespace dtl {

/// @brief Replicated partition maintains full copy on each rank
/// @details All ranks have a complete copy of the data. Useful for
///          read-only shared data or small lookup tables.
///
/// @par Characteristics:
/// - All ranks own all elements
/// - Local size equals global size
/// - All accesses are local (no communication for reads)
/// - Updates require broadcast to maintain consistency
struct replicated {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief Determine which rank owns a global index
    /// @return Always returns the querying rank (all ranks own all data)
    /// @note For replicated data, ownership is shared; this returns my_rank
    ///       from the context, but since we don't have context here,
    ///       we return all_ranks to indicate shared ownership.
    [[nodiscard]] static constexpr rank_t owner(index_t /*global_idx*/,
                                                 size_type /*global_size*/,
                                                 rank_t /*num_ranks*/) noexcept {
        return all_ranks;  // Indicates all ranks have the data
    }

    /// @brief Check if element is local
    /// @return Always true for replicated data
    [[nodiscard]] static constexpr bool is_local(index_t /*global_idx*/,
                                                  size_type /*global_size*/,
                                                  rank_t /*num_ranks*/,
                                                  rank_t /*rank*/) noexcept {
        return true;  // Everything is local in replicated mode
    }

    /// @brief Calculate local size for a given rank
    /// @return Global size (full copy on each rank)
    [[nodiscard]] static constexpr size_type local_size(size_type global_size,
                                                        rank_t /*num_ranks*/,
                                                        rank_t /*rank*/) noexcept {
        return global_size;
    }

    /// @brief Convert global index to local index
    /// @return Same as global index (identity mapping)
    [[nodiscard]] static constexpr index_t to_local(index_t global_idx,
                                                     size_type /*global_size*/,
                                                     rank_t /*num_ranks*/,
                                                     rank_t /*rank*/) noexcept {
        return global_idx;
    }

    /// @brief Convert local index to global index
    /// @return Same as local index (identity mapping)
    [[nodiscard]] static constexpr index_t to_global(index_t local_idx,
                                                      size_type /*global_size*/,
                                                      rank_t /*num_ranks*/,
                                                      rank_t /*rank*/) noexcept {
        return local_idx;
    }

    /// @brief Create partition info for this policy
    [[nodiscard]] static constexpr partition_info make_info(size_type global_size,
                                                             rank_t num_ranks,
                                                             rank_t my_rank) noexcept {
        partition_info info{};
        info.global_size = global_size;
        info.num_ranks = num_ranks;
        info.my_rank = my_rank;
        info.local_size = global_size;  // Full copy
        info.local_start = 0;
        info.local_end = static_cast<index_t>(global_size);
        return info;
    }
};

}  // namespace dtl
