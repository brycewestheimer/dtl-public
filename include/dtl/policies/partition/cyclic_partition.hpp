// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cyclic_partition.hpp
/// @brief Cyclic (round-robin) partition policy
/// @details Distributes data in a round-robin fashion across ranks.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

namespace dtl {

/// @brief Cyclic partition distributes elements round-robin across ranks
/// @tparam N Cycle size (0 = single element per cycle)
/// @details Cyclic partitioning assigns element i to rank (i % num_ranks).
///          This provides good load balancing for irregular access patterns.
///
/// @par Example (10 elements, 3 ranks):
/// @code
///   Rank 0: elements 0, 3, 6, 9
///   Rank 1: elements 1, 4, 7
///   Rank 2: elements 2, 5, 8
/// @endcode
template <size_type N>
struct cyclic_partition {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief Cycle size (elements per rank before cycling)
    static constexpr size_type cycle_size = N == 0 ? 1 : N;

    /// @brief Determine which rank owns a global index
    /// @param global_idx The global index to query
    /// @param global_size Total size of the distributed range (unused)
    /// @param num_ranks Number of ranks in the communicator
    /// @return The owning rank
    [[nodiscard]] static constexpr rank_t owner(index_t global_idx,
                                                 [[maybe_unused]] size_type global_size,
                                                 rank_t num_ranks) noexcept {
        if (num_ranks <= 0) {
            return 0;
        }
        if constexpr (cycle_size == 1) {
            return static_cast<rank_t>(global_idx % static_cast<index_t>(num_ranks));
        } else {
            const auto cycle = global_idx / static_cast<index_t>(cycle_size);
            return static_cast<rank_t>(cycle % static_cast<index_t>(num_ranks));
        }
    }

    /// @brief Calculate local size for a given rank
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank to query
    /// @return Number of elements on the given rank
    [[nodiscard]] static constexpr size_type local_size(size_type global_size,
                                                        rank_t num_ranks,
                                                        rank_t rank) noexcept {
        if (global_size == 0 || num_ranks <= 0) {
            return 0;
        }

        if constexpr (cycle_size == 1) {
            const size_type base = global_size / static_cast<size_type>(num_ranks);
            const size_type remainder = global_size % static_cast<size_type>(num_ranks);
            return base + (static_cast<size_type>(rank) < remainder ? 1 : 0);
        } else {
            // Count complete cycles and partial cycle
            const size_type total_cycles = global_size / cycle_size;
            const size_type partial = global_size % cycle_size;

            const size_type full_cycles_per_rank = total_cycles / static_cast<size_type>(num_ranks);
            const size_type extra_cycles = total_cycles % static_cast<size_type>(num_ranks);

            size_type count = full_cycles_per_rank * cycle_size;
            if (static_cast<size_type>(rank) < extra_cycles) {
                count += cycle_size;
            } else if (static_cast<size_type>(rank) == extra_cycles) {
                count += partial;
            }
            return count;
        }
    }

    /// @brief Convert global index to local index
    /// @param global_idx The global index
    /// @param global_size Total size (unused)
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank to convert for
    /// @return Local index on the given rank
    [[nodiscard]] static constexpr index_t to_local(index_t global_idx,
                                                     [[maybe_unused]] size_type global_size,
                                                     rank_t num_ranks,
                                                     [[maybe_unused]] rank_t rank) noexcept {
        if constexpr (cycle_size == 1) {
            return global_idx / static_cast<index_t>(num_ranks);
        } else {
            const auto n = static_cast<index_t>(num_ranks);
            const auto c = static_cast<index_t>(cycle_size);
            const auto cycle = global_idx / c;
            const auto offset_in_cycle = global_idx % c;
            const auto local_cycle = cycle / n;
            return local_cycle * c + offset_in_cycle;
        }
    }

    /// @brief Convert local index to global index
    /// @param local_idx The local index
    /// @param global_size Total size (unused)
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank the local index is on
    /// @return Corresponding global index
    [[nodiscard]] static constexpr index_t to_global(index_t local_idx,
                                                      [[maybe_unused]] size_type global_size,
                                                      rank_t num_ranks,
                                                      rank_t rank) noexcept {
        if constexpr (cycle_size == 1) {
            return local_idx * static_cast<index_t>(num_ranks) + static_cast<index_t>(rank);
        } else {
            const auto n = static_cast<index_t>(num_ranks);
            const auto c = static_cast<index_t>(cycle_size);
            const auto local_cycle = local_idx / c;
            const auto offset_in_cycle = local_idx % c;
            const auto global_cycle = local_cycle * n + static_cast<index_t>(rank);
            return global_cycle * c + offset_in_cycle;
        }
    }

    /// @brief Create partition info for this policy
    [[nodiscard]] static constexpr partition_info make_info(size_type global_size,
                                                             rank_t num_ranks,
                                                             rank_t my_rank) noexcept {
        partition_info info{};
        info.global_size = global_size;
        info.num_ranks = num_ranks;
        info.my_rank = my_rank;
        info.local_size = local_size(global_size, num_ranks, my_rank);
        // For cyclic, local_start/end don't have traditional meaning
        info.local_start = 0;
        info.local_end = static_cast<index_t>(info.local_size);
        return info;
    }
};

}  // namespace dtl
