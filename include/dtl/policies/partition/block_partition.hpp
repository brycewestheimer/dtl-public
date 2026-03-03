// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file block_partition.hpp
/// @brief Block (contiguous chunk) partition policy
/// @details Distributes data as contiguous chunks, giving each rank
///          approximately equal sized portions.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

namespace dtl {

/// @brief Block partition distributes contiguous chunks to ranks
/// @tparam N Number of dimensions for ND partitioning (0 = 1D default)
/// @details Block partitioning divides data into N contiguous chunks,
///          where N is the number of ranks. Each rank gets elements
///          [start, end) where start = rank * chunk_size and
///          end = start + local_size.
///
/// @par Example (10 elements, 3 ranks):
/// @code
///   Rank 0: elements [0, 3)  -> indices 0, 1, 2, 3
///   Rank 1: elements [4, 6)  -> indices 4, 5, 6
///   Rank 2: elements [7, 9)  -> indices 7, 8, 9
/// @endcode
template <size_type N>
struct block_partition {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief Dimensionality of the partition (0 = 1D)
    static constexpr size_type dimensions = N == 0 ? 1 : N;

    /// @brief Determine which rank owns a global index
    /// @param global_idx The global index to query
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @return The owning rank
    [[nodiscard]] static constexpr rank_t owner(index_t global_idx,
                                                 size_type global_size,
                                                 rank_t num_ranks) noexcept {
        if (global_size == 0 || num_ranks <= 0) {
            return 0;
        }
        const size_type chunk_size = global_size / static_cast<size_type>(num_ranks);
        const size_type remainder = global_size % static_cast<size_type>(num_ranks);

        // First 'remainder' ranks get chunk_size + 1 elements
        const auto idx = static_cast<size_type>(global_idx);
        const size_type boundary = remainder * (chunk_size + 1);

        if (idx < boundary) {
            return static_cast<rank_t>(idx / (chunk_size + 1));
        } else {
            return static_cast<rank_t>(remainder + (idx - boundary) / chunk_size);
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
        const size_type chunk_size = global_size / static_cast<size_type>(num_ranks);
        const size_type remainder = global_size % static_cast<size_type>(num_ranks);

        // First 'remainder' ranks get one extra element
        return chunk_size + (static_cast<size_type>(rank) < remainder ? 1 : 0);
    }

    /// @brief Get the global index of the first element on a rank
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank to query
    /// @return Global index of first element on rank
    [[nodiscard]] static constexpr index_t local_start(size_type global_size,
                                                        rank_t num_ranks,
                                                        rank_t rank) noexcept {
        if (global_size == 0 || num_ranks <= 0 || rank < 0) {
            return 0;
        }
        const size_type chunk_size = global_size / static_cast<size_type>(num_ranks);
        const size_type remainder = global_size % static_cast<size_type>(num_ranks);
        const auto r = static_cast<size_type>(rank);

        if (r < remainder) {
            return static_cast<index_t>(r * (chunk_size + 1));
        } else {
            return static_cast<index_t>(remainder * (chunk_size + 1) + (r - remainder) * chunk_size);
        }
    }

    /// @brief Convert global index to local index
    /// @param global_idx The global index
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank to convert for
    /// @return Local index on the given rank
    [[nodiscard]] static constexpr index_t to_local(index_t global_idx,
                                                     size_type global_size,
                                                     rank_t num_ranks,
                                                     rank_t rank) noexcept {
        return global_idx - local_start(global_size, num_ranks, rank);
    }

    /// @brief Convert local index to global index
    /// @param local_idx The local index
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @param rank The rank the local index is on
    /// @return Corresponding global index
    [[nodiscard]] static constexpr index_t to_global(index_t local_idx,
                                                      size_type global_size,
                                                      rank_t num_ranks,
                                                      rank_t rank) noexcept {
        return local_idx + local_start(global_size, num_ranks, rank);
    }

    /// @brief Create partition info for this policy
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @return Partition info structure
    [[nodiscard]] static constexpr partition_info make_info(size_type global_size,
                                                             rank_t num_ranks,
                                                             rank_t my_rank) noexcept {
        partition_info info{};
        info.global_size = global_size;
        info.num_ranks = num_ranks;
        info.my_rank = my_rank;
        info.local_size = local_size(global_size, num_ranks, my_rank);
        info.local_start = local_start(global_size, num_ranks, my_rank);
        info.local_end = info.local_start + static_cast<index_t>(info.local_size);
        return info;
    }
};

}  // namespace dtl
