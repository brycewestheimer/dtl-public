// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file partition_map.hpp
/// @brief Partition map for index translation with a specific partition policy
/// @details Provides a unified interface for global/local index conversion.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/policies/partition/partition_policy.hpp>
#include <dtl/policies/partition/block_partition.hpp>
#include <dtl/policies/partition/cyclic_partition.hpp>

namespace dtl {

/// @brief Maps between global and local indices using a partition policy
/// @tparam PartitionPolicy The partition policy to use
/// @details Encapsulates all index translation operations for a given
///          distribution configuration. Caches the partition info for
///          efficient repeated queries.
///
/// @par Design:
/// - All operations are constexpr where possible
/// - No communication occurs through this class
/// - Thread-safe for concurrent reads (immutable after construction)
///
/// @par Example:
/// @code
/// using Policy = block_partition<>;
/// partition_map<Policy> map(1000, 4, 1);  // 1000 elements, 4 ranks, I'm rank 1
///
/// // Query ownership
/// rank_t owner = map.owner(500);
/// bool local = map.is_local(500);
///
/// // Convert indices
/// index_t local_idx = map.to_local(500);  // Only valid if is_local(500)
/// index_t global_idx = map.to_global(0);  // My first element's global index
/// @endcode
///
/// @par Invariants:
/// - owner(g) returns valid rank in [0, num_ranks)
/// - to_local(g) succeeds only when owner(g) == this_rank
/// - local_sizes sum to global_size
/// - Roundtrip: to_global(to_local(g)) == g for local indices
template <typename PartitionPolicy>
class partition_map {
public:
    using policy_type = PartitionPolicy;

    /// @brief Construct a partition map
    /// @param global_size Total number of elements across all ranks
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    constexpr partition_map(size_type global_size, rank_t num_ranks, rank_t my_rank) noexcept
        : info_{PartitionPolicy::make_info(global_size, num_ranks, my_rank)} {}

    /// @brief Construct from pre-computed partition info
    /// @param info The partition info structure
    constexpr explicit partition_map(const partition_info& info) noexcept
        : info_{info} {}

    // =========================================================================
    // Size Queries
    // =========================================================================

    /// @brief Get total global size
    [[nodiscard]] constexpr size_type global_size() const noexcept {
        return info_.global_size;
    }

    /// @brief Get local size for this rank
    [[nodiscard]] constexpr size_type local_size() const noexcept {
        return info_.local_size;
    }

    /// @brief Get local size for any rank
    /// @param rank The rank to query
    [[nodiscard]] constexpr size_type local_size(rank_t rank) const noexcept {
        return PartitionPolicy::local_size(info_.global_size, info_.num_ranks, rank);
    }

    /// @brief Get number of ranks
    [[nodiscard]] constexpr rank_t num_ranks() const noexcept {
        return info_.num_ranks;
    }

    /// @brief Get this rank's ID
    [[nodiscard]] constexpr rank_t my_rank() const noexcept {
        return info_.my_rank;
    }

    // =========================================================================
    // Ownership Queries
    // =========================================================================

    /// @brief Get the owner of a global index
    /// @param global_idx The global index to query
    /// @return The rank that owns this index
    [[nodiscard]] constexpr rank_t owner(index_t global_idx) const noexcept {
        return PartitionPolicy::owner(global_idx, info_.global_size, info_.num_ranks);
    }

    /// @brief Check if a global index is local to this rank
    /// @param global_idx The global index to check
    /// @return true if the element is on this rank
    /// @note For replicated partitions, owner() returns all_ranks (-2) and is_local is always true
    [[nodiscard]] constexpr bool is_local(index_t global_idx) const noexcept {
        rank_t owner_rank = owner(global_idx);
        return owner_rank == info_.my_rank || owner_rank == all_ranks;
    }

    // =========================================================================
    // Index Translation
    // =========================================================================

    /// @brief Convert global index to local index
    /// @param global_idx The global index to convert
    /// @return The local index on this rank
    /// @pre is_local(global_idx) must be true
    /// @note Returns unspecified value if global_idx is not local
    [[nodiscard]] constexpr index_t to_local(index_t global_idx) const noexcept {
        return PartitionPolicy::to_local(global_idx, info_.global_size,
                                         info_.num_ranks, info_.my_rank);
    }

    /// @brief Convert local index to global index
    /// @param local_idx The local index to convert
    /// @return The corresponding global index
    /// @pre local_idx must be in [0, local_size())
    [[nodiscard]] constexpr index_t to_global(index_t local_idx) const noexcept {
        return PartitionPolicy::to_global(local_idx, info_.global_size,
                                          info_.num_ranks, info_.my_rank);
    }

    /// @brief Get the global offset (start index) of this rank's data
    [[nodiscard]] constexpr index_t local_offset() const noexcept {
        return info_.local_start;
    }

    /// @brief Get the global offset for a specific rank
    /// @param rank The rank to query
    [[nodiscard]] constexpr index_t local_offset(rank_t rank) const noexcept {
        // Use the policy's calculation for other ranks
        return PartitionPolicy::local_start(info_.global_size, info_.num_ranks, rank);
    }

    // =========================================================================
    // Range Queries
    // =========================================================================

    /// @brief Get the global start index of this rank's partition
    [[nodiscard]] constexpr index_t local_start() const noexcept {
        return info_.local_start;
    }

    /// @brief Get the global end index (exclusive) of this rank's partition
    [[nodiscard]] constexpr index_t local_end() const noexcept {
        return info_.local_end;
    }

    /// @brief Check if local partition is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return info_.local_size == 0;
    }

    // =========================================================================
    // Partition Info Access
    // =========================================================================

    /// @brief Get the underlying partition info
    [[nodiscard]] constexpr const partition_info& info() const noexcept {
        return info_;
    }

private:
    partition_info info_;
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Create a partition map with block partitioning
/// @param global_size Total number of elements
/// @param num_ranks Number of ranks
/// @param my_rank This rank's ID
/// @return partition_map<block_partition<>>
template <size_type N = 0>
[[nodiscard]] constexpr auto make_block_partition_map(
    size_type global_size, rank_t num_ranks, rank_t my_rank) {
    return partition_map<block_partition<N>>(global_size, num_ranks, my_rank);
}

/// @brief Create a partition map with cyclic partitioning
/// @param global_size Total number of elements
/// @param num_ranks Number of ranks
/// @param my_rank This rank's ID
/// @return partition_map<cyclic_partition<>>
template <size_type N = 0>
[[nodiscard]] constexpr auto make_cyclic_partition_map(
    size_type global_size, rank_t num_ranks, rank_t my_rank) {
    return partition_map<cyclic_partition<N>>(global_size, num_ranks, my_rank);
}

// =============================================================================
// Deduction Guide
// =============================================================================

// Deduction from partition_info (defaults to block_partition)
partition_map(const partition_info&) -> partition_map<block_partition<>>;

}  // namespace dtl
