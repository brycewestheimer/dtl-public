// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file dynamic_block.hpp
/// @brief Runtime-configurable block partition
/// @details Block partition with runtime-specified parameters.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>
#include <dtl/policies/partition/block_partition.hpp>

#include <algorithm>
#include <vector>

namespace dtl {

/// @brief Dynamic block partition with runtime-configurable distribution
/// @details Unlike block_partition<N> which has compile-time dimension count,
///          dynamic_block allows runtime specification of partition boundaries.
///          Useful when partition sizes need to be determined at runtime
///          (e.g., based on load balancing analysis).
struct dynamic_block {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief Construct with uniform distribution (delegates to block_partition)
    dynamic_block() = default;

    /// @brief Construct with explicit partition boundaries
    /// @param boundaries Global indices where each rank's partition starts
    ///        (boundaries[rank] = first element of rank)
    explicit dynamic_block(std::vector<index_t> boundaries)
        : boundaries_{std::move(boundaries)} {}

    /// @brief Construct with explicit local sizes per rank
    /// @param local_sizes Number of elements on each rank
    static dynamic_block from_sizes(const std::vector<size_type>& local_sizes) {
        std::vector<index_t> boundaries;
        boundaries.reserve(local_sizes.size() + 1);
        index_t offset = 0;
        for (size_type size : local_sizes) {
            boundaries.push_back(offset);
            offset += static_cast<index_t>(size);
        }
        boundaries.push_back(offset);  // End boundary
        return dynamic_block{std::move(boundaries)};
    }

    /// @brief Check if using explicit boundaries
    [[nodiscard]] bool has_explicit_boundaries() const noexcept {
        return !boundaries_.empty();
    }

    /// @brief Determine which rank owns a global index
    [[nodiscard]] rank_t owner(index_t global_idx,
                               size_type global_size,
                               rank_t num_ranks) const noexcept {
        if (boundaries_.empty()) {
            // Fall back to uniform block partition
            return block_partition<0>::owner(global_idx, global_size, num_ranks);
        }

        // Binary search in boundaries (O(log n))
        auto it = std::upper_bound(boundaries_.begin(), boundaries_.end(), global_idx);
        if (it == boundaries_.begin()) {
            return 0;
        }
        auto r = static_cast<rank_t>(std::distance(boundaries_.begin(), it) - 1);
        if (r >= static_cast<rank_t>(boundaries_.size()) - 1) {
            return num_ranks - 1;
        }
        return r;
    }

    /// @brief Calculate local size for a given rank
    [[nodiscard]] size_type local_size(size_type global_size,
                                       rank_t num_ranks,
                                       rank_t rank) const noexcept {
        if (boundaries_.empty()) {
            return block_partition<0>::local_size(global_size, num_ranks, rank);
        }

        const auto r = static_cast<size_type>(rank);
        if (r + 1 >= boundaries_.size()) {
            return 0;
        }
        return static_cast<size_type>(boundaries_[r + 1] - boundaries_[r]);
    }

    /// @brief Get the starting global index for a rank
    [[nodiscard]] index_t local_start(size_type global_size,
                                      rank_t num_ranks,
                                      rank_t rank) const noexcept {
        if (boundaries_.empty()) {
            return block_partition<0>::local_start(global_size, num_ranks, rank);
        }
        return boundaries_[static_cast<size_type>(rank)];
    }

    /// @brief Convert global index to local index
    [[nodiscard]] index_t to_local(index_t global_idx,
                                   size_type global_size,
                                   rank_t num_ranks,
                                   rank_t rank) const noexcept {
        return global_idx - local_start(global_size, num_ranks, rank);
    }

    /// @brief Convert local index to global index
    [[nodiscard]] index_t to_global(index_t local_idx,
                                    size_type global_size,
                                    rank_t num_ranks,
                                    rank_t rank) const noexcept {
        return local_idx + local_start(global_size, num_ranks, rank);
    }

    /// @brief Create partition info for this policy
    [[nodiscard]] partition_info make_info(size_type global_size,
                                           rank_t num_ranks,
                                           rank_t my_rank) const noexcept {
        partition_info info{};
        info.global_size = global_size;
        info.num_ranks = num_ranks;
        info.my_rank = my_rank;
        info.local_size = local_size(global_size, num_ranks, my_rank);
        info.local_start = local_start(global_size, num_ranks, my_rank);
        info.local_end = info.local_start + static_cast<index_t>(info.local_size);
        return info;
    }

    /// @brief Get the explicit boundaries (if set)
    [[nodiscard]] const std::vector<index_t>& boundaries() const noexcept {
        return boundaries_;
    }

private:
    std::vector<index_t> boundaries_;
};

// Verify concept conformance
static_assert(PartitionPolicyConcept<dynamic_block>,
              "dynamic_block must satisfy PartitionPolicyConcept");

}  // namespace dtl
