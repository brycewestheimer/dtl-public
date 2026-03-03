// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file custom_partition.hpp
/// @brief User-defined custom partition policy
/// @details Allows user-provided mapping functions for flexible partitioning.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

#include <functional>

namespace dtl {

/// @brief Custom partition with user-defined mapping function
/// @tparam OwnerFn Type of the owner mapping function
/// @details Allows arbitrary partition schemes through user-provided functions.
///
/// @par Expected function signature:
/// @code
/// rank_t owner_fn(index_t global_idx, size_type global_size, rank_t num_ranks);
/// @endcode
template <typename OwnerFn>
struct custom_partition {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief The owner function type
    using owner_function = OwnerFn;

    /// @brief Construct with owner function
    /// @param fn The function mapping global index to owner rank
    explicit custom_partition(OwnerFn fn) : owner_fn_{std::move(fn)} {}

    /// @brief Determine which rank owns a global index
    /// @param global_idx The global index to query
    /// @param global_size Total size of the distributed range
    /// @param num_ranks Number of ranks in the communicator
    /// @return The owning rank
    [[nodiscard]] rank_t owner(index_t global_idx,
                               size_type global_size,
                               rank_t num_ranks) const {
        return owner_fn_(global_idx, global_size, num_ranks);
    }

    /// @brief Calculate local size for a given rank
    /// @note Custom partitions require scanning to compute exact local size.
    ///       This returns an estimate.
    [[nodiscard]] static constexpr size_type local_size(size_type global_size,
                                                        rank_t num_ranks,
                                                        rank_t /*rank*/) noexcept {
        if (global_size == 0 || num_ranks <= 0) {
            return 0;
        }
        // Return average as estimate
        return global_size / static_cast<size_type>(num_ranks);
    }

    /// @brief Calculate exact local size by scanning
    /// @param global_size Total size
    /// @param num_ranks Number of ranks
    /// @param rank The rank to query
    /// @return Exact count of elements owned by rank
    [[nodiscard]] size_type exact_local_size(size_type global_size,
                                             rank_t num_ranks,
                                             rank_t rank) const {
        size_type count = 0;
        for (size_type i = 0; i < global_size; ++i) {
            if (owner_fn_(static_cast<index_t>(i), global_size, num_ranks) == rank) {
                ++count;
            }
        }
        return count;
    }

    /// @brief Convert global index to local index
    /// @note For custom partitions, requires scanning previous elements.
    ///       Stub implementation returns unchecked conversion.
    [[nodiscard]] index_t to_local(index_t global_idx,
                                   size_type global_size,
                                   rank_t num_ranks,
                                   rank_t rank) const {
        // Count owned elements before global_idx
        index_t local_idx = 0;
        for (index_t i = 0; i < global_idx; ++i) {
            if (owner_fn_(i, global_size, num_ranks) == rank) {
                ++local_idx;
            }
        }
        return local_idx;
    }

    /// @brief Convert local index to global index
    /// @note For custom partitions, requires scanning to find nth owned element.
    [[nodiscard]] index_t to_global(index_t local_idx,
                                    size_type global_size,
                                    rank_t num_ranks,
                                    rank_t rank) const {
        index_t count = 0;
        for (index_t i = 0; i < static_cast<index_t>(global_size); ++i) {
            if (owner_fn_(i, global_size, num_ranks) == rank) {
                if (count == local_idx) {
                    return i;
                }
                ++count;
            }
        }
        return -1;  // Not found
    }

private:
    OwnerFn owner_fn_;
};

/// @brief Factory function to create custom partition
/// @tparam Fn Function type
/// @param fn The owner mapping function
/// @return custom_partition<Fn> instance
template <typename Fn>
[[nodiscard]] auto make_custom_partition(Fn&& fn) {
    return custom_partition<std::decay_t<Fn>>{std::forward<Fn>(fn)};
}

/// @brief Type-erased custom partition using std::function
using dynamic_custom_partition = custom_partition<
    std::function<rank_t(index_t, size_type, rank_t)>>;

// Verify concept conformance
static_assert(PartitionPolicyConcept<dynamic_custom_partition>,
              "custom_partition must satisfy PartitionPolicyConcept");

}  // namespace dtl
