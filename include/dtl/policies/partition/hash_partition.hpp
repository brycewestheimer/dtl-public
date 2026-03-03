// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hash_partition.hpp
/// @brief Hash-based partition policy
/// @details Distributes data based on hash values, useful for associative containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

#include <functional>

namespace dtl {

/// @brief Hash partition distributes elements based on hash function
/// @tparam Hash Hash function type (default: std::hash)
/// @details Hash partitioning uses a hash function to determine ownership.
///          This is primarily useful for distributed maps and sets where
///          the key's hash determines the owning rank.
///
/// @par Example:
/// @code
///   hash_partition<std::hash<Key>> partition;
///   rank_t owner = partition.owner(hash(key), global_size, num_ranks);
/// @endcode
template <typename Hash = std::hash<size_type>>
struct hash_partition {
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief The hash function type
    using hasher = Hash;

    /// @brief Determine which rank owns a global index (hash value)
    /// @param hash_value The hash value (used as index)
    /// @param global_size Total number of buckets (unused for simple modulo)
    /// @param num_ranks Number of ranks in the communicator
    /// @return The owning rank
    [[nodiscard]] static constexpr rank_t owner(index_t hash_value,
                                                 [[maybe_unused]] size_type global_size,
                                                 rank_t num_ranks) noexcept {
        if (num_ranks <= 0) {
            return 0;
        }
        // Use absolute value to handle potential negative hash values
        const auto abs_hash = static_cast<size_type>(
            hash_value >= 0 ? hash_value : -hash_value);
        return static_cast<rank_t>(abs_hash % static_cast<size_type>(num_ranks));
    }

    /// @brief Determine which rank owns a key
    /// @tparam Key The key type
    /// @param key The key to hash
    /// @param global_size Total size (unused)
    /// @param num_ranks Number of ranks
    /// @return The owning rank
    template <typename Key>
    [[nodiscard]] static rank_t owner_of(const Key& key,
                                          size_type global_size,
                                          rank_t num_ranks) {
        Hash h{};
        return owner(static_cast<index_t>(h(key)), global_size, num_ranks);
    }

    /// @brief Calculate local size for a given rank
    /// @note For hash partitions, local size is not easily predictable
    ///       This returns an estimate based on uniform distribution
    [[nodiscard]] static constexpr size_type local_size(size_type global_size,
                                                        rank_t num_ranks,
                                                        rank_t /*rank*/) noexcept {
        if (global_size == 0 || num_ranks <= 0) {
            return 0;
        }
        // Estimate based on uniform distribution
        return (global_size + static_cast<size_type>(num_ranks) - 1) /
               static_cast<size_type>(num_ranks);
    }

    /// @brief Convert global index to local index
    /// @note For hash partitions, this is identity (hash is the local key)
    [[nodiscard]] static constexpr index_t to_local(index_t global_idx,
                                                     size_type /*global_size*/,
                                                     rank_t /*num_ranks*/,
                                                     rank_t /*rank*/) noexcept {
        return global_idx;
    }

    /// @brief Convert local index to global index
    /// @note For hash partitions, this is identity
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
        info.local_size = local_size(global_size, num_ranks, my_rank);
        info.local_start = 0;
        info.local_end = static_cast<index_t>(info.local_size);
        return info;
    }
};

}  // namespace dtl
