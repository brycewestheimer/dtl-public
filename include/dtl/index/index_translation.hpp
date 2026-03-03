// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file index_translation.hpp
/// @brief Index translation utilities: owner(), is_local(), to_local(), to_global()
/// @details Provides functions to translate between global and local indices.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/index/global_index.hpp>
#include <dtl/index/local_index.hpp>

namespace dtl {

// ============================================================================
// Index Translation Context
// ============================================================================

/// @brief Context for index translation operations
/// @details Encapsulates the information needed to translate between global
///          and local indices for a given distribution configuration.
template <typename T = index_t>
struct index_translation_context {
    /// @brief Total global size
    T global_size = 0;

    /// @brief Number of ranks in the communicator
    rank_t num_ranks = 0;

    /// @brief This process's rank
    rank_t my_rank = no_rank;

    /// @brief Check if context is valid
    [[nodiscard]] constexpr bool valid() const noexcept {
        return global_size > 0 && num_ranks > 0 && my_rank != no_rank;
    }
};

// ============================================================================
// Block Partition Translation
// ============================================================================

namespace block_partition_translation {

/// @brief Calculate the owner rank for a global index (block partition)
/// @tparam T Index type
/// @param global_idx Global index
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @return Owning rank
template <typename T = index_t>
[[nodiscard]] constexpr rank_t owner(
    T global_idx,
    T global_size,
    rank_t num_ranks) noexcept {
    if (global_idx < 0 || global_idx >= global_size || num_ranks <= 0) {
        return no_rank;
    }

    // Block size (rounded up for some ranks)
    T base_size = global_size / num_ranks;
    T remainder = global_size % num_ranks;

    // First 'remainder' ranks get one extra element
    // Elements [0, remainder*(base_size+1)) are on ranks [0, remainder)
    // Elements [remainder*(base_size+1), global_size) are on ranks [remainder, num_ranks)

    T threshold = remainder * (base_size + 1);
    if (global_idx < threshold) {
        return static_cast<rank_t>(global_idx / (base_size + 1));
    } else {
        return static_cast<rank_t>(remainder + (global_idx - threshold) / base_size);
    }
}

/// @brief Calculate the owner rank for a global_index (block partition)
template <typename T = index_t>
[[nodiscard]] constexpr rank_t owner(
    global_index<T> idx,
    T global_size,
    rank_t num_ranks) noexcept {
    return owner(idx.value(), global_size, num_ranks);
}

/// @brief Calculate local size for a given rank (block partition)
/// @tparam T Index type
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Local size for the rank
template <typename T = index_t>
[[nodiscard]] constexpr T local_size(
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    if (num_ranks <= 0 || rank < 0 || rank >= num_ranks) {
        return 0;
    }

    T base_size = global_size / num_ranks;
    T remainder = global_size % num_ranks;

    // First 'remainder' ranks get one extra element
    return base_size + (rank < remainder ? 1 : 0);
}

/// @brief Calculate the global offset for a rank's local data (block partition)
/// @tparam T Index type
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Global offset where this rank's data begins
template <typename T = index_t>
[[nodiscard]] constexpr T rank_offset(
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    if (num_ranks <= 0 || rank < 0 || rank >= num_ranks) {
        return 0;
    }

    T base_size = global_size / num_ranks;
    T remainder = global_size % num_ranks;

    if (rank < remainder) {
        return rank * (base_size + 1);
    } else {
        return remainder * (base_size + 1) + (rank - remainder) * base_size;
    }
}

/// @brief Convert global index to local index (block partition)
/// @tparam T Index type
/// @param global_idx Global index
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Local index, or invalid if not owned by rank
template <typename T = index_t>
[[nodiscard]] constexpr local_index<T> to_local(
    T global_idx,
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    rank_t owning_rank = owner(global_idx, global_size, num_ranks);
    if (owning_rank != rank) {
        return local_index<T>();  // Invalid
    }

    T offset = rank_offset<T>(global_size, num_ranks, rank);
    return local_index<T>(global_idx - offset);
}

/// @brief Convert global_index to local_index (block partition)
template <typename T = index_t>
[[nodiscard]] constexpr local_index<T> to_local(
    global_index<T> idx,
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    return to_local(idx.value(), global_size, num_ranks, rank);
}

/// @brief Convert local index to global index (block partition)
/// @tparam T Index type
/// @param local_idx Local index
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Source rank
/// @return Global index
template <typename T = index_t>
[[nodiscard]] constexpr global_index<T> to_global(
    T local_idx,
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    T local_sz = local_size<T>(global_size, num_ranks, rank);
    if (local_idx < 0 || local_idx >= local_sz) {
        return global_index<T>();  // Invalid
    }

    T offset = rank_offset<T>(global_size, num_ranks, rank);
    return global_index<T>(offset + local_idx);
}

/// @brief Convert local_index to global_index (block partition)
template <typename T = index_t>
[[nodiscard]] constexpr global_index<T> to_global(
    local_index<T> idx,
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    return to_global(idx.value(), global_size, num_ranks, rank);
}

/// @brief Check if global index is local (block partition)
/// @tparam T Index type
/// @param global_idx Global index
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param my_rank Calling rank
/// @return true if index is owned by my_rank
template <typename T = index_t>
[[nodiscard]] constexpr bool is_local(
    T global_idx,
    T global_size,
    rank_t num_ranks,
    rank_t my_rank) noexcept {
    return owner(global_idx, global_size, num_ranks) == my_rank;
}

/// @brief Check if global_index is local (block partition)
template <typename T = index_t>
[[nodiscard]] constexpr bool is_local(
    global_index<T> idx,
    T global_size,
    rank_t num_ranks,
    rank_t my_rank) noexcept {
    return is_local(idx.value(), global_size, num_ranks, my_rank);
}

/// @brief Get the global range owned by a rank (block partition)
/// @tparam T Index type
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Range [begin, end) owned by the rank
template <typename T = index_t>
[[nodiscard]] constexpr global_index_range<T> owned_range(
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    T offset = rank_offset<T>(global_size, num_ranks, rank);
    T size = local_size<T>(global_size, num_ranks, rank);
    return global_index_range<T>(offset, offset + size);
}

}  // namespace block_partition_translation

// ============================================================================
// Cyclic Partition Translation
// ============================================================================

namespace cyclic_partition_translation {

/// @brief Calculate the owner rank for a global index (cyclic partition)
/// @tparam T Index type
/// @param global_idx Global index
/// @param num_ranks Number of ranks
/// @return Owning rank
template <typename T = index_t>
[[nodiscard]] constexpr rank_t owner(T global_idx, rank_t num_ranks) noexcept {
    if (global_idx < 0 || num_ranks <= 0) {
        return no_rank;
    }
    return static_cast<rank_t>(global_idx % num_ranks);
}

/// @brief Calculate local size for a given rank (cyclic partition)
/// @tparam T Index type
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Local size for the rank
template <typename T = index_t>
[[nodiscard]] constexpr T local_size(
    T global_size,
    rank_t num_ranks,
    rank_t rank) noexcept {
    if (num_ranks <= 0 || rank < 0 || rank >= num_ranks) {
        return 0;
    }

    T base_count = global_size / num_ranks;
    T remainder = global_size % num_ranks;

    return base_count + (rank < remainder ? 1 : 0);
}

/// @brief Convert global index to local index (cyclic partition)
/// @tparam T Index type
/// @param global_idx Global index
/// @param num_ranks Number of ranks
/// @param rank Target rank
/// @return Local index, or invalid if not owned by rank
template <typename T = index_t>
[[nodiscard]] constexpr local_index<T> to_local(
    T global_idx,
    rank_t num_ranks,
    rank_t rank) noexcept {
    if (owner(global_idx, num_ranks) != rank) {
        return local_index<T>();  // Invalid
    }
    return local_index<T>(global_idx / num_ranks);
}

/// @brief Convert local index to global index (cyclic partition)
/// @tparam T Index type
/// @param local_idx Local index
/// @param num_ranks Number of ranks
/// @param rank Source rank
/// @return Global index
template <typename T = index_t>
[[nodiscard]] constexpr global_index<T> to_global(
    T local_idx,
    rank_t num_ranks,
    rank_t rank) noexcept {
    if (local_idx < 0 || num_ranks <= 0 || rank < 0 || rank >= num_ranks) {
        return global_index<T>();  // Invalid
    }
    return global_index<T>(local_idx * num_ranks + rank);
}

/// @brief Check if global index is local (cyclic partition)
template <typename T = index_t>
[[nodiscard]] constexpr bool is_local(
    T global_idx,
    rank_t num_ranks,
    rank_t my_rank) noexcept {
    return owner(global_idx, num_ranks) == my_rank;
}

}  // namespace cyclic_partition_translation

// ============================================================================
// Generic Translation Interface
// ============================================================================

/// @brief Index translator for a specific partition policy
/// @tparam PartitionPolicy The partition policy type
template <typename PartitionPolicy>
class index_translator {
public:
    using index_type = index_t;

    /// @brief Construct with translation context
    /// @param global_size Total global size
    /// @param num_ranks Number of ranks
    /// @param my_rank Calling rank
    index_translator(index_type global_size, rank_t num_ranks, rank_t my_rank)
        : global_size_(global_size)
        , num_ranks_(num_ranks)
        , my_rank_(my_rank) {}

    /// @brief Get the owner of a global index
    [[nodiscard]] rank_t owner(global_index<index_type> idx) const noexcept {
        return PartitionPolicy::owner(idx.value(), global_size_, num_ranks_);
    }

    /// @brief Check if index is local
    [[nodiscard]] bool is_local(global_index<index_type> idx) const noexcept {
        return owner(idx) == my_rank_;
    }

    /// @brief Convert to local index
    [[nodiscard]] local_index<index_type> to_local(global_index<index_type> idx) const noexcept {
        return PartitionPolicy::to_local(idx.value(), global_size_, num_ranks_, my_rank_);
    }

    /// @brief Convert to global index
    [[nodiscard]] global_index<index_type> to_global(local_index<index_type> idx) const noexcept {
        return PartitionPolicy::to_global(idx.value(), global_size_, num_ranks_, my_rank_);
    }

    /// @brief Get local size for my rank
    [[nodiscard]] index_type local_size() const noexcept {
        return PartitionPolicy::local_size(global_size_, num_ranks_, my_rank_);
    }

    /// @brief Get local size for any rank
    [[nodiscard]] index_type local_size(rank_t rank) const noexcept {
        return PartitionPolicy::local_size(global_size_, num_ranks_, rank);
    }

    /// @brief Get global size
    [[nodiscard]] index_type global_size() const noexcept {
        return global_size_;
    }

    /// @brief Get number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get my rank
    [[nodiscard]] rank_t my_rank() const noexcept {
        return my_rank_;
    }

private:
    index_type global_size_;
    rank_t num_ranks_;
    rank_t my_rank_;
};

// ============================================================================
// Policy Wrapper for Block Partition
// ============================================================================

/// @brief Policy wrapper for block partition translation functions
struct block_partition_policy {
    template <typename T>
    static constexpr rank_t owner(T global_idx, T global_size, rank_t num_ranks) noexcept {
        return block_partition_translation::owner(global_idx, global_size, num_ranks);
    }

    template <typename T>
    static constexpr T local_size(T global_size, rank_t num_ranks, rank_t rank) noexcept {
        return block_partition_translation::local_size(global_size, num_ranks, rank);
    }

    template <typename T>
    static constexpr local_index<T> to_local(
        T global_idx, T global_size, rank_t num_ranks, rank_t rank) noexcept {
        return block_partition_translation::to_local(global_idx, global_size, num_ranks, rank);
    }

    template <typename T>
    static constexpr global_index<T> to_global(
        T local_idx, T global_size, rank_t num_ranks, rank_t rank) noexcept {
        return block_partition_translation::to_global(local_idx, global_size, num_ranks, rank);
    }
};

/// @brief Type alias for block partition translator
using block_translator = index_translator<block_partition_policy>;

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Create a block partition translator
/// @param global_size Total global size
/// @param num_ranks Number of ranks
/// @param my_rank Calling rank
/// @return Block partition translator
[[nodiscard]] inline block_translator make_block_translator(
    index_t global_size,
    rank_t num_ranks,
    rank_t my_rank) {
    return block_translator(global_size, num_ranks, my_rank);
}

}  // namespace dtl
