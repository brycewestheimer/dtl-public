// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file block_nd_partition.hpp
/// @brief N-dimensional block decomposition partition policy
/// @details Supports 2D+ block decomposition via a process grid.
///          Rank-to-grid mapping uses row-major linearization.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/partition/partition_policy.hpp>
#include <dtl/policies/partition/block_partition.hpp>

#include <array>
#include <numeric>

namespace dtl {

/// @brief N-dimensional block partition policy
/// @tparam N Number of dimensions in the process grid
/// @details Decomposes a multi-dimensional domain using an N-dimensional
///          process grid. Each rank is assigned a position in the grid
///          via row-major linearization:
///            rank = g[0] * (G[1]*G[2]*...) + g[1] * (G[2]*...) + ... + g[N-1]
///
///          Each dimension is independently block-partitioned according
///          to the number of processes assigned along that dimension.
///
/// @par Example (2D, 12x8 domain on 6 ranks with 3x2 grid):
/// @code
///   Process grid:
///     (0,0) (0,1)
///     (1,0) (1,1)
///     (2,0) (2,1)
///
///   Rank 0 -> grid (0,0) -> rows [0,4), cols [0,4)
///   Rank 1 -> grid (0,1) -> rows [0,4), cols [4,8)
///   Rank 2 -> grid (1,0) -> rows [4,8), cols [0,4)
///   Rank 3 -> grid (1,1) -> rows [4,8), cols [4,8)
///   Rank 4 -> grid (2,0) -> rows [8,12), cols [0,4)
///   Rank 5 -> grid (2,1) -> rows [8,12), cols [4,8)
/// @endcode
template <size_type N>
class block_nd_partition {
public:
    /// @brief Policy category tag
    using policy_category = partition_policy_tag;

    /// @brief Number of dimensions
    static constexpr size_type dimensions = N;

    /// @brief Process grid dimensions type
    using grid_type = std::array<rank_t, N>;

    /// @brief Extent type (domain shape)
    using extent_type = nd_extent<N>;

    /// @brief Grid coordinate type
    using grid_coord_type = std::array<rank_t, N>;

    // =========================================================================
    // Construction
    // =========================================================================

    /// @brief Construct with process grid dimensions
    /// @param proc_grid Number of processes along each dimension
    /// @details The product of all grid dimensions must equal the total
    ///          number of ranks used.
    constexpr explicit block_nd_partition(const grid_type& proc_grid) noexcept
        : proc_grid_{proc_grid} {}

    /// @brief Default constructor (all grid dims = 1)
    constexpr block_nd_partition() noexcept {
        proc_grid_.fill(1);
    }

    // =========================================================================
    // Process Grid Queries
    // =========================================================================

    /// @brief Get the process grid dimensions
    [[nodiscard]] constexpr const grid_type& proc_grid() const noexcept {
        return proc_grid_;
    }

    /// @brief Get the number of processes along dimension d
    [[nodiscard]] constexpr rank_t proc_grid_dim(size_type d) const noexcept {
        return proc_grid_[d];
    }

    /// @brief Total number of ranks required by this grid
    [[nodiscard]] constexpr rank_t total_ranks() const noexcept {
        rank_t total = 1;
        for (size_type d = 0; d < N; ++d) {
            total *= proc_grid_[d];
        }
        return total;
    }

    // =========================================================================
    // Rank-to-Grid Mapping (Row-Major Linearization)
    // =========================================================================

    /// @brief Convert a linear rank to N-dimensional grid coordinates
    /// @param rank The linear rank
    /// @return Grid coordinates (d0, d1, ..., dN-1)
    [[nodiscard]] constexpr grid_coord_type rank_to_grid(rank_t rank) const noexcept {
        grid_coord_type coords{};
        rank_t remaining = rank;
        for (size_type d = N; d > 0; --d) {
            coords[d - 1] = remaining % proc_grid_[d - 1];
            remaining /= proc_grid_[d - 1];
        }
        return coords;
    }

    /// @brief Convert N-dimensional grid coordinates to a linear rank
    /// @param coords Grid coordinates (d0, d1, ..., dN-1)
    /// @return Linear rank
    [[nodiscard]] constexpr rank_t grid_to_rank(const grid_coord_type& coords) const noexcept {
        rank_t rank = 0;
        rank_t stride = 1;
        for (size_type d = N; d > 0; --d) {
            rank += coords[d - 1] * stride;
            stride *= proc_grid_[d - 1];
        }
        return rank;
    }

    // =========================================================================
    // Local Extent Computation
    // =========================================================================

    /// @brief Compute the local extent for a given rank along a given dimension
    /// @param global_extent The global extent along this dimension
    /// @param grid_dim Number of processes along this dimension
    /// @param grid_coord The rank's grid coordinate along this dimension
    /// @return Number of elements owned by this rank along this dimension
    [[nodiscard]] static constexpr size_type local_extent_1d(
        size_type global_extent, rank_t grid_dim, rank_t grid_coord) noexcept {
        return block_partition<0>::local_size(global_extent, grid_dim, grid_coord);
    }

    /// @brief Compute the local start index along a given dimension
    /// @param global_extent The global extent along this dimension
    /// @param grid_dim Number of processes along this dimension
    /// @param grid_coord The rank's grid coordinate along this dimension
    /// @return Global start index for this rank along this dimension
    [[nodiscard]] static constexpr index_t local_start_1d(
        size_type global_extent, rank_t grid_dim, rank_t grid_coord) noexcept {
        return block_partition<0>::local_start(global_extent, grid_dim, grid_coord);
    }

    /// @brief Compute local extents for a given rank
    /// @param global_extents Global extents of the domain
    /// @param rank The rank to compute for
    /// @return Local extent along each dimension
    [[nodiscard]] constexpr extent_type local_extents(
        const extent_type& global_extents, rank_t rank) const noexcept {
        auto coords = rank_to_grid(rank);
        extent_type result{};
        for (size_type d = 0; d < N; ++d) {
            result[d] = local_extent_1d(global_extents[d], proc_grid_[d], coords[d]);
        }
        return result;
    }

    /// @brief Compute local start indices (offsets) for a given rank
    /// @param global_extents Global extents of the domain
    /// @param rank The rank to compute for
    /// @return Local start index along each dimension
    [[nodiscard]] constexpr nd_index<N> local_starts(
        const extent_type& global_extents, rank_t rank) const noexcept {
        auto coords = rank_to_grid(rank);
        nd_index<N> result{};
        for (size_type d = 0; d < N; ++d) {
            result[d] = local_start_1d(global_extents[d], proc_grid_[d], coords[d]);
        }
        return result;
    }

    /// @brief Compute total local size (product of local extents) for a given rank
    /// @param global_extents Global extents of the domain
    /// @param rank The rank to compute for
    /// @return Total number of local elements
    [[nodiscard]] constexpr size_type local_size(
        const extent_type& global_extents, rank_t rank) const noexcept {
        auto le = local_extents(global_extents, rank);
        size_type total = 1;
        for (size_type d = 0; d < N; ++d) {
            total *= le[d];
        }
        return total;
    }

    // =========================================================================
    // Ownership Queries
    // =========================================================================

    /// @brief Determine the owner rank of a global ND index
    /// @param global_idx Global ND index
    /// @param global_extents Global extents of the domain
    /// @return The rank that owns the given index
    [[nodiscard]] constexpr rank_t owner(
        const nd_index<N>& global_idx,
        const extent_type& global_extents) const noexcept {
        grid_coord_type coords{};
        for (size_type d = 0; d < N; ++d) {
            coords[d] = block_partition<0>::owner(
                global_idx[d], global_extents[d], proc_grid_[d]);
        }
        return grid_to_rank(coords);
    }

    /// @brief Check if a global ND index is local to a given rank
    /// @param global_idx Global ND index
    /// @param global_extents Global extents of the domain
    /// @param rank The rank to check against
    /// @return true if the index is owned by the given rank
    [[nodiscard]] constexpr bool is_local(
        const nd_index<N>& global_idx,
        const extent_type& global_extents,
        rank_t rank) const noexcept {
        return owner(global_idx, global_extents) == rank;
    }

    /// @brief Convert global ND index to local ND index
    /// @param global_idx Global ND index
    /// @param global_extents Global extents of the domain
    /// @param rank The rank to convert for
    /// @return Local ND index
    [[nodiscard]] constexpr nd_index<N> to_local(
        const nd_index<N>& global_idx,
        const extent_type& global_extents,
        rank_t rank) const noexcept {
        auto starts = local_starts(global_extents, rank);
        nd_index<N> result{};
        for (size_type d = 0; d < N; ++d) {
            result[d] = global_idx[d] - starts[d];
        }
        return result;
    }

    /// @brief Convert local ND index to global ND index
    /// @param local_idx Local ND index
    /// @param global_extents Global extents of the domain
    /// @param rank The rank the local index belongs to
    /// @return Global ND index
    [[nodiscard]] constexpr nd_index<N> to_global(
        const nd_index<N>& local_idx,
        const extent_type& global_extents,
        rank_t rank) const noexcept {
        auto starts = local_starts(global_extents, rank);
        nd_index<N> result{};
        for (size_type d = 0; d < N; ++d) {
            result[d] = local_idx[d] + starts[d];
        }
        return result;
    }

private:
    grid_type proc_grid_;
};

// =============================================================================
// Convenience Aliases
// =============================================================================

/// @brief 2D block partition
using block_2d_partition = block_nd_partition<2>;

/// @brief 3D block partition
using block_3d_partition = block_nd_partition<3>;

}  // namespace dtl
