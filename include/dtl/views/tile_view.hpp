// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file tile_view.hpp
/// @brief Multi-dimensional tiling view for batch processing
/// @details Divides multi-dimensional ranges into tiles for cache-efficient
///          and GPU-friendly access patterns.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <array>

namespace dtl {

// ============================================================================
// Tile Extent
// ============================================================================

/// @brief Multi-dimensional tile extent descriptor
/// @tparam N Number of dimensions
template <std::size_t N>
struct tile_extent {
    /// @brief Size in each dimension
    std::array<size_type, N> sizes;

    /// @brief Get total number of elements in tile
    [[nodiscard]] constexpr size_type total_size() const noexcept {
        size_type result = 1;
        for (auto s : sizes) result *= s;
        return result;
    }

    /// @brief Get size in dimension d
    [[nodiscard]] constexpr size_type size(std::size_t d) const noexcept {
        return sizes[d];
    }

    /// @brief Number of dimensions
    [[nodiscard]] static constexpr std::size_t rank() noexcept {
        return N;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const tile_extent& other) const noexcept {
        return sizes == other.sizes;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const tile_extent& other) const noexcept {
        return sizes != other.sizes;
    }
};

// ============================================================================
// Tile Type
// ============================================================================

/// @brief A single tile from a tile_view
/// @tparam MDRange The underlying multi-dimensional range type
/// @tparam N Number of dimensions
template <typename MDRange, std::size_t N>
class tile {
public:
    /// @brief Value type
    using value_type = typename MDRange::value_type;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Extent type
    using extent_type = tile_extent<N>;

    /// @brief Origin indices of this tile
    std::array<index_t, N> origin;

    /// @brief Extent of this tile (may be smaller at boundaries)
    extent_type extent;

    /// @brief Requested tile size (for boundary detection)
    extent_type requested_size;

    /// @brief Reference to underlying range
    MDRange* range;

    /// @brief Get total number of elements in tile
    [[nodiscard]] constexpr size_type size() const noexcept {
        return extent.total_size();
    }

    /// @brief Check if tile is at boundary (has reduced extent)
    /// @return true if any dimension is smaller than the requested tile size
    [[nodiscard]] bool is_boundary_tile() const noexcept {
        for (std::size_t d = 0; d < N; ++d) {
            if (extent.sizes[d] < requested_size.sizes[d]) {
                return true;
            }
        }
        return false;
    }

    /// @brief Check if tile is at boundary in specific dimension
    /// @param dim Dimension to check
    [[nodiscard]] bool is_boundary_in_dim(std::size_t dim) const noexcept {
        return dim < N && extent.sizes[dim] < requested_size.sizes[dim];
    }

    /// @brief Get the tile's linear index within the tiled grid
    [[nodiscard]] size_type tile_linear_index(const std::array<size_type, N>& num_tiles) const noexcept {
        size_type idx = 0;
        size_type stride = 1;
        for (std::size_t d = 0; d < N; ++d) {
            size_type tile_coord = static_cast<size_type>(origin[d]) / requested_size.sizes[d];
            idx += tile_coord * stride;
            stride *= num_tiles[d];
        }
        return idx;
    }
};

// ============================================================================
// Tile Iterator
// ============================================================================

/// @brief Iterator over tiles in a tile_view
/// @tparam MDRange The underlying multi-dimensional range type
/// @tparam N Number of dimensions
template <typename MDRange, std::size_t N>
class tile_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = tile<MDRange, N>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    /// @brief Construct tile iterator
    /// @param range Underlying range
    /// @param tile_size Size of each tile
    /// @param range_extent Total extent of the range
    /// @param current Current tile origin position
    tile_iterator(MDRange* range, tile_extent<N> tile_size,
                  std::array<size_type, N> range_extent,
                  std::array<index_t, N> current)
        : range_{range}
        , tile_size_{tile_size}
        , range_extent_{range_extent}
        , current_{current} {}

    /// @brief Dereference to get current tile
    [[nodiscard]] value_type operator*() const {
        value_type t;
        t.origin = current_;
        t.requested_size = tile_size_;
        t.range = range_;

        // Compute actual extent (may be smaller at boundaries)
        for (std::size_t d = 0; d < N; ++d) {
            size_type remaining = range_extent_[d] - static_cast<size_type>(current_[d]);
            t.extent.sizes[d] = std::min(tile_size_.sizes[d], remaining);
        }

        return t;
    }

    /// @brief Pre-increment (advance to next tile)
    tile_iterator& operator++() {
        // Row-major traversal: increment last dimension first
        for (std::size_t d = N; d > 0; --d) {
            std::size_t dim = d - 1;
            current_[dim] += static_cast<index_t>(tile_size_.sizes[dim]);

            // Check if we've passed the end of this dimension
            if (static_cast<size_type>(current_[dim]) < range_extent_[dim]) {
                return *this;  // No carry needed
            }

            // Carry to next dimension (reset this dimension)
            current_[dim] = 0;
        }

        // If we get here, we've gone past the end - set to end position
        current_[0] = static_cast<index_t>(range_extent_[0]);
        return *this;
    }

    /// @brief Post-increment
    tile_iterator operator++(int) {
        tile_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Equality comparison
    [[nodiscard]] bool operator==(const tile_iterator& other) const noexcept {
        return current_ == other.current_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] bool operator!=(const tile_iterator& other) const noexcept {
        return !(*this == other);
    }

private:
    MDRange* range_;
    tile_extent<N> tile_size_;
    std::array<size_type, N> range_extent_;
    std::array<index_t, N> current_;
};

// ============================================================================
// Tile View
// ============================================================================

/// @brief View that tiles a multi-dimensional range
/// @tparam MDRange The underlying multi-dimensional range type
/// @tparam N Number of dimensions
///
/// @par Design Rationale:
/// Tiling enables:
/// - Cache-efficient access patterns (blocking)
/// - GPU thread block mapping
/// - Parallel decomposition for stencil computations
///
/// @par Usage:
/// @code
/// distributed_tensor<float, 2> matrix(1024, 1024, ctx);
/// for (auto tile : tile_view(matrix.local_view(), {32, 32})) {
///     // Process 32x32 tiles for cache efficiency
///     process_tile(tile);
/// }
/// @endcode
template <typename MDRange, std::size_t N>
class tile_view {
public:
    /// @brief Value type (tile)
    using value_type = tile<MDRange, N>;

    /// @brief Iterator type
    using iterator = tile_iterator<MDRange, N>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Extent type
    using extent_type = tile_extent<N>;

    /// @brief Construct from range and tile size
    /// @param range The multi-dimensional range to tile
    /// @param tile_size Size of each tile in each dimension
    constexpr tile_view(MDRange& range, extent_type tile_size) noexcept
        : range_{&range}
        , tile_size_{tile_size} {
        // Initialize range extent from range (assumes range has extent() or size())
        // For now, initialize to zero; concrete implementations should override
        range_extent_.fill(0);
    }

    /// @brief Construct from range, tile size, and explicit range extent
    /// @param range The multi-dimensional range to tile
    /// @param tile_size Size of each tile in each dimension
    /// @param range_extent The extent of the range in each dimension
    constexpr tile_view(MDRange& range, extent_type tile_size,
                        std::array<size_type, N> range_extent) noexcept
        : range_{&range}
        , tile_size_{tile_size}
        , range_extent_{range_extent} {}

    /// @brief Get iterator to first tile
    [[nodiscard]] iterator begin() noexcept {
        std::array<index_t, N> start{};
        return iterator{range_, tile_size_, range_extent_, start};
    }

    /// @brief Get const iterator to first tile
    [[nodiscard]] iterator begin() const noexcept {
        std::array<index_t, N> start{};
        return iterator{range_, tile_size_, range_extent_, start};
    }

    /// @brief Get iterator past last tile
    [[nodiscard]] iterator end() noexcept {
        std::array<index_t, N> end_pos{};
        // End position: first dimension at range_extent, others at 0
        end_pos[0] = static_cast<index_t>(range_extent_[0]);
        return iterator{range_, tile_size_, range_extent_, end_pos};
    }

    /// @brief Get const iterator past last tile
    [[nodiscard]] iterator end() const noexcept {
        std::array<index_t, N> end_pos{};
        end_pos[0] = static_cast<index_t>(range_extent_[0]);
        return iterator{range_, tile_size_, range_extent_, end_pos};
    }

    /// @brief Get tile size
    [[nodiscard]] constexpr extent_type tile_size() const noexcept {
        return tile_size_;
    }

    /// @brief Get the range extent
    [[nodiscard]] constexpr std::array<size_type, N> range_extent() const noexcept {
        return range_extent_;
    }

    /// @brief Set the range extent (for deferred initialization)
    void set_range_extent(std::array<size_type, N> extent) noexcept {
        range_extent_ = extent;
    }

    /// @brief Get number of tiles in each dimension
    [[nodiscard]] std::array<size_type, N> num_tiles() const noexcept {
        std::array<size_type, N> result{};
        for (std::size_t d = 0; d < N; ++d) {
            if (tile_size_.sizes[d] == 0) {
                result[d] = 0;
            } else {
                // Ceiling division: (extent + tile_size - 1) / tile_size
                result[d] = (range_extent_[d] + tile_size_.sizes[d] - 1) / tile_size_.sizes[d];
            }
        }
        return result;
    }

    /// @brief Get total number of tiles
    [[nodiscard]] size_type total_tiles() const noexcept {
        auto counts = num_tiles();
        size_type total = 1;
        for (auto c : counts) total *= c;
        return total;
    }

    /// @brief Check if the view is empty (no tiles)
    [[nodiscard]] bool empty() const noexcept {
        for (std::size_t d = 0; d < N; ++d) {
            if (range_extent_[d] == 0) return true;
        }
        return false;
    }

    /// @brief Get underlying range reference
    [[nodiscard]] MDRange& underlying() noexcept {
        return *range_;
    }

    /// @brief Get underlying range reference (const)
    [[nodiscard]] const MDRange& underlying() const noexcept {
        return *range_;
    }

private:
    MDRange* range_;
    extent_type tile_size_;
    std::array<size_type, N> range_extent_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Factory function for 2D tiling
/// @tparam MDRange The multi-dimensional range type
/// @param range The range to tile
/// @param tile_rows Number of rows per tile
/// @param tile_cols Number of columns per tile
template <typename MDRange>
[[nodiscard]] constexpr auto make_tile_view(MDRange& range, size_type tile_rows, size_type tile_cols) {
    return tile_view<MDRange, 2>{range, tile_extent<2>{{tile_rows, tile_cols}}};
}

/// @brief Factory function for 3D tiling
/// @tparam MDRange The multi-dimensional range type
template <typename MDRange>
[[nodiscard]] constexpr auto make_tile_view(MDRange& range, size_type d0, size_type d1, size_type d2) {
    return tile_view<MDRange, 3>{range, tile_extent<3>{{d0, d1, d2}}};
}

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Check if a type is a tile_view
template <typename T>
struct is_tile_view : std::false_type {};

template <typename MDRange, std::size_t N>
struct is_tile_view<tile_view<MDRange, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_tile_view_v = is_tile_view<T>::value;

}  // namespace dtl
