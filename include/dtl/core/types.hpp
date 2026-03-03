// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file types.hpp
/// @brief Core type definitions for DTL
/// @details Defines fundamental types used throughout the library including
///          index types, rank types, tag types, and multi-dimensional extents.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#include <cstddef>
#include <cstdint>
#include <array>
#include <type_traits>

namespace dtl {

// =============================================================================
// Fundamental Type Aliases
// =============================================================================

/// @brief Primary index type for 1D container indexing
/// @details Signed type to allow negative indices in certain contexts
using index_t = std::ptrdiff_t;

/// @brief Rank identifier type (MPI-compatible)
/// @details Uses int for direct compatibility with MPI rank types
using rank_t = int;

/// @brief Size type for container sizes and counts
using size_type = std::size_t;

/// @brief Difference type for iterator arithmetic
using difference_type = std::ptrdiff_t;

// =============================================================================
// Sentinel Values
// =============================================================================

/// @brief Sentinel value indicating no specific rank
inline constexpr rank_t no_rank = -1;

/// @brief Sentinel value indicating all ranks (broadcast target)
inline constexpr rank_t all_ranks = -2;

/// @brief Sentinel value for root rank in collective operations
inline constexpr rank_t root_rank = 0;

/// @brief Dynamic extent marker for extents template
/// @details Used to indicate runtime-determined dimensions
inline constexpr size_type dynamic_extent = static_cast<size_type>(-1);

// =============================================================================
// Tag Types
// =============================================================================

/// @brief Tag type for local operations (no communication)
struct local_tag {
    explicit local_tag() = default;
};

/// @brief Tag type for global operations (may communicate)
struct global_tag {
    explicit global_tag() = default;
};

/// @brief Tag type for synchronous operations
struct sync_tag {
    explicit sync_tag() = default;
};

/// @brief Tag type for asynchronous operations
struct async_tag {
    explicit async_tag() = default;
};

/// @brief Tag type for blocking operations
struct blocking_tag {
    explicit blocking_tag() = default;
};

/// @brief Tag type for non-blocking operations
struct nonblocking_tag {
    explicit nonblocking_tag() = default;
};

/// @brief Tag type for in-place operations
struct in_place_tag {
    explicit in_place_tag() = default;
};

/// @brief Tag type for read-only access
struct read_only_tag {
    explicit read_only_tag() = default;
};

/// @brief Tag type for write-only access
struct write_only_tag {
    explicit write_only_tag() = default;
};

/// @brief Tag type for read-write access
struct read_write_tag {
    explicit read_write_tag() = default;
};

/// @brief Tag instances for convenient use
/// @{
inline constexpr local_tag local{};
inline constexpr global_tag global{};
inline constexpr sync_tag sync{};
inline constexpr async_tag async_v{};  // Note: 'async' conflicts with execution policy
inline constexpr blocking_tag blocking{};
inline constexpr nonblocking_tag nonblocking{};
inline constexpr in_place_tag in_place{};
inline constexpr read_only_tag read_only{};
inline constexpr write_only_tag write_only{};
inline constexpr read_write_tag read_write{};
/// @}

// =============================================================================
// Multi-dimensional Extent Types
// =============================================================================

/// @brief Multi-dimensional extent descriptor
/// @tparam Extents Compile-time extents (use dynamic_extent for runtime dimensions)
/// @details Inspired by std::extents from mdspan. Supports mixed static/dynamic dimensions.
template <size_type... Extents>
struct extents {
    /// @brief Number of dimensions
    /// @return The rank (dimensionality) of the extents
    static constexpr size_type rank() noexcept {
        return sizeof...(Extents);
    }

    /// @brief Number of dynamic (runtime-determined) dimensions
    /// @return Count of dimensions that are dynamic_extent
    static constexpr size_type rank_dynamic() noexcept {
        return ((Extents == dynamic_extent ? 1 : 0) + ...);
    }

    /// @brief Get static extent for dimension n
    /// @param n The dimension index
    /// @return The static extent, or dynamic_extent if runtime-determined
    static constexpr size_type static_extent(size_type n) noexcept {
        constexpr std::array<size_type, sizeof...(Extents)> ext{Extents...};
        return n < rank() ? ext[n] : dynamic_extent;
    }

    /// @brief Default constructor
    constexpr extents() noexcept = default;

    /// @brief Construct with dynamic extents
    /// @tparam OtherSizeTypes Size types for dynamic dimensions
    /// @param dynamic_extents Values for dynamic dimensions
    template <typename... OtherSizeTypes>
        requires(sizeof...(OtherSizeTypes) == rank_dynamic()) &&
                (std::is_convertible_v<OtherSizeTypes, size_type> && ...)
    constexpr explicit extents(OtherSizeTypes... dynamic_extents) noexcept
        : dynamic_extents_{static_cast<size_type>(dynamic_extents)...} {}

    /// @brief Get extent for dimension n
    /// @param n The dimension index
    /// @return The extent of dimension n
    [[nodiscard]] constexpr size_type extent(size_type n) const noexcept {
        if constexpr (rank_dynamic() == 0) {
            return static_extent(n);
        } else {
            constexpr std::array<size_type, sizeof...(Extents)> static_ext{Extents...};
            if (static_ext[n] != dynamic_extent) {
                return static_ext[n];
            }
            // Count dynamic extents before n
            size_type dyn_idx = 0;
            for (size_type i = 0; i < n; ++i) {
                if (static_ext[i] == dynamic_extent) {
                    ++dyn_idx;
                }
            }
            return dynamic_extents_[dyn_idx];
        }
    }

    /// @brief Calculate total number of elements
    /// @return Product of all extents
    [[nodiscard]] constexpr size_type size() const noexcept {
        size_type result = 1;
        for (size_type i = 0; i < rank(); ++i) {
            result *= extent(i);
        }
        return result;
    }

private:
    DTL_NO_UNIQUE_ADDRESS
    std::array<size_type, rank_dynamic()> dynamic_extents_{};
};

/// @brief Deduction guide for all-dynamic extents
template <typename... SizeTypes>
extents(SizeTypes...) -> extents<(dynamic_extent, static_cast<void>(std::declval<SizeTypes>()), dynamic_extent)...>;

/// @brief Common extent type aliases
/// @{
using extents_1d = extents<dynamic_extent>;
using extents_2d = extents<dynamic_extent, dynamic_extent>;
using extents_3d = extents<dynamic_extent, dynamic_extent, dynamic_extent>;
/// @}

// =============================================================================
// N-Dimensional Index Type
// =============================================================================

/// @brief N-dimensional index for multi-dimensional indexing
/// @tparam N Number of dimensions
/// @details Represents a coordinate in N-dimensional space for use with
///          tensors and other multi-dimensional containers.
template <size_type N>
struct nd_index {
    /// @brief The indices for each dimension
    std::array<index_t, N> indices{};

    /// @brief Default constructor (all zeros)
    constexpr nd_index() noexcept = default;

    /// @brief Construct from variadic indices
    template <typename... Indices>
        requires (sizeof...(Indices) == N) &&
                 (std::is_convertible_v<Indices, index_t> && ...)
    constexpr nd_index(Indices... idx) noexcept
        : indices{static_cast<index_t>(idx)...} {}

    /// @brief Number of dimensions
    [[nodiscard]] static constexpr size_type rank() noexcept {
        return N;
    }

    /// @brief Access index at dimension d
    [[nodiscard]] constexpr index_t operator[](size_type d) const noexcept {
        return indices[d];
    }

    /// @brief Access index at dimension d (mutable)
    [[nodiscard]] constexpr index_t& operator[](size_type d) noexcept {
        return indices[d];
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const nd_index& other) const noexcept {
        return indices == other.indices;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const nd_index& other) const noexcept {
        return indices != other.indices;
    }
};

/// @brief Deduction guide for nd_index
template <typename... Indices>
nd_index(Indices...) -> nd_index<sizeof...(Indices)>;

/// @brief N-dimensional extent (dimensions)
/// @tparam N Number of dimensions
template <size_type N>
using nd_extent = std::array<size_type, N>;

// =============================================================================
// Span-like Utility Types
// =============================================================================

/// @brief Simple non-owning view of contiguous elements
/// @tparam T Element type
/// @note Placeholder for std::span compatibility
template <typename T>
struct span {
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = dtl::size_type;
    using difference_type = dtl::difference_type;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    /// @brief Default constructor (empty span)
    constexpr span() noexcept = default;

    /// @brief Construct from pointer and size
    constexpr span(pointer ptr, size_type count) noexcept
        : data_{ptr}, size_{count} {}

    /// @brief Construct from two pointers
    constexpr span(pointer first, pointer last) noexcept
        : data_{first}, size_{static_cast<size_type>(last - first)} {}

    /// @brief Get pointer to data
    [[nodiscard]] constexpr pointer data() const noexcept { return data_; }

    /// @brief Get number of elements
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }

    /// @brief Check if empty
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    /// @brief Get element at index
    [[nodiscard]] constexpr reference operator[](size_type idx) const noexcept {
        return data_[idx];
    }

    /// @brief Get iterator to beginning
    [[nodiscard]] constexpr iterator begin() const noexcept { return data_; }

    /// @brief Get iterator to end
    [[nodiscard]] constexpr iterator end() const noexcept { return data_ + size_; }

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
};

// =============================================================================
// Byte Span Aliases (for serialization)
// =============================================================================

/// @brief Non-owning view over bytes (for serialization output)
using byte_span = span<std::byte>;

/// @brief Non-owning view over const bytes (for serialization input)
using const_byte_span = span<const std::byte>;

}  // namespace dtl
