// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_span.hpp
/// @brief Non-owning view over distributed memory
/// @details Similar to std::span but for distributed data.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/error/result.hpp>

namespace dtl {

// Forward declarations
template <typename Container>
class local_view;

template <typename Container>
class global_view;

/// @brief Non-owning view over distributed memory
/// @tparam T Element type
/// @tparam Extent Static extent (dynamic_extent for runtime size)
/// @details Provides a lightweight, non-owning view over distributed data.
///          Similar to std::span but understands distributed partitioning.
///
/// @par Use Cases:
/// - Passing distributed data to functions without copying
/// - Viewing subsets of distributed containers
/// - Interfacing with external distributed data
///
/// @par Important:
/// The span does not own the data and does not manage its lifetime.
/// The underlying data must remain valid for the span's lifetime.
template <typename T, size_type Extent>
class distributed_span {
public:
    // ========================================================================
    // Type Aliases
    // ========================================================================

    /// @brief Element type
    using element_type = T;

    /// @brief Value type (without const)
    using value_type = std::remove_cv_t<T>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    /// @brief Pointer type
    using pointer = T*;

    /// @brief Const pointer type
    using const_pointer = const T*;

    /// @brief Reference type
    using reference = T&;

    /// @brief Const reference type
    using const_reference = const T&;

    /// @brief Iterator type (pointer, for contiguous range compliance)
    using iterator = T*;

    /// @brief Const iterator type
    using const_iterator = const T*;

    /// @brief Static extent
    static constexpr size_type extent = Extent;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (empty span)
    constexpr distributed_span() noexcept = default;

    /// @brief Construct from distributed container
    /// @tparam Container Distributed container type
    /// @param container The container to span
    template <typename Container>
    explicit distributed_span(Container& container)
        : global_size_{container.size()}
        , local_data_{container.local_data()}
        , local_size_{container.local_size()}
        , my_rank_{container.rank()}
        , num_ranks_{container.num_ranks()} {}

    /// @brief Construct from pointer and sizes (single-rank mode)
    /// @param data Pointer to local data
    /// @param local_size Local element count
    /// @param global_size Global element count
    distributed_span(pointer data, size_type local_size, size_type global_size)
        : global_size_{global_size}
        , local_data_{data}
        , local_size_{local_size}
        , my_rank_{0}
        , num_ranks_{1} {}

    /// @brief Construct from pointer, sizes, and rank info
    /// @param data Pointer to local data
    /// @param local_size Local element count
    /// @param global_size Global element count
    /// @param my_rank This rank's ID
    /// @param num_ranks Total number of ranks
    distributed_span(pointer data, size_type local_size, size_type global_size,
                     rank_t my_rank, rank_t num_ranks)
        : global_size_{global_size}
        , local_data_{data}
        , local_size_{local_size}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks} {}

    // ========================================================================
    // Size Queries
    // ========================================================================

    /// @brief Get global size
    [[nodiscard]] constexpr size_type size() const noexcept {
        return global_size_;
    }

    /// @brief Get local size
    [[nodiscard]] constexpr size_type local_size() const noexcept {
        return local_size_;
    }

    /// @brief Get size in bytes (local)
    [[nodiscard]] constexpr size_type size_bytes() const noexcept {
        return local_size_ * sizeof(T);
    }

    /// @brief Check if globally empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return global_size_ == 0;
    }

    // ========================================================================
    // Element Access
    // ========================================================================

    /// @brief Get pointer to local data
    [[nodiscard]] constexpr pointer data() const noexcept {
        return local_data_;
    }

    /// @brief Access local element
    [[nodiscard]] constexpr reference operator[](size_type idx) const {
        return local_data_[idx];
    }

    /// @brief Access first local element
    [[nodiscard]] constexpr reference front() const {
        return local_data_[0];
    }

    /// @brief Access last local element
    [[nodiscard]] constexpr reference back() const {
        return local_data_[local_size_ - 1];
    }

    // ========================================================================
    // Iterators (Local Only)
    // ========================================================================

    /// @brief Get iterator to beginning of local data
    [[nodiscard]] constexpr pointer begin() const noexcept {
        return local_data_;
    }

    /// @brief Get iterator to end of local data
    [[nodiscard]] constexpr pointer end() const noexcept {
        return local_data_ + local_size_;
    }

    // ========================================================================
    // Subspans
    // ========================================================================

    /// @brief Get first N elements (local)
    /// @param count Number of elements
    [[nodiscard]] constexpr distributed_span<T, dynamic_extent>
    first(size_type count) const {
        return {local_data_, count, count};
    }

    /// @brief Get last N elements (local)
    /// @param count Number of elements
    [[nodiscard]] constexpr distributed_span<T, dynamic_extent>
    last(size_type count) const {
        return {local_data_ + local_size_ - count, count, count};
    }

    /// @brief Get subspan (local)
    /// @param offset Starting offset
    /// @param count Number of elements (dynamic_extent = to end)
    [[nodiscard]] constexpr distributed_span<T, dynamic_extent>
    subspan(size_type offset, size_type count = dynamic_extent) const {
        if (count == dynamic_extent) {
            count = local_size_ - offset;
        }
        return {local_data_ + offset, count, count};
    }

    // ========================================================================
    // Distribution Queries
    // ========================================================================

    /// @brief Get number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get current rank
    [[nodiscard]] rank_t rank() const noexcept {
        return my_rank_;
    }

private:
    size_type global_size_ = 0;
    pointer local_data_ = nullptr;
    size_type local_size_ = 0;
    rank_t my_rank_ = 0;
    rank_t num_ranks_ = 1;
};

/// @brief Create distributed span from container
/// @tparam Container Distributed container type
/// @param container The container to span
template <typename Container>
[[nodiscard]] auto make_distributed_span(Container& container) {
    return distributed_span<typename Container::value_type>{container};
}

/// @brief Create distributed span from const container
template <typename Container>
[[nodiscard]] auto make_distributed_span(const Container& container) {
    return distributed_span<const typename Container::value_type>{container};
}

// =============================================================================
// Type Trait Specializations
// =============================================================================

template <typename T, size_type Extent>
struct is_distributed_container<distributed_span<T, Extent>> : std::true_type {};

template <typename T, size_type Extent>
struct is_distributed_span<distributed_span<T, Extent>> : std::true_type {};

}  // namespace dtl
