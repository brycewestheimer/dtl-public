// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file local_view.hpp
/// @brief Local view providing access to local partition only
/// @details Never communicates - safe for STL algorithm use.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <iterator>
#include <span>
#include <type_traits>

namespace dtl {

// Forward declarations
template <typename T, typename... Policies>
class distributed_vector;

template <typename T, size_type Rank, typename... Policies>
class distributed_tensor;

/// @brief Standalone local view over contiguous data
/// @tparam T The element type
/// @details A lightweight view that provides STL-compatible access to
///          contiguous local data. This is the non-owning variant that
///          can be constructed from raw pointers or spans.
///
/// @par Guarantees:
/// - No communication ever occurs through this view
/// - Direct memory access to local elements
/// - STL-compatible iterators (random access, contiguous)
/// - Thread-safe for concurrent reads (same as underlying storage)
///
/// @par Usage:
/// @code
/// std::vector<int> data = {1, 2, 3, 4, 5};
/// local_view<int> view(data.data(), data.size());
/// // Use with STL algorithms - no communication
/// std::sort(view.begin(), view.end());
/// @endcode
template <typename T>
class local_view {
public:
    /// @brief Value type of the view
    using value_type = std::remove_cv_t<T>;

    /// @brief Element type (may be const)
    using element_type = T;

    /// @brief Reference type (direct reference, never remote_ref)
    using reference = T&;

    /// @brief Const reference type
    using const_reference = const T&;

    /// @brief Pointer type
    using pointer = T*;

    /// @brief Const pointer type
    using const_pointer = const T*;

    /// @brief Iterator type (random access, contiguous)
    using iterator = pointer;

    /// @brief Const iterator type
    using const_iterator = const_pointer;

    /// @brief Reverse iterator type
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// @brief Const reverse iterator type
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Default constructor (empty view)
    constexpr local_view() noexcept = default;

    /// @brief Construct from pointer and size
    /// @param data Pointer to the first element
    /// @param size Number of elements
    constexpr local_view(pointer data, size_type size) noexcept
        : data_{data}, size_{size} {}

    /// @brief Construct from pointer and size with metadata
    /// @param data Pointer to the first element
    /// @param size Number of elements
    /// @param rank The owning rank
    /// @param global_offset Global offset of first element
    constexpr local_view(pointer data, size_type size,
                        rank_t rank, index_t global_offset) noexcept
        : data_{data}, size_{size}, rank_{rank}, global_offset_{global_offset} {}

    /// @brief Construct from span
    /// @param span The span to view
    constexpr explicit local_view(std::span<T> span) noexcept
        : data_{span.data()}, size_{static_cast<size_type>(span.size())} {}

    /// @brief Construct from contiguous range
    /// @param range A contiguous range to view
    template <typename Range>
        requires std::ranges::contiguous_range<Range> &&
                 std::same_as<std::ranges::range_value_t<Range>, value_type>
    constexpr explicit local_view(Range& range) noexcept
        : data_{std::ranges::data(range)}
        , size_{static_cast<size_type>(std::ranges::size(range))} {}

    // =========================================================================
    // Iterators
    // =========================================================================

    /// @brief Get iterator to beginning of local partition
    [[nodiscard]] constexpr iterator begin() noexcept {
        return data_;
    }

    /// @brief Get const iterator to beginning of local partition
    [[nodiscard]] constexpr const_iterator begin() const noexcept {
        return data_;
    }

    /// @brief Get const iterator to beginning of local partition
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
        return data_;
    }

    /// @brief Get iterator to end of local partition
    [[nodiscard]] constexpr iterator end() noexcept {
        return data_ + size_;
    }

    /// @brief Get const iterator to end of local partition
    [[nodiscard]] constexpr const_iterator end() const noexcept {
        return data_ + size_;
    }

    /// @brief Get const iterator to end of local partition
    [[nodiscard]] constexpr const_iterator cend() const noexcept {
        return data_ + size_;
    }

    /// @brief Get reverse iterator to reverse beginning
    [[nodiscard]] constexpr reverse_iterator rbegin() noexcept {
        return reverse_iterator{end()};
    }

    /// @brief Get const reverse iterator to reverse beginning
    [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator{end()};
    }

    /// @brief Get const reverse iterator to reverse beginning
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator{cend()};
    }

    /// @brief Get reverse iterator to reverse end
    [[nodiscard]] constexpr reverse_iterator rend() noexcept {
        return reverse_iterator{begin()};
    }

    /// @brief Get const reverse iterator to reverse end
    [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator{begin()};
    }

    /// @brief Get const reverse iterator to reverse end
    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator{cbegin()};
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    /// @brief Access element by local index
    /// @param idx Local index (0-based within this rank's partition)
    /// @return Reference to the element
    [[nodiscard]] constexpr reference operator[](size_type idx) noexcept {
        return data_[idx];
    }

    /// @brief Access element by local index (const)
    [[nodiscard]] constexpr const_reference operator[](size_type idx) const noexcept {
        return data_[idx];
    }

    /// @brief Access element with bounds checking
    /// @param idx Local index
    /// @return Reference to the element
    /// @throws std::out_of_range if idx >= size()
    [[nodiscard]] constexpr reference at(size_type idx) {
        if (idx >= size_) {
            throw std::out_of_range("local_view::at: index out of range");
        }
        return data_[idx];
    }

    /// @brief Access element with bounds checking (const)
    [[nodiscard]] constexpr const_reference at(size_type idx) const {
        if (idx >= size_) {
            throw std::out_of_range("local_view::at: index out of range");
        }
        return data_[idx];
    }

    /// @brief Get first element
    [[nodiscard]] constexpr reference front() noexcept {
        return data_[0];
    }

    /// @brief Get first element (const)
    [[nodiscard]] constexpr const_reference front() const noexcept {
        return data_[0];
    }

    /// @brief Get last element
    [[nodiscard]] constexpr reference back() noexcept {
        return data_[size_ - 1];
    }

    /// @brief Get last element (const)
    [[nodiscard]] constexpr const_reference back() const noexcept {
        return data_[size_ - 1];
    }

    /// @brief Get pointer to underlying local data
    [[nodiscard]] constexpr pointer data() noexcept {
        return data_;
    }

    /// @brief Get const pointer to underlying local data
    [[nodiscard]] constexpr const_pointer data() const noexcept {
        return data_;
    }

    // =========================================================================
    // Capacity
    // =========================================================================

    /// @brief Get number of elements in local partition
    [[nodiscard]] constexpr size_type size() const noexcept {
        return size_;
    }

    /// @brief Get number of elements (alias for size)
    [[nodiscard]] constexpr size_type length() const noexcept {
        return size_;
    }

    /// @brief Check if local partition is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    /// @brief Get size in bytes
    [[nodiscard]] constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(T);
    }

    // =========================================================================
    // Distribution Metadata
    // =========================================================================

    /// @brief Get the rank that owns this local view
    [[nodiscard]] constexpr rank_t rank() const noexcept {
        return rank_;
    }

    /// @brief Get global offset of first local element
    [[nodiscard]] constexpr index_t global_offset() const noexcept {
        return global_offset_;
    }

    /// @brief Convert local index to global index
    [[nodiscard]] constexpr index_t to_global(size_type local_idx) const noexcept {
        return global_offset_ + static_cast<index_t>(local_idx);
    }

    // =========================================================================
    // Subviews
    // =========================================================================

    /// @brief Get a subview of the first n elements
    /// @param count Number of elements
    [[nodiscard]] constexpr local_view first(size_type count) const noexcept {
        return local_view{data_, count < size_ ? count : size_, rank_, global_offset_};
    }

    /// @brief Get a subview of the last n elements
    /// @param count Number of elements
    [[nodiscard]] constexpr local_view last(size_type count) const noexcept {
        size_type actual = count < size_ ? count : size_;
        size_type offset = size_ - actual;
        return local_view{data_ + offset, actual, rank_,
                         global_offset_ + static_cast<index_t>(offset)};
    }

    /// @brief Get a subview starting at offset with count elements
    /// @param offset Start offset
    /// @param count Number of elements
    [[nodiscard]] constexpr local_view subview(size_type offset, size_type count) const noexcept {
        if (offset >= size_) {
            return local_view{};
        }
        size_type actual = (offset + count > size_) ? (size_ - offset) : count;
        return local_view{data_ + offset, actual, rank_,
                         global_offset_ + static_cast<index_t>(offset)};
    }

    // =========================================================================
    // Conversion
    // =========================================================================

    /// @brief Convert to std::span
    [[nodiscard]] constexpr std::span<T> as_span() const noexcept {
        return std::span<T>{data_, size_};
    }

    /// @brief Implicit conversion to std::span
    constexpr operator std::span<T>() const noexcept {
        return as_span();
    }

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
    rank_t rank_ = 0;
    index_t global_offset_ = 0;
};

// =============================================================================
// Type Deduction Guides
// =============================================================================

template <typename T>
local_view(T*, size_type) -> local_view<T>;

template <typename T>
local_view(std::span<T>) -> local_view<T>;

template <std::ranges::contiguous_range R>
local_view(R&) -> local_view<std::ranges::range_value_t<R>>;

// =============================================================================
// Type Trait Specialization
// =============================================================================

template <typename T>
struct is_local_view<local_view<T>> : std::true_type {};

// =============================================================================
// Static Assertions for STL Compatibility
// =============================================================================

static_assert(std::random_access_iterator<local_view<int>::iterator>,
              "local_view iterator must be random access");
static_assert(std::contiguous_iterator<local_view<int>::iterator>,
              "local_view iterator must be contiguous");
static_assert(std::ranges::contiguous_range<local_view<int>>,
              "local_view must be a contiguous range");

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Create a local view from pointer and size
/// @tparam T The element type
/// @param data Pointer to the first element
/// @param size Number of elements
/// @return local_view<T>
template <typename T>
[[nodiscard]] constexpr auto make_local_view(T* data, size_type size) noexcept {
    return local_view<T>{data, size};
}

/// @brief Create a local view from span
/// @tparam T The element type
/// @param span The span to view
/// @return local_view<T>
template <typename T>
[[nodiscard]] constexpr auto make_local_view(std::span<T> span) noexcept {
    return local_view<T>{span};
}

/// @brief Create a local view from a contiguous range
/// @tparam Range A contiguous range type
/// @param range The range to view
/// @return local_view of the range's value type
template <std::ranges::contiguous_range Range>
[[nodiscard]] constexpr auto make_local_view(Range& range) noexcept {
    return local_view<std::ranges::range_value_t<Range>>{range};
}

}  // namespace dtl
