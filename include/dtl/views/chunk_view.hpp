// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file chunk_view.hpp
/// @brief Fixed-size chunk view for batch processing
/// @details Divides a range into fixed-size contiguous chunks for throughput-oriented
///          patterns. Enables efficient batch processing with predictable work units.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <iterator>
#include <ranges>

namespace dtl {

// ============================================================================
// Chunk Type
// ============================================================================

/// @brief A single chunk from a chunk_view
/// @tparam Range The underlying range type
/// @details Represents a contiguous subsequence of the underlying range.
///          Each chunk has at most N elements (the last chunk may be smaller).
template <typename Range>
class chunk {
public:
    /// @brief Value type
    using value_type = std::ranges::range_value_t<Range>;

    /// @brief Iterator type
    using iterator = std::ranges::iterator_t<Range>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct a chunk from iterators
    /// @param begin Iterator to first element
    /// @param end Iterator past last element
    constexpr chunk(iterator begin, iterator end) noexcept
        : begin_{begin}
        , end_{end} {}

    /// @brief Get iterator to beginning
    [[nodiscard]] constexpr iterator begin() noexcept { return begin_; }

    /// @brief Get const iterator to beginning
    [[nodiscard]] constexpr iterator begin() const noexcept { return begin_; }

    /// @brief Get iterator past end
    [[nodiscard]] constexpr iterator end() noexcept { return end_; }

    /// @brief Get const iterator past end
    [[nodiscard]] constexpr iterator end() const noexcept { return end_; }

    /// @brief Get chunk size
    [[nodiscard]] constexpr size_type size() const noexcept {
        return static_cast<size_type>(std::distance(begin_, end_));
    }

    /// @brief Check if chunk is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return begin_ == end_;
    }

    /// @brief Access element by index
    /// @param idx Index within chunk
    [[nodiscard]] constexpr decltype(auto) operator[](size_type idx) const {
        return *(begin_ + static_cast<std::ptrdiff_t>(idx));
    }

private:
    iterator begin_;
    iterator end_;
};

// ============================================================================
// Chunk Iterator
// ============================================================================

/// @brief Iterator over chunks in a chunk_view
/// @tparam Range The underlying range type
template <typename Range>
class chunk_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = chunk<Range>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    /// @brief Construct chunk iterator
    /// @param current Current position in underlying range
    /// @param end End of underlying range
    /// @param chunk_size Size of each chunk
    constexpr chunk_iterator(std::ranges::iterator_t<Range> current,
                             std::ranges::iterator_t<Range> end,
                             size_type chunk_size) noexcept
        : current_{current}
        , end_{end}
        , chunk_size_{chunk_size} {}

    /// @brief Dereference to get current chunk
    [[nodiscard]] constexpr value_type operator*() const {
        auto chunk_end = current_;
        auto remaining = static_cast<size_type>(std::distance(current_, end_));
        auto actual_size = std::min(chunk_size_, remaining);
        std::advance(chunk_end, static_cast<std::ptrdiff_t>(actual_size));
        return value_type{current_, chunk_end};
    }

    /// @brief Pre-increment
    constexpr chunk_iterator& operator++() {
        auto remaining = static_cast<size_type>(std::distance(current_, end_));
        auto advance_by = std::min(chunk_size_, remaining);
        std::advance(current_, static_cast<std::ptrdiff_t>(advance_by));
        return *this;
    }

    /// @brief Post-increment
    constexpr chunk_iterator operator++(int) {
        chunk_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const chunk_iterator& other) const noexcept {
        return current_ == other.current_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const chunk_iterator& other) const noexcept {
        return !(*this == other);
    }

private:
    std::ranges::iterator_t<Range> current_;
    std::ranges::iterator_t<Range> end_;
    size_type chunk_size_;
};

// ============================================================================
// Chunk View
// ============================================================================

/// @brief View that divides a range into fixed-size chunks
/// @tparam Range The underlying range type
/// @details Provides iteration over contiguous chunks of a range, enabling
///          efficient batch processing patterns.
///
/// @par Design Rationale:
/// Fixed-size chunking enables:
/// - Predictable work unit sizes for load balancing
/// - Vectorization-friendly batch sizes
/// - Reduced scheduling overhead vs element-by-element
/// - Natural fit for GPU workloads (warp/block sizes)
///
/// @par Usage:
/// @code
/// std::vector<int> data(10000);
/// for (auto chunk : chunk_view(data, 256)) {
///     // Process 256 elements at a time
///     for (auto& elem : chunk) {
///         elem *= 2;
///     }
/// }
/// @endcode
template <typename Range>
class chunk_view {
public:
    /// @brief Value type (chunk)
    using value_type = chunk<Range>;

    /// @brief Iterator type
    using iterator = chunk_iterator<Range>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct from range and chunk size
    /// @param range The range to chunk
    /// @param chunk_size Number of elements per chunk
    constexpr chunk_view(Range& range, size_type chunk_size) noexcept
        : range_{&range}
        , chunk_size_{chunk_size} {}

    /// @brief Get iterator to first chunk
    [[nodiscard]] constexpr iterator begin() noexcept {
        return iterator{std::ranges::begin(*range_), std::ranges::end(*range_), chunk_size_};
    }

    /// @brief Get iterator to first chunk (const)
    [[nodiscard]] constexpr iterator begin() const noexcept {
        return iterator{std::ranges::begin(*range_), std::ranges::end(*range_), chunk_size_};
    }

    /// @brief Get iterator past last chunk
    [[nodiscard]] constexpr iterator end() noexcept {
        return iterator{std::ranges::end(*range_), std::ranges::end(*range_), chunk_size_};
    }

    /// @brief Get iterator past last chunk (const)
    [[nodiscard]] constexpr iterator end() const noexcept {
        return iterator{std::ranges::end(*range_), std::ranges::end(*range_), chunk_size_};
    }

    /// @brief Get number of chunks
    [[nodiscard]] constexpr size_type num_chunks() const noexcept {
        auto total = static_cast<size_type>(std::ranges::size(*range_));
        return (total + chunk_size_ - 1) / chunk_size_;
    }

    /// @brief Get chunk size
    [[nodiscard]] constexpr size_type chunk_size() const noexcept {
        return chunk_size_;
    }

    /// @brief Get total element count
    [[nodiscard]] constexpr size_type total_size() const noexcept {
        return static_cast<size_type>(std::ranges::size(*range_));
    }

private:
    Range* range_;
    size_type chunk_size_;
};

// Deduction guides
template <typename Range>
chunk_view(Range&, size_type) -> chunk_view<Range>;

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a chunk view from a range
/// @tparam Range The range type
/// @param range The range to chunk
/// @param chunk_size Number of elements per chunk
/// @return chunk_view<Range>
template <typename Range>
[[nodiscard]] constexpr auto make_chunk_view(Range& range, size_type chunk_size) {
    return chunk_view<Range>{range, chunk_size};
}

/// @brief Pipe operator for chunk view
/// @param chunk_size Number of elements per chunk
inline constexpr auto chunked(size_type chunk_size) {
    return [chunk_size]<typename Range>(Range& range) {
        return chunk_view<Range>{range, chunk_size};
    };
}

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Check if a type is a chunk_view
template <typename T>
struct is_chunk_view : std::false_type {};

template <typename Range>
struct is_chunk_view<chunk_view<Range>> : std::true_type {};

template <typename T>
inline constexpr bool is_chunk_view_v = is_chunk_view<T>::value;

}  // namespace dtl
