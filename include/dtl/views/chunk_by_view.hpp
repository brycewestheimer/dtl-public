// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file chunk_by_view.hpp
/// @brief Predicate-based chunking view
/// @details Groups consecutive elements based on a binary predicate,
///          enabling adaptive work unit sizing based on data properties.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <iterator>
#include <ranges>
#include <functional>

namespace dtl {

// ============================================================================
// Chunk By Type
// ============================================================================

/// @brief A single group from a chunk_by_view
/// @tparam Range The underlying range type
template <typename Range>
class predicate_chunk {
public:
    /// @brief Value type
    using value_type = std::ranges::range_value_t<Range>;

    /// @brief Iterator type
    using iterator = std::ranges::iterator_t<Range>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct a chunk from iterators
    constexpr predicate_chunk(iterator begin, iterator end) noexcept
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

    /// @brief Get first element (representative of the group)
    [[nodiscard]] constexpr decltype(auto) front() const {
        return *begin_;
    }

private:
    iterator begin_;
    iterator end_;
};

// ============================================================================
// Chunk By Iterator
// ============================================================================

/// @brief Iterator over groups in a chunk_by_view
/// @tparam Range The underlying range type
/// @tparam Predicate Binary predicate type
template <typename Range, typename Predicate>
class chunk_by_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = predicate_chunk<Range>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    /// @brief Construct chunk_by iterator
    chunk_by_iterator(std::ranges::iterator_t<Range> current,
                      std::ranges::iterator_t<Range> end,
                      Predicate pred)
        : current_{current}
        , end_{end}
        , pred_{pred} {}

    /// @brief Dereference to get current chunk
    [[nodiscard]] value_type operator*() const {
        auto chunk_end = find_chunk_end();
        return value_type{current_, chunk_end};
    }

    /// @brief Pre-increment
    chunk_by_iterator& operator++() {
        current_ = find_chunk_end();
        return *this;
    }

    /// @brief Post-increment
    chunk_by_iterator operator++(int) {
        chunk_by_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Equality comparison
    [[nodiscard]] bool operator==(const chunk_by_iterator& other) const noexcept {
        return current_ == other.current_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] bool operator!=(const chunk_by_iterator& other) const noexcept {
        return !(*this == other);
    }

private:
    /// @brief Find end of current chunk (where predicate becomes false)
    std::ranges::iterator_t<Range> find_chunk_end() const {
        if (current_ == end_) return end_;

        auto it = current_;
        auto prev = it;
        ++it;

        while (it != end_) {
            if (!std::invoke(pred_, *prev, *it)) {
                break;
            }
            prev = it;
            ++it;
        }

        return it;
    }

    std::ranges::iterator_t<Range> current_;
    std::ranges::iterator_t<Range> end_;
    Predicate pred_;
};

// ============================================================================
// Chunk By View
// ============================================================================

/// @brief View that chunks a range based on a predicate
/// @tparam Range The underlying range type
/// @tparam Predicate Binary predicate for grouping
///
/// @par Design Rationale:
/// Predicate-based chunking enables:
/// - Grouping by key for group-by operations
/// - Adaptive work unit sizing based on data properties
/// - Processing runs of similar elements together
///
/// @par Usage:
/// @code
/// std::vector<int> data = {1, 1, 2, 2, 2, 3, 3};
/// // Chunk by equality (groups runs of equal elements)
/// for (auto chunk : chunk_by_view(data, std::equal_to<>{})) {
///     // chunk contains [1,1], then [2,2,2], then [3,3]
/// }
/// @endcode
///
/// @par Custom Predicates:
/// @code
/// // Group by key extraction
/// struct Record { int key; std::string value; };
/// std::vector<Record> records = ...;
/// for (auto group : chunk_by_view(records, [](auto& a, auto& b) {
///     return a.key == b.key;
/// })) {
///     // Process each group with same key
/// }
/// @endcode
template <typename Range, typename Predicate>
class chunk_by_view {
public:
    /// @brief Value type (chunk)
    using value_type = predicate_chunk<Range>;

    /// @brief Iterator type
    using iterator = chunk_by_iterator<Range, Predicate>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct from range and predicate
    /// @param range The range to chunk
    /// @param pred Binary predicate: returns true if two consecutive
    ///             elements should be in the same chunk
    constexpr chunk_by_view(Range& range, Predicate pred) noexcept
        : range_{&range}
        , pred_{pred} {}

    /// @brief Get iterator to first chunk
    [[nodiscard]] iterator begin() noexcept {
        return iterator{std::ranges::begin(*range_), std::ranges::end(*range_), pred_};
    }

    /// @brief Get iterator to first chunk (const)
    [[nodiscard]] iterator begin() const noexcept {
        return iterator{std::ranges::begin(*range_), std::ranges::end(*range_), pred_};
    }

    /// @brief Get iterator past last chunk
    [[nodiscard]] iterator end() noexcept {
        return iterator{std::ranges::end(*range_), std::ranges::end(*range_), pred_};
    }

    /// @brief Get iterator past last chunk (const)
    [[nodiscard]] iterator end() const noexcept {
        return iterator{std::ranges::end(*range_), std::ranges::end(*range_), pred_};
    }

private:
    Range* range_;
    Predicate pred_;
};

// Deduction guides
template <typename Range, typename Predicate>
chunk_by_view(Range&, Predicate) -> chunk_by_view<Range, Predicate>;

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a chunk_by view from a range and predicate
/// @tparam Range The range type
/// @tparam Predicate Binary predicate type
/// @param range The range to chunk
/// @param pred Predicate for grouping
template <typename Range, typename Predicate>
[[nodiscard]] constexpr auto make_chunk_by_view(Range& range, Predicate pred) {
    return chunk_by_view<Range, Predicate>{range, pred};
}

/// @brief Create a chunk_by view that groups equal elements
/// @tparam Range The range type
/// @param range The range to chunk
template <typename Range>
[[nodiscard]] constexpr auto make_chunk_by_equal(Range& range) {
    return chunk_by_view{range, std::equal_to<>{}};
}

/// @brief Pipe operator for chunk_by view
template <typename Predicate>
constexpr auto chunk_by(Predicate pred) {
    return [pred]<typename Range>(Range& range) {
        return chunk_by_view<Range, Predicate>{range, pred};
    };
}

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Check if a type is a chunk_by_view
template <typename T>
struct is_chunk_by_view : std::false_type {};

template <typename Range, typename Predicate>
struct is_chunk_by_view<chunk_by_view<Range, Predicate>> : std::true_type {};

template <typename T>
inline constexpr bool is_chunk_by_view_v = is_chunk_by_view<T>::value;

}  // namespace dtl
