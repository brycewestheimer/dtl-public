// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file local_index.hpp
/// @brief Local index type and operations for distributed containers
/// @details Provides strongly-typed local indexing within a rank's partition.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/index/global_index.hpp>

#include <compare>
#include <functional>
#include <limits>
#include <type_traits>

namespace dtl {

// ============================================================================
// Local Index
// ============================================================================

/// @brief Strongly-typed local index within a rank's partition
/// @tparam T Underlying index type (default: index_t)
/// @details Local indices represent positions within the local portion of a
///          distributed container owned by a single rank. They are always
///          zero-based and independent of the partition's position in the
///          global address space.
template <typename T>
class local_index {
public:
    static_assert(std::is_integral_v<T>, "Index type must be integral");

    using value_type = T;

    /// @brief Invalid index sentinel
    static constexpr T invalid_value = std::numeric_limits<T>::max();

    /// @brief Default constructor (invalid index)
    constexpr local_index() noexcept : value_(invalid_value) {}

    /// @brief Construct from raw value
    /// @param value The local index value
    constexpr explicit local_index(T value) noexcept : value_(value) {}

    /// @brief Get the raw value
    [[nodiscard]] constexpr T value() const noexcept { return value_; }

    /// @brief Implicit conversion to raw value
    [[nodiscard]] constexpr operator T() const noexcept { return value_; }

    /// @brief Check if index is valid
    [[nodiscard]] constexpr bool valid() const noexcept {
        return value_ != invalid_value;
    }

    /// @brief Boolean conversion (true if valid)
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return valid();
    }

    // ------------------------------------------------------------------------
    // Arithmetic Operations
    // ------------------------------------------------------------------------

    /// @brief Pre-increment
    constexpr local_index& operator++() noexcept {
        ++value_;
        return *this;
    }

    /// @brief Post-increment
    constexpr local_index operator++(int) noexcept {
        local_index tmp = *this;
        ++value_;
        return tmp;
    }

    /// @brief Pre-decrement
    constexpr local_index& operator--() noexcept {
        --value_;
        return *this;
    }

    /// @brief Post-decrement
    constexpr local_index operator--(int) noexcept {
        local_index tmp = *this;
        --value_;
        return tmp;
    }

    /// @brief Add offset
    constexpr local_index& operator+=(T offset) noexcept {
        value_ += offset;
        return *this;
    }

    /// @brief Subtract offset
    constexpr local_index& operator-=(T offset) noexcept {
        value_ -= offset;
        return *this;
    }

    /// @brief Addition
    [[nodiscard]] constexpr local_index operator+(T offset) const noexcept {
        return local_index(value_ + offset);
    }

    /// @brief Subtraction
    [[nodiscard]] constexpr local_index operator-(T offset) const noexcept {
        return local_index(value_ - offset);
    }

    /// @brief Difference between indices
    [[nodiscard]] constexpr T operator-(const local_index& other) const noexcept {
        return value_ - other.value_;
    }

    // ------------------------------------------------------------------------
    // Comparison Operations
    // ------------------------------------------------------------------------

    [[nodiscard]] constexpr auto operator<=>(const local_index&) const noexcept = default;
    [[nodiscard]] constexpr bool operator==(const local_index&) const noexcept = default;

private:
    T value_;
};

// ============================================================================
// Local Index Factories
// ============================================================================

/// @brief Create a local index from a raw value
/// @tparam T Index type
/// @param value Raw index value
/// @return Local index
template <typename T = index_t>
[[nodiscard]] constexpr local_index<T> make_local_index(T value) noexcept {
    return local_index<T>(value);
}

/// @brief Create an invalid local index
/// @tparam T Index type
/// @return Invalid local index
template <typename T = index_t>
[[nodiscard]] constexpr local_index<T> invalid_local_index() noexcept {
    return local_index<T>();
}

// ============================================================================
// Multi-Dimensional Local Index
// ============================================================================

/// @brief Multi-dimensional local index
/// @tparam N Number of dimensions
/// @tparam T Underlying index type
template <size_type N, typename T = index_t>
class md_local_index {
public:
    using value_type = T;
    static constexpr size_type rank = N;

    /// @brief Default constructor (all zeros)
    constexpr md_local_index() noexcept : indices_{} {}

    /// @brief Construct from individual indices
    template <typename... Indices>
        requires (sizeof...(Indices) == N && (std::is_convertible_v<Indices, T> && ...))
    constexpr md_local_index(Indices... indices) noexcept
        : indices_{static_cast<T>(indices)...} {}

    /// @brief Access index for dimension d
    [[nodiscard]] constexpr T operator[](size_type d) const noexcept {
        return indices_[d];
    }

    /// @brief Access index for dimension d (mutable)
    [[nodiscard]] constexpr T& operator[](size_type d) noexcept {
        return indices_[d];
    }

    /// @brief Get pointer to underlying array
    [[nodiscard]] constexpr const T* data() const noexcept {
        return indices_;
    }

    /// @brief Get pointer to underlying array (mutable)
    [[nodiscard]] constexpr T* data() noexcept {
        return indices_;
    }

    /// @brief Linearize to 1D index given local extents
    /// @param extents Array of local extents for each dimension
    /// @return Linear local index
    [[nodiscard]] constexpr T linearize(const T* extents) const noexcept {
        T result = 0;
        T stride = 1;
        for (size_type d = N; d > 0; --d) {
            result += indices_[d - 1] * stride;
            stride *= extents[d - 1];
        }
        return result;
    }

    /// @brief Create from linear index given local extents
    /// @param linear Linear index
    /// @param extents Array of local extents for each dimension
    /// @return Multi-dimensional local index
    [[nodiscard]] static constexpr md_local_index from_linear(T linear, const T* extents) noexcept {
        md_local_index result;
        for (size_type d = N; d > 0; --d) {
            result.indices_[d - 1] = linear % extents[d - 1];
            linear /= extents[d - 1];
        }
        return result;
    }

    // Comparison
    [[nodiscard]] constexpr auto operator<=>(const md_local_index&) const noexcept = default;
    [[nodiscard]] constexpr bool operator==(const md_local_index&) const noexcept = default;

private:
    T indices_[N];
};

// ============================================================================
// Local Index Range
// ============================================================================

/// @brief Range of local indices
/// @tparam T Index type
template <typename T = index_t>
class local_index_range {
public:
    using index_type = local_index<T>;

    /// @brief Construct from begin and end
    /// @param begin First index (inclusive)
    /// @param end Last index (exclusive)
    constexpr local_index_range(index_type begin, index_type end) noexcept
        : begin_(begin), end_(end) {}

    /// @brief Construct from raw values
    constexpr local_index_range(T begin, T end) noexcept
        : begin_(begin), end_(end) {}

    /// @brief Get beginning index of range
    [[nodiscard]] constexpr index_type start() const noexcept { return begin_; }

    /// @brief Get end index of range
    [[nodiscard]] constexpr index_type stop() const noexcept { return end_; }

    /// @brief Get range size
    [[nodiscard]] constexpr T size() const noexcept {
        return end_.value() - begin_.value();
    }

    /// @brief Check if range is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return begin_ >= end_;
    }

    /// @brief Check if index is in range
    [[nodiscard]] constexpr bool contains(index_type idx) const noexcept {
        return idx >= begin_ && idx < end_;
    }

    /// @brief Check if index value is in range
    [[nodiscard]] constexpr bool contains(T value) const noexcept {
        return value >= begin_.value() && value < end_.value();
    }

    // Iterator support for range-based for
    class iterator {
    public:
        using value_type = index_type;
        using difference_type = T;
        using reference = index_type;
        using iterator_category = std::random_access_iterator_tag;

        constexpr iterator() noexcept = default;
        constexpr explicit iterator(index_type idx) noexcept : current_(idx) {}

        [[nodiscard]] constexpr index_type operator*() const noexcept { return current_; }
        constexpr iterator& operator++() noexcept { ++current_; return *this; }
        constexpr iterator operator++(int) noexcept { auto tmp = *this; ++current_; return tmp; }
        constexpr iterator& operator--() noexcept { --current_; return *this; }
        constexpr iterator operator--(int) noexcept { auto tmp = *this; --current_; return tmp; }
        constexpr iterator& operator+=(difference_type n) noexcept { current_ += n; return *this; }
        constexpr iterator& operator-=(difference_type n) noexcept { current_ -= n; return *this; }
        [[nodiscard]] constexpr iterator operator+(difference_type n) const noexcept { return iterator(current_ + n); }
        [[nodiscard]] constexpr iterator operator-(difference_type n) const noexcept { return iterator(current_ - n); }
        [[nodiscard]] constexpr difference_type operator-(const iterator& other) const noexcept { return current_ - other.current_; }
        [[nodiscard]] constexpr index_type operator[](difference_type n) const noexcept { return current_ + n; }
        [[nodiscard]] constexpr auto operator<=>(const iterator&) const noexcept = default;
        [[nodiscard]] constexpr bool operator==(const iterator&) const noexcept = default;

    private:
        index_type current_;
    };

    [[nodiscard]] constexpr iterator begin() const noexcept { return iterator(begin_); }
    [[nodiscard]] constexpr iterator end() const noexcept { return iterator(end_); }

private:
    index_type begin_;
    index_type end_;
};

/// @brief Create a local index range
template <typename T = index_t>
[[nodiscard]] constexpr local_index_range<T>
make_local_range(local_index<T> begin, local_index<T> end) noexcept {
    return local_index_range<T>(begin, end);
}

/// @brief Create a local index range from raw values
template <typename T = index_t>
[[nodiscard]] constexpr local_index_range<T>
make_local_range(T begin, T end) noexcept {
    return local_index_range<T>(begin, end);
}

// ============================================================================
// Ranked Local Index
// ============================================================================

/// @brief Local index with associated rank information
/// @tparam T Index type
template <typename T = index_t>
struct ranked_local_index {
    /// @brief The local index
    local_index<T> index;

    /// @brief The rank that owns this index
    rank_t rank = no_rank;

    /// @brief Default constructor
    constexpr ranked_local_index() noexcept = default;

    /// @brief Construct with index and rank
    constexpr ranked_local_index(local_index<T> idx, rank_t r) noexcept
        : index(idx), rank(r) {}

    /// @brief Construct with raw index and rank
    constexpr ranked_local_index(T idx, rank_t r) noexcept
        : index(idx), rank(r) {}

    /// @brief Check if valid
    [[nodiscard]] constexpr bool valid() const noexcept {
        return index.valid() && rank != no_rank;
    }

    /// @brief Boolean conversion
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return valid();
    }

    // Comparison
    [[nodiscard]] constexpr auto operator<=>(const ranked_local_index&) const noexcept = default;
    [[nodiscard]] constexpr bool operator==(const ranked_local_index&) const noexcept = default;
};

/// @brief Create a ranked local index
template <typename T = index_t>
[[nodiscard]] constexpr ranked_local_index<T>
make_ranked_local_index(local_index<T> index, rank_t rank) noexcept {
    return ranked_local_index<T>(index, rank);
}

/// @brief Create a ranked local index from raw value
template <typename T = index_t>
[[nodiscard]] constexpr ranked_local_index<T>
make_ranked_local_index(T index, rank_t rank) noexcept {
    return ranked_local_index<T>(index, rank);
}

}  // namespace dtl

// ============================================================================
// Standard Library Specializations
// ============================================================================

namespace std {

/// @brief Hash specialization for local_index
template <typename T>
struct hash<dtl::local_index<T>> {
    [[nodiscard]] constexpr size_t operator()(const dtl::local_index<T>& idx) const noexcept {
        return hash<T>{}(idx.value());
    }
};

/// @brief Hash specialization for ranked_local_index
template <typename T>
struct hash<dtl::ranked_local_index<T>> {
    [[nodiscard]] constexpr size_t operator()(const dtl::ranked_local_index<T>& idx) const noexcept {
        size_t h1 = hash<T>{}(idx.index.value());
        size_t h2 = hash<dtl::rank_t>{}(idx.rank);
        return h1 ^ (h2 << 1);
    }
};

/// @brief Numeric limits for local_index
template <typename T>
struct numeric_limits<dtl::local_index<T>> : numeric_limits<T> {
    [[nodiscard]] static constexpr dtl::local_index<T> max() noexcept {
        return dtl::local_index<T>(numeric_limits<T>::max() - 1);
    }
    [[nodiscard]] static constexpr dtl::local_index<T> min() noexcept {
        return dtl::local_index<T>(numeric_limits<T>::min());
    }
};

}  // namespace std
