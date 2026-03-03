// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file window_view.hpp
/// @brief Sliding window view for stencil operations
/// @details Provides sliding windows over a range for stencil computations,
///          moving averages, convolutions, and time-series analysis.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <iterator>
#include <ranges>

namespace dtl {

// ============================================================================
// Window Type
// ============================================================================

/// @brief A single window from a window_view
/// @tparam Range The underlying range type
template <typename Range>
class window {
public:
    /// @brief Value type
    using value_type = std::ranges::range_value_t<Range>;

    /// @brief Iterator type
    using iterator = std::ranges::iterator_t<Range>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct a window from iterator and size
    constexpr window(iterator begin, size_type size) noexcept
        : begin_{begin}
        , size_{size} {}

    /// @brief Get iterator to beginning
    [[nodiscard]] constexpr iterator begin() const noexcept { return begin_; }

    /// @brief Get iterator past end
    [[nodiscard]] constexpr iterator end() const noexcept {
        auto it = begin_;
        std::advance(it, static_cast<std::ptrdiff_t>(size_));
        return it;
    }

    /// @brief Get window size
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }

    /// @brief Check if window is empty
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    /// @brief Access element by index within window
    [[nodiscard]] constexpr decltype(auto) operator[](size_type idx) const {
        auto it = begin_;
        std::advance(it, static_cast<std::ptrdiff_t>(idx));
        return *it;
    }

    /// @brief Get first element
    [[nodiscard]] constexpr decltype(auto) front() const { return *begin_; }

    /// @brief Get last element
    [[nodiscard]] constexpr decltype(auto) back() const {
        return (*this)[size_ - 1];
    }

    /// @brief Get center element (for odd-sized windows)
    [[nodiscard]] constexpr decltype(auto) center() const {
        return (*this)[size_ / 2];
    }

private:
    iterator begin_;
    size_type size_;
};

// ============================================================================
// Window Iterator
// ============================================================================

/// @brief Iterator over windows in a window_view
/// @tparam Range The underlying range type
template <typename Range>
class window_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = window<Range>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

    /// @brief Construct window iterator
    constexpr window_iterator(std::ranges::iterator_t<Range> current,
                              std::ranges::iterator_t<Range> end,
                              size_type window_size,
                              size_type stride) noexcept
        : current_{current}
        , end_{end}
        , window_size_{window_size}
        , stride_{stride} {}

    /// @brief Dereference to get current window
    [[nodiscard]] constexpr value_type operator*() const {
        return value_type{current_, window_size_};
    }

    /// @brief Pre-increment
    constexpr window_iterator& operator++() {
        std::advance(current_, static_cast<std::ptrdiff_t>(stride_));
        return *this;
    }

    /// @brief Post-increment
    constexpr window_iterator operator++(int) {
        window_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const window_iterator& other) const noexcept {
        return current_ == other.current_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const window_iterator& other) const noexcept {
        return !(*this == other);
    }

private:
    std::ranges::iterator_t<Range> current_;
    std::ranges::iterator_t<Range> end_;
    size_type window_size_;
    size_type stride_;
};

// ============================================================================
// Window View
// ============================================================================

/// @brief View that provides sliding windows over a range
/// @tparam Range The underlying range type
///
/// @par Design Rationale:
/// Sliding windows enable:
/// - Stencil computations (halo access patterns)
/// - Moving average calculations
/// - Convolution operations
/// - Time-series analysis
///
/// @par Usage:
/// @code
/// std::vector<int> data = {1, 2, 3, 4, 5};
/// // Sliding window of size 3
/// for (auto win : window_view(data, 3)) {
///     // win contains [1,2,3], then [2,3,4], then [3,4,5]
///     auto sum = std::accumulate(win.begin(), win.end(), 0);
/// }
/// @endcode
///
/// @par With stride:
/// @code
/// // Window size 3, stride 2 (non-overlapping)
/// for (auto win : window_view(data, 3, 2)) {
///     // win contains [1,2,3], then [3,4,5]
/// }
/// @endcode
///
/// @par Stencil Example:
/// @code
/// // 3-point stencil computation
/// for (auto win : window_view(data, 3)) {
///     float result = 0.25f * win[0] + 0.5f * win[1] + 0.25f * win[2];
///     // ...
/// }
/// @endcode
template <typename Range>
class window_view {
public:
    /// @brief Value type (window)
    using value_type = window<Range>;

    /// @brief Iterator type
    using iterator = window_iterator<Range>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Construct with window size (stride = 1)
    /// @param range The range to window over
    /// @param window_size Number of elements in each window
    constexpr window_view(Range& range, size_type window_size) noexcept
        : range_{&range}
        , window_size_{window_size}
        , stride_{1} {}

    /// @brief Construct with window size and stride
    /// @param range The range to window over
    /// @param window_size Number of elements in each window
    /// @param stride Number of elements to advance between windows
    constexpr window_view(Range& range, size_type window_size, size_type stride) noexcept
        : range_{&range}
        , window_size_{window_size}
        , stride_{stride} {}

    /// @brief Get iterator to first window
    [[nodiscard]] constexpr iterator begin() noexcept {
        auto total = static_cast<size_type>(std::ranges::size(*range_));
        if (total < window_size_) {
            // No windows possible - return end iterator
            return end();
        }
        return iterator{std::ranges::begin(*range_), compute_end_position(), window_size_, stride_};
    }

    /// @brief Get iterator to first window (const)
    [[nodiscard]] constexpr iterator begin() const noexcept {
        auto total = static_cast<size_type>(std::ranges::size(*range_));
        if (total < window_size_) {
            // No windows possible - return end iterator
            return end();
        }
        return iterator{std::ranges::begin(*range_), compute_end_position(), window_size_, stride_};
    }

    /// @brief Get iterator past last window
    [[nodiscard]] constexpr iterator end() noexcept {
        return iterator{compute_end_position(), compute_end_position(), window_size_, stride_};
    }

    /// @brief Get iterator past last window (const)
    [[nodiscard]] constexpr iterator end() const noexcept {
        return iterator{compute_end_position(), compute_end_position(), window_size_, stride_};
    }

    /// @brief Get window size
    [[nodiscard]] constexpr size_type window_size() const noexcept {
        return window_size_;
    }

    /// @brief Get stride
    [[nodiscard]] constexpr size_type stride() const noexcept {
        return stride_;
    }

    /// @brief Get number of windows
    [[nodiscard]] constexpr size_type num_windows() const noexcept {
        auto total = static_cast<size_type>(std::ranges::size(*range_));
        if (total < window_size_) return 0;
        return (total - window_size_) / stride_ + 1;
    }

private:
    /// @brief Compute end position for iteration
    std::ranges::iterator_t<Range> compute_end_position() const {
        auto total = static_cast<size_type>(std::ranges::size(*range_));
        if (total < window_size_) {
            return std::ranges::end(*range_);
        }

        auto num_windows = (total - window_size_) / stride_ + 1;
        auto end_offset = (num_windows - 1) * stride_ + stride_;

        auto it = std::ranges::begin(*range_);
        std::advance(it, static_cast<std::ptrdiff_t>(std::min(end_offset, total)));
        return it;
    }

    Range* range_;
    size_type window_size_;
    size_type stride_;
};

// Deduction guides
template <typename Range>
window_view(Range&, size_type) -> window_view<Range>;

template <typename Range>
window_view(Range&, size_type, size_type) -> window_view<Range>;

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a window view with stride 1
/// @tparam Range The range type
/// @param range The range to window over
/// @param window_size Number of elements per window
template <typename Range>
[[nodiscard]] constexpr auto make_window_view(Range& range, size_type window_size) {
    return window_view<Range>{range, window_size};
}

/// @brief Create a window view with custom stride
/// @tparam Range The range type
/// @param range The range to window over
/// @param window_size Number of elements per window
/// @param stride Number of elements to advance between windows
template <typename Range>
[[nodiscard]] constexpr auto make_window_view(Range& range, size_type window_size, size_type stride) {
    return window_view<Range>{range, window_size, stride};
}

/// @brief Pipe operator for window view (stride 1)
inline constexpr auto windowed(size_type window_size) {
    return [window_size]<typename Range>(Range& range) {
        return window_view<Range>{range, window_size};
    };
}

/// @brief Pipe operator for window view with stride
inline constexpr auto windowed(size_type window_size, size_type stride) {
    return [window_size, stride]<typename Range>(Range& range) {
        return window_view<Range>{range, window_size, stride};
    };
}

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Check if a type is a window_view
template <typename T>
struct is_window_view : std::false_type {};

template <typename Range>
struct is_window_view<window_view<Range>> : std::true_type {};

template <typename T>
inline constexpr bool is_window_view_v = is_window_view<T>::value;

}  // namespace dtl
