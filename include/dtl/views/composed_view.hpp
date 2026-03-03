// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file composed_view.hpp
/// @brief View composition utilities for chaining view transformations
/// @details Provides a pipe operator `|` and `compose()` function for
///          composing strided_view, subview, and other view adapters.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/views/strided_view.hpp>
#include <dtl/views/subview.hpp>

#include <type_traits>

namespace dtl {

// =============================================================================
// View Adapter Tags (for pipe composition)
// =============================================================================

/// @brief Deferred strided view adapter
/// @details Created by `stride(n)` or `stride(n, offset)` and applied
///          to a view via the pipe operator.
struct stride_adapter {
    std::ptrdiff_t stride_val;
    size_type offset_val;

    constexpr stride_adapter(std::ptrdiff_t s, size_type o = 0) noexcept
        : stride_val{s}, offset_val{o} {}
};

/// @brief Deferred subview adapter
/// @details Created by `slice(offset, count)` and applied to a view
///          via the pipe operator.
struct slice_adapter {
    size_type offset_val;
    size_type count_val;

    constexpr slice_adapter(size_type o, size_type c) noexcept
        : offset_val{o}, count_val{c} {}
};

/// @brief Deferred take adapter
/// @details Created by `take_n(count)` and applied to a view.
struct take_adapter {
    size_type count_val;

    constexpr explicit take_adapter(size_type c) noexcept
        : count_val{c} {}
};

/// @brief Deferred drop adapter
/// @details Created by `drop_n(count)` and applied to a view.
struct drop_adapter {
    size_type count_val;

    constexpr explicit drop_adapter(size_type c) noexcept
        : count_val{c} {}
};

// =============================================================================
// Adapter Factory Functions
// =============================================================================

/// @brief Create a strided view adapter for pipe composition
/// @param s The stride (step size)
/// @param offset Starting offset (default 0)
/// @return stride_adapter that can be piped into a view
/// @code
/// auto view = local | stride(2);       // every other element
/// auto view = local | stride(3, 1);    // every 3rd element, starting at 1
/// @endcode
[[nodiscard]] constexpr inline stride_adapter stride(
    std::ptrdiff_t s, size_type offset = 0) noexcept {
    return stride_adapter{s, offset};
}

/// @brief Create a subview adapter for pipe composition
/// @param offset Starting offset within the view
/// @param count Number of elements
/// @return slice_adapter that can be piped into a view
/// @code
/// auto view = local | slice(10, 20);  // elements [10, 30)
/// @endcode
[[nodiscard]] constexpr inline slice_adapter slice(
    size_type offset, size_type count) noexcept {
    return slice_adapter{offset, count};
}

/// @brief Create a take adapter (first N elements) for pipe composition
/// @param count Number of elements from the beginning
/// @return take_adapter that can be piped into a view
/// @code
/// auto view = local | take_n(10);  // first 10 elements
/// @endcode
[[nodiscard]] constexpr inline take_adapter take_n(size_type count) noexcept {
    return take_adapter{count};
}

/// @brief Create a drop adapter (skip first N elements) for pipe composition
/// @param count Number of elements to skip
/// @return drop_adapter that can be piped into a view
/// @code
/// auto view = local | drop_n(5);  // skip first 5 elements
/// @endcode
[[nodiscard]] constexpr inline drop_adapter drop_n(size_type count) noexcept {
    return drop_adapter{count};
}

// =============================================================================
// Pipe Operators (View | Adapter -> Composed View)
// =============================================================================

/// @brief Pipe operator: View | stride_adapter -> strided_view<View>
/// @tparam View Any view type with begin()/end()/size()/operator[]
/// @param view The source view
/// @param adapter The stride adapter
/// @return strided_view wrapping the source view
template <typename View>
[[nodiscard]] auto operator|(View& view, const stride_adapter& adapter) {
    return strided_view<View>{view, adapter.stride_val, adapter.offset_val};
}

/// @brief Pipe operator: View | slice_adapter -> subview<View>
/// @tparam View Any view type
/// @param view The source view
/// @param adapter The slice adapter
/// @return subview wrapping the source view
template <typename View>
[[nodiscard]] auto operator|(View& view, const slice_adapter& adapter) {
    return subview<View>{view, adapter.offset_val, adapter.count_val};
}

/// @brief Pipe operator: View | take_adapter -> subview<View>
/// @tparam View Any view type with size()
/// @param view The source view
/// @param adapter The take adapter
/// @return subview of the first N elements
template <typename View>
[[nodiscard]] auto operator|(View& view, const take_adapter& adapter) {
    return subview<View>{view, 0, adapter.count_val};
}

/// @brief Pipe operator: View | drop_adapter -> subview<View>
/// @tparam View Any view type with size()
/// @param view The source view
/// @param adapter The drop adapter
/// @return subview skipping the first N elements
template <typename View>
[[nodiscard]] auto operator|(View& view, const drop_adapter& adapter) {
    size_type new_size = (view.size() > adapter.count_val)
                             ? (view.size() - adapter.count_val)
                             : 0;
    return subview<View>{view, adapter.count_val, new_size};
}

// =============================================================================
// compose() Function (Explicit Composition)
// =============================================================================

/// @brief Compose a strided view from a view
/// @tparam View Any view type
/// @param view The source view
/// @param s Stride size
/// @param offset Starting offset
/// @return strided_view<View>
template <typename View>
[[nodiscard]] auto compose_strided(View& view, std::ptrdiff_t s, size_type offset = 0) {
    return strided_view<View>{view, s, offset};
}

/// @brief Compose a subview from a view
/// @tparam View Any view type
/// @param view The source view
/// @param offset Starting offset
/// @param count Number of elements
/// @return subview<View>
template <typename View>
[[nodiscard]] auto compose_subview(View& view, size_type offset, size_type count) {
    return subview<View>{view, offset, count};
}

/// @brief Compose a strided view of a subview
/// @tparam View Any view type
/// @param view The source view
/// @param sub_offset Starting offset for the subview
/// @param sub_count Number of elements in the subview
/// @param s Stride within the subview
/// @return strided_view<subview<View>>
template <typename View>
[[nodiscard]] auto compose_strided_subview(
    View& view, size_type sub_offset, size_type sub_count, std::ptrdiff_t s) {
    auto sub = subview<View>{view, sub_offset, sub_count};
    return strided_view<subview<View>>{sub, s};
}

}  // namespace dtl
