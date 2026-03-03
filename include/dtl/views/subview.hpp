// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file subview.hpp
/// @brief Sub-range view for accessing portions of a view
/// @details Useful for halo regions, boundary conditions, etc.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

namespace dtl {

/// @brief A view representing a contiguous sub-range of another view
/// @tparam View The underlying view type
/// @details Provides access to a subset of elements without copying.
///          Commonly used for:
///          - Halo/ghost regions in stencil computations
///          - Boundary elements for communication
///          - Splitting work among threads
///
/// @par Usage:
/// @code
/// auto local = vec.local_view();
/// // Get first 10 elements (e.g., halo region)
/// auto halo = make_subview(local, 0, 10);
/// // Get interior (excluding halo on both ends)
/// auto interior = make_subview(local, 10, local.size() - 10);
/// @endcode
template <typename View>
class subview {
public:
    /// @brief Value type
    using value_type = typename View::value_type;

    /// @brief Reference type
    using reference = typename View::reference;

    /// @brief Const reference type
    using const_reference = typename View::const_reference;

    /// @brief Pointer type
    using pointer = typename View::pointer;

    /// @brief Const pointer type
    using const_pointer = typename View::const_pointer;

    /// @brief Iterator type
    using iterator = typename View::iterator;

    /// @brief Const iterator type
    using const_iterator = typename View::const_iterator;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    /// @brief Construct a subview
    /// @param view The underlying view
    /// @param offset Starting offset within the view
    /// @param count Number of elements in the subview
    subview(View& view, size_type offset, size_type count)
        : view_{&view}
        , offset_{offset}
        , count_{count} {}

    /// @brief Get iterator to beginning
    [[nodiscard]] iterator begin() noexcept {
        return view_->begin() + offset_;
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator begin() const noexcept {
        return view_->begin() + offset_;
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return begin();
    }

    /// @brief Get iterator to end
    [[nodiscard]] iterator end() noexcept {
        return view_->begin() + offset_ + count_;
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator end() const noexcept {
        return view_->begin() + offset_ + count_;
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator cend() const noexcept {
        return end();
    }

    /// @brief Get number of elements
    [[nodiscard]] size_type size() const noexcept {
        return count_;
    }

    /// @brief Check if empty
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }

    /// @brief Access element by index (relative to subview start)
    [[nodiscard]] reference operator[](size_type idx) {
        return (*view_)[offset_ + idx];
    }

    /// @brief Access element by index (const)
    [[nodiscard]] const_reference operator[](size_type idx) const {
        return (*view_)[offset_ + idx];
    }

    /// @brief Get pointer to data
    [[nodiscard]] pointer data() noexcept {
        return view_->data() + offset_;
    }

    /// @brief Get const pointer to data
    [[nodiscard]] const_pointer data() const noexcept {
        return view_->data() + offset_;
    }

    /// @brief Get first element
    [[nodiscard]] reference front() {
        return (*this)[0];
    }

    /// @brief Get first element (const)
    [[nodiscard]] const_reference front() const {
        return (*this)[0];
    }

    /// @brief Get last element
    [[nodiscard]] reference back() {
        return (*this)[count_ - 1];
    }

    /// @brief Get last element (const)
    [[nodiscard]] const_reference back() const {
        return (*this)[count_ - 1];
    }

    /// @brief Get offset within parent view
    [[nodiscard]] size_type offset() const noexcept {
        return offset_;
    }

    /// @brief Create a further subview of this subview
    /// @param offset Starting offset within this subview
    /// @param count Number of elements
    /// @return New subview
    [[nodiscard]] subview subrange(size_type offset, size_type count) const {
        return subview{*view_, offset_ + offset, count};
    }

private:
    View* view_;
    size_type offset_;
    size_type count_;
};

/// @brief Create a subview from a view
/// @tparam View The view type
/// @param view The view to create a subview of
/// @param offset Starting offset
/// @param count Number of elements
/// @return subview<View>
template <typename View>
[[nodiscard]] auto make_subview(View& view, size_type offset, size_type count) {
    return subview<View>{view, offset, count};
}

/// @brief Create a subview of the first n elements
/// @tparam View The view type
/// @param view The view
/// @param count Number of elements from the beginning
/// @return subview<View>
template <typename View>
[[nodiscard]] auto take(View& view, size_type count) {
    return subview<View>{view, 0, count};
}

/// @brief Create a subview excluding the first n elements
/// @tparam View The view type
/// @param view The view
/// @param count Number of elements to skip
/// @return subview<View>
template <typename View>
[[nodiscard]] auto drop(View& view, size_type count) {
    return subview<View>{view, count, view.size() - count};
}

}  // namespace dtl
