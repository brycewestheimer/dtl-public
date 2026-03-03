// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file strided_view.hpp
/// @brief Strided access pattern view
/// @details Access elements at regular intervals (every Nth element).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <iterator>

namespace dtl {

/// @brief Iterator that steps by a fixed stride
/// @tparam BaseIterator The underlying iterator type
template <typename BaseIterator>
class strided_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::iterator_traits<BaseIterator>::value_type;
    using difference_type = typename std::iterator_traits<BaseIterator>::difference_type;
    using pointer = typename std::iterator_traits<BaseIterator>::pointer;
    using reference = typename std::iterator_traits<BaseIterator>::reference;

    /// @brief Construct strided iterator
    /// @param base The underlying iterator
    /// @param stride The step size between elements
    strided_iterator(BaseIterator base, difference_type stride)
        : base_{base}
        , stride_{stride} {}

    /// @brief Dereference
    [[nodiscard]] reference operator*() const {
        return *base_;
    }

    /// @brief Arrow operator
    [[nodiscard]] pointer operator->() const {
        return &(*base_);
    }

    /// @brief Pre-increment (advance by stride)
    strided_iterator& operator++() {
        base_ += stride_;
        return *this;
    }

    /// @brief Post-increment
    strided_iterator operator++(int) {
        strided_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Pre-decrement
    strided_iterator& operator--() {
        base_ -= stride_;
        return *this;
    }

    /// @brief Post-decrement
    strided_iterator operator--(int) {
        strided_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    /// @brief Advance by n positions
    strided_iterator& operator+=(difference_type n) {
        base_ += n * stride_;
        return *this;
    }

    /// @brief Retreat by n positions
    strided_iterator& operator-=(difference_type n) {
        base_ -= n * stride_;
        return *this;
    }

    /// @brief Addition
    [[nodiscard]] strided_iterator operator+(difference_type n) const {
        return strided_iterator{base_ + n * stride_, stride_};
    }

    /// @brief Subtraction
    [[nodiscard]] strided_iterator operator-(difference_type n) const {
        return strided_iterator{base_ - n * stride_, stride_};
    }

    /// @brief Distance between iterators
    [[nodiscard]] difference_type operator-(const strided_iterator& other) const {
        return (base_ - other.base_) / stride_;
    }

    /// @brief Index access
    [[nodiscard]] reference operator[](difference_type n) const {
        return base_[n * stride_];
    }

    /// @brief Equality comparison
    [[nodiscard]] bool operator==(const strided_iterator& other) const noexcept {
        return base_ == other.base_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] bool operator!=(const strided_iterator& other) const noexcept {
        return !(*this == other);
    }

    /// @brief Less than
    [[nodiscard]] bool operator<(const strided_iterator& other) const noexcept {
        return base_ < other.base_;
    }

    /// @brief Greater than
    [[nodiscard]] bool operator>(const strided_iterator& other) const noexcept {
        return base_ > other.base_;
    }

    /// @brief Less than or equal
    [[nodiscard]] bool operator<=(const strided_iterator& other) const noexcept {
        return !(other < *this);
    }

    /// @brief Greater than or equal
    [[nodiscard]] bool operator>=(const strided_iterator& other) const noexcept {
        return !(*this < other);
    }

    /// @brief Get underlying iterator
    [[nodiscard]] BaseIterator base() const noexcept {
        return base_;
    }

    /// @brief Get stride
    [[nodiscard]] difference_type stride() const noexcept {
        return stride_;
    }

private:
    BaseIterator base_;
    difference_type stride_;
};

/// @brief View that accesses elements at regular intervals
/// @tparam View The underlying view type
/// @details Provides access to every Nth element of the underlying view.
///          Useful for:
///          - Red-black Gauss-Seidel iterations
///          - Interleaved data access
///          - Decimation/downsampling
///
/// @par Usage:
/// @code
/// auto local = vec.local_view();
/// // Access every other element (stride 2)
/// auto even_indices = make_strided_view(local, 2);
/// auto odd_indices = make_strided_view(local, 2, 1);  // offset by 1
/// @endcode
template <typename View>
class strided_view {
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
    using iterator = strided_iterator<typename View::iterator>;

    /// @brief Const iterator type
    using const_iterator = strided_iterator<typename View::const_iterator>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    /// @brief Construct a strided view
    /// @param view The underlying view
    /// @param stride Step size between elements
    /// @param offset Starting offset (default 0)
    strided_view(View& view, difference_type stride, size_type offset = 0)
        : view_{&view}
        , stride_{static_cast<size_type>(stride)}
        , offset_{offset} {}

    /// @brief Get iterator to beginning
    [[nodiscard]] iterator begin() noexcept {
        return iterator{view_->begin() + static_cast<difference_type>(offset_), static_cast<difference_type>(stride_)};
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator begin() const noexcept {
        return const_iterator{view_->begin() + static_cast<difference_type>(offset_), static_cast<difference_type>(stride_)};
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return begin();
    }

    /// @brief Get iterator to end
    [[nodiscard]] iterator end() noexcept {
        return iterator{view_->begin() + static_cast<difference_type>(offset_ + size() * stride_), static_cast<difference_type>(stride_)};
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator end() const noexcept {
        return const_iterator{view_->begin() + static_cast<difference_type>(offset_ + size() * stride_), static_cast<difference_type>(stride_)};
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator cend() const noexcept {
        return end();
    }

    /// @brief Get number of elements in strided view
    [[nodiscard]] size_type size() const noexcept {
        if (view_->size() <= offset_) return 0;
        return (view_->size() - offset_ + stride_ - 1) / stride_;
    }

    /// @brief Check if empty
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    /// @brief Access element by strided index
    [[nodiscard]] reference operator[](size_type idx) {
        return (*view_)[offset_ + idx * stride_];
    }

    /// @brief Access element by strided index (const)
    [[nodiscard]] const_reference operator[](size_type idx) const {
        return (*view_)[offset_ + idx * stride_];
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
        return (*this)[size() - 1];
    }

    /// @brief Get last element (const)
    [[nodiscard]] const_reference back() const {
        return (*this)[size() - 1];
    }

    /// @brief Get the stride
    [[nodiscard]] size_type stride() const noexcept {
        return stride_;
    }

    /// @brief Get the offset
    [[nodiscard]] size_type offset() const noexcept {
        return offset_;
    }

private:
    View* view_;
    size_type stride_;
    size_type offset_;
};

/// @brief Create a strided view
/// @tparam View The view type
/// @param view The view to stride over
/// @param stride Step size between elements
/// @param offset Starting offset (default 0)
/// @return strided_view<View>
template <typename View>
[[nodiscard]] auto make_strided_view(View& view,
                                      std::ptrdiff_t stride,
                                      size_type offset = 0) {
    return strided_view<View>{view, stride, offset};
}

}  // namespace dtl
