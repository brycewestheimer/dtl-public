// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file local_iterator.hpp
/// @brief STL-compatible iterator for local partition
/// @details Never communicates - safe for standard algorithms.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <iterator>

namespace dtl {

/// @brief Random access iterator for local partition data
/// @tparam Container The distributed container type
/// @details Provides STL-compatible random access iteration over the local
///          partition. This iterator never communicates and is safe to use
///          with all standard library algorithms.
///
/// @par Iterator Category:
/// Random access iterator (supports +, -, [], <, >, <=, >=)
///
/// @par Thread Safety:
/// - Multiple readers are safe
/// - Writer must synchronize with readers
/// - Same guarantees as iterating over std::vector
///
/// @par Usage:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto local = vec.local_view();
/// for (auto it = local.begin(); it != local.end(); ++it) {
///     *it = process(*it);
/// }
/// @endcode
template <typename Container>
class local_iterator {
public:
    // ========================================================================
    // Iterator Type Aliases (STL Requirements)
    // ========================================================================

    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename Container::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename Container::pointer;
    using reference = typename Container::reference;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (singular iterator)
    local_iterator() noexcept = default;

    /// @brief Construct from pointer
    /// @param ptr Pointer to element
    explicit local_iterator(pointer ptr) noexcept : ptr_{ptr} {}

    // ========================================================================
    // Dereference Operations
    // ========================================================================

    /// @brief Dereference
    [[nodiscard]] reference operator*() const noexcept {
        return *ptr_;
    }

    /// @brief Arrow operator
    [[nodiscard]] pointer operator->() const noexcept {
        return ptr_;
    }

    /// @brief Index operator
    [[nodiscard]] reference operator[](difference_type n) const noexcept {
        return ptr_[n];
    }

    // ========================================================================
    // Increment/Decrement
    // ========================================================================

    /// @brief Pre-increment
    local_iterator& operator++() noexcept {
        ++ptr_;
        return *this;
    }

    /// @brief Post-increment
    local_iterator operator++(int) noexcept {
        local_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Pre-decrement
    local_iterator& operator--() noexcept {
        --ptr_;
        return *this;
    }

    /// @brief Post-decrement
    local_iterator operator--(int) noexcept {
        local_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    // ========================================================================
    // Arithmetic Operations
    // ========================================================================

    /// @brief Addition
    [[nodiscard]] local_iterator operator+(difference_type n) const noexcept {
        return local_iterator{ptr_ + n};
    }

    /// @brief Subtraction
    [[nodiscard]] local_iterator operator-(difference_type n) const noexcept {
        return local_iterator{ptr_ - n};
    }

    /// @brief Distance between iterators
    [[nodiscard]] difference_type operator-(const local_iterator& other) const noexcept {
        return ptr_ - other.ptr_;
    }

    /// @brief Compound addition
    local_iterator& operator+=(difference_type n) noexcept {
        ptr_ += n;
        return *this;
    }

    /// @brief Compound subtraction
    local_iterator& operator-=(difference_type n) noexcept {
        ptr_ -= n;
        return *this;
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    [[nodiscard]] bool operator==(const local_iterator& other) const noexcept {
        return ptr_ == other.ptr_;
    }

    [[nodiscard]] bool operator!=(const local_iterator& other) const noexcept {
        return ptr_ != other.ptr_;
    }

    [[nodiscard]] bool operator<(const local_iterator& other) const noexcept {
        return ptr_ < other.ptr_;
    }

    [[nodiscard]] bool operator>(const local_iterator& other) const noexcept {
        return ptr_ > other.ptr_;
    }

    [[nodiscard]] bool operator<=(const local_iterator& other) const noexcept {
        return ptr_ <= other.ptr_;
    }

    [[nodiscard]] bool operator>=(const local_iterator& other) const noexcept {
        return ptr_ >= other.ptr_;
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// @brief Get underlying pointer
    [[nodiscard]] pointer base() const noexcept {
        return ptr_;
    }

private:
    pointer ptr_ = nullptr;
};

/// @brief Addition with difference on left
template <typename Container>
[[nodiscard]] local_iterator<Container> operator+(
    typename local_iterator<Container>::difference_type n,
    const local_iterator<Container>& it) noexcept {
    return it + n;
}

/// @brief Const version of local iterator
template <typename Container>
class const_local_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename Container::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename Container::const_pointer;
    using reference = typename Container::const_reference;

    const_local_iterator() noexcept = default;
    explicit const_local_iterator(pointer ptr) noexcept : ptr_{ptr} {}

    // Allow conversion from non-const iterator
    const_local_iterator(const local_iterator<Container>& other) noexcept
        : ptr_{other.base()} {}

    [[nodiscard]] reference operator*() const noexcept { return *ptr_; }
    [[nodiscard]] pointer operator->() const noexcept { return ptr_; }
    [[nodiscard]] reference operator[](difference_type n) const noexcept { return ptr_[n]; }

    const_local_iterator& operator++() noexcept { ++ptr_; return *this; }
    const_local_iterator operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    const_local_iterator& operator--() noexcept { --ptr_; return *this; }
    const_local_iterator operator--(int) noexcept { auto t = *this; --(*this); return t; }

    [[nodiscard]] const_local_iterator operator+(difference_type n) const noexcept {
        return const_local_iterator{ptr_ + n};
    }
    [[nodiscard]] const_local_iterator operator-(difference_type n) const noexcept {
        return const_local_iterator{ptr_ - n};
    }
    [[nodiscard]] difference_type operator-(const const_local_iterator& o) const noexcept {
        return ptr_ - o.ptr_;
    }

    const_local_iterator& operator+=(difference_type n) noexcept { ptr_ += n; return *this; }
    const_local_iterator& operator-=(difference_type n) noexcept { ptr_ -= n; return *this; }

    [[nodiscard]] bool operator==(const const_local_iterator& o) const noexcept { return ptr_ == o.ptr_; }
    [[nodiscard]] bool operator!=(const const_local_iterator& o) const noexcept { return ptr_ != o.ptr_; }
    [[nodiscard]] bool operator<(const const_local_iterator& o) const noexcept { return ptr_ < o.ptr_; }
    [[nodiscard]] bool operator>(const const_local_iterator& o) const noexcept { return ptr_ > o.ptr_; }
    [[nodiscard]] bool operator<=(const const_local_iterator& o) const noexcept { return ptr_ <= o.ptr_; }
    [[nodiscard]] bool operator>=(const const_local_iterator& o) const noexcept { return ptr_ >= o.ptr_; }

    [[nodiscard]] pointer base() const noexcept { return ptr_; }

private:
    pointer ptr_ = nullptr;
};

}  // namespace dtl
