// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_iterator.hpp
/// @brief Iterator for GPU-resident data
/// @details Device-side iterator for use in kernels.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <iterator>

namespace dtl {

/// @brief Iterator for GPU device memory
/// @tparam T Element type
/// @details Provides an iterator interface for data residing in GPU device
///          memory. This iterator is designed to be usable both on host
///          (for setup) and device (inside kernels).
///
/// @par Host vs Device Usage:
/// - Host: Used for range setup, kernel launch parameters
/// - Device: Used inside kernels for actual iteration
///
/// @par Memory Model:
/// - Points to device memory (cudaMalloc'd or similar)
/// - Dereference only valid in device code
/// - Host code should not dereference
///
/// @par Usage (CUDA example):
/// @code
/// // Host: setup
/// auto begin = vec.device_begin();
/// auto end = vec.device_end();
///
/// // Kernel launch
/// my_kernel<<<grid, block>>>(begin, end);
///
/// // Device: inside kernel
/// __global__ void my_kernel(device_iterator<int> begin, device_iterator<int> end) {
///     for (auto it = begin + threadIdx.x; it < end; it += blockDim.x) {
///         *it = compute(*it);
///     }
/// }
/// @endcode
template <typename T>
class device_iterator {
public:
    // ========================================================================
    // Iterator Type Aliases
    // ========================================================================

    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor
    DTL_HOST_DEVICE
    device_iterator() noexcept = default;

    /// @brief Construct from device pointer
    /// @param ptr Pointer to device memory
    DTL_HOST_DEVICE
    explicit device_iterator(pointer ptr) noexcept : ptr_{ptr} {}

    // ========================================================================
    // Dereference (Device Only)
    // ========================================================================

    /// @brief Dereference
    /// @warning Only valid in device code
    DTL_HOST_DEVICE
    [[nodiscard]] reference operator*() const noexcept {
        return *ptr_;
    }

    /// @brief Arrow operator
    DTL_HOST_DEVICE
    [[nodiscard]] pointer operator->() const noexcept {
        return ptr_;
    }

    /// @brief Index operator
    DTL_HOST_DEVICE
    [[nodiscard]] reference operator[](difference_type n) const noexcept {
        return ptr_[n];
    }

    // ========================================================================
    // Increment/Decrement
    // ========================================================================

    DTL_HOST_DEVICE
    device_iterator& operator++() noexcept {
        ++ptr_;
        return *this;
    }

    DTL_HOST_DEVICE
    device_iterator operator++(int) noexcept {
        device_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    DTL_HOST_DEVICE
    device_iterator& operator--() noexcept {
        --ptr_;
        return *this;
    }

    DTL_HOST_DEVICE
    device_iterator operator--(int) noexcept {
        device_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    // ========================================================================
    // Arithmetic
    // ========================================================================

    DTL_HOST_DEVICE
    [[nodiscard]] device_iterator operator+(difference_type n) const noexcept {
        return device_iterator{ptr_ + n};
    }

    DTL_HOST_DEVICE
    [[nodiscard]] device_iterator operator-(difference_type n) const noexcept {
        return device_iterator{ptr_ - n};
    }

    DTL_HOST_DEVICE
    [[nodiscard]] difference_type operator-(const device_iterator& other) const noexcept {
        return ptr_ - other.ptr_;
    }

    DTL_HOST_DEVICE
    device_iterator& operator+=(difference_type n) noexcept {
        ptr_ += n;
        return *this;
    }

    DTL_HOST_DEVICE
    device_iterator& operator-=(difference_type n) noexcept {
        ptr_ -= n;
        return *this;
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator==(const device_iterator& other) const noexcept {
        return ptr_ == other.ptr_;
    }

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator!=(const device_iterator& other) const noexcept {
        return ptr_ != other.ptr_;
    }

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator<(const device_iterator& other) const noexcept {
        return ptr_ < other.ptr_;
    }

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator>(const device_iterator& other) const noexcept {
        return ptr_ > other.ptr_;
    }

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator<=(const device_iterator& other) const noexcept {
        return ptr_ <= other.ptr_;
    }

    DTL_HOST_DEVICE
    [[nodiscard]] bool operator>=(const device_iterator& other) const noexcept {
        return ptr_ >= other.ptr_;
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// @brief Get raw device pointer
    DTL_HOST_DEVICE
    [[nodiscard]] pointer get() const noexcept {
        return ptr_;
    }

    /// @brief Get raw device pointer (alias for compatibility)
    DTL_HOST_DEVICE
    [[nodiscard]] pointer data() const noexcept {
        return ptr_;
    }

private:
    pointer ptr_ = nullptr;
};

/// @brief Addition with difference on left
template <typename T>
DTL_HOST_DEVICE
[[nodiscard]] device_iterator<T> operator+(
    typename device_iterator<T>::difference_type n,
    const device_iterator<T>& it) noexcept {
    return it + n;
}

/// @brief Const device iterator
template <typename T>
using const_device_iterator = device_iterator<const T>;

/// @brief Make device iterator from pointer
template <typename T>
DTL_HOST_DEVICE
[[nodiscard]] device_iterator<T> make_device_iterator(T* ptr) noexcept {
    return device_iterator<T>{ptr};
}

}  // namespace dtl
