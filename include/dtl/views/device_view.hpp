// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_view.hpp
/// @brief Device-accessible view for GPU containers
/// @details Provides a view type that wraps device memory without exposing
///          host-iterable raw pointers, preventing accidental host dereferencing.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/device_concepts.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {

/// @brief Device-accessible view of contiguous memory
/// @tparam T Element type (must satisfy DeviceStorable)
/// @details This view provides access to device memory in a way that:
///          - Exposes data() and size() for use with kernels/thrust
///          - Does NOT expose begin()/end() returning raw pointers
///          - Cannot be accidentally used with STL algorithms
///
/// @par Usage
/// @code
/// auto view = container.device_view();
/// thrust::fill(thrust::device, view.data(), view.data() + view.size(), 42);
/// @endcode
///
/// @par Why Not Just Use data()/size()?
/// By returning a distinct view type, we make the API intention clear:
/// - `local_view()` → host iteration, STL algorithms
/// - `device_view()` → GPU kernels, thrust algorithms
///
/// Compile-time constraints prevent calling the wrong view for a placement.
template <typename T>
    requires DeviceStorable<T>
class device_view {
public:
    using value_type = std::remove_cv_t<T>;
    using element_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = dtl::size_type;

    // Note: No iterator types - device memory is not host-iterable

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (empty view)
    constexpr device_view() noexcept = default;

    /// @brief Construct from device pointer and size
    /// @param data Device pointer to first element
    /// @param size Number of elements
    constexpr device_view(pointer data, size_type size) noexcept
        : data_(data), size_(size) {}

    /// @brief Construct from device pointer, size, and device ID
    /// @param data Device pointer to first element
    /// @param size Number of elements
    /// @param device_id The device this memory resides on
    constexpr device_view(pointer data, size_type size, int device_id) noexcept
        : data_(data), size_(size), device_id_(device_id) {}

    // ========================================================================
    // Accessors
    // ========================================================================

    /// @brief Get device pointer to data
    /// @return Pointer to first element (device memory)
    [[nodiscard]] constexpr pointer data() noexcept {
        return data_;
    }

    /// @brief Get const device pointer to data
    [[nodiscard]] constexpr const_pointer data() const noexcept {
        return data_;
    }

    /// @brief Get number of elements
    [[nodiscard]] constexpr size_type size() const noexcept {
        return size_;
    }

    /// @brief Check if view is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    /// @brief Get size in bytes
    [[nodiscard]] constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(T);
    }

    /// @brief Get device ID
    [[nodiscard]] constexpr int device_id() const noexcept {
        return device_id_;
    }

    // ========================================================================
    // Subviews
    // ========================================================================

    /// @brief Get a subview starting at offset
    /// @param offset Starting offset
    /// @param count Number of elements (default: rest of view)
    [[nodiscard]] constexpr device_view subview(size_type offset,
                                                  size_type count = npos) const noexcept {
        if (offset >= size_) {
            return device_view{};
        }
        size_type actual_count = (count == npos || offset + count > size_)
                                     ? (size_ - offset)
                                     : count;
        return device_view{data_ + offset, actual_count, device_id_};
    }

    /// @brief Get first n elements
    [[nodiscard]] constexpr device_view first(size_type n) const noexcept {
        return subview(0, n);
    }

    /// @brief Get last n elements
    [[nodiscard]] constexpr device_view last(size_type n) const noexcept {
        return n >= size_ ? *this : subview(size_ - n, n);
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    [[nodiscard]] friend constexpr bool operator==(const device_view& a,
                                                    const device_view& b) noexcept {
        return a.data_ == b.data_ && a.size_ == b.size_;
    }

    [[nodiscard]] friend constexpr bool operator!=(const device_view& a,
                                                    const device_view& b) noexcept {
        return !(a == b);
    }

    // ========================================================================
    // Constants
    // ========================================================================

    static constexpr size_type npos = static_cast<size_type>(-1);

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
    int device_id_ = -1;
};

// ============================================================================
// Device View Factory Functions
// ============================================================================

/// @brief Create a device view from a pointer and size
/// @tparam T Element type
/// @param data Device pointer
/// @param size Number of elements
/// @return device_view<T>
template <typename T>
    requires DeviceStorable<T>
[[nodiscard]] constexpr device_view<T> make_device_view(T* data, size_type size) noexcept {
    return device_view<T>{data, size};
}

/// @brief Create a device view from a pointer, size, and device ID
template <typename T>
    requires DeviceStorable<T>
[[nodiscard]] constexpr device_view<T> make_device_view(T* data, size_type size,
                                                         int device_id) noexcept {
    return device_view<T>{data, size, device_id};
}

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Check if a type is a device_view
template <typename T>
struct is_device_view : std::false_type {};

template <typename T>
struct is_device_view<device_view<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_device_view_v = is_device_view<T>::value;

}  // namespace dtl
