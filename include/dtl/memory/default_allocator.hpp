// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file default_allocator.hpp
/// @brief Default allocator using host memory space
/// @details Provides the default allocator for DTL containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/memory/allocator.hpp>
#include <dtl/memory/host_memory_space.hpp>

// Forward declarations for placement policies
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/device_preferred.hpp>
#include <dtl/policies/placement/unified_memory.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/memory/cuda_memory_space.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>
#endif

namespace dtl {

// ============================================================================
// Default Allocator
// ============================================================================

/// @brief Default allocator type for DTL containers
/// @tparam T Element type
template <typename T>
using default_allocator = memory_space_allocator<T, host_memory_space>;

// ============================================================================
// Allocator Selection
// ============================================================================

/// @brief Select allocator based on placement policy
/// @tparam T Element type
/// @tparam Placement Placement policy type
template <typename T, typename Placement>
struct select_allocator {
    /// @brief Selected allocator type (default: host)
    using type = default_allocator<T>;
};

/// @brief Select allocator for host_only placement
template <typename T>
struct select_allocator<T, host_only> {
    using type = memory_space_allocator<T, host_memory_space>;
};

/// @brief Select allocator for device_only placement
/// @details When CUDA is enabled, uses cuda_device_memory_space_for<DeviceId>
///          which ensures allocations occur on the specified device.
///          When CUDA is disabled, static_assert fires when instantiated.
/// @note Different DeviceId values produce different allocator types,
///       fixing the issue where all device_only<N> used the same allocator.
template <typename T, int DeviceId>
struct select_allocator<T, device_only<DeviceId>> {
#if DTL_ENABLE_CUDA
    using type = memory_space_allocator<T, cuda::cuda_device_memory_space_for<DeviceId>>;
#else
    // Make static_assert depend on T so it's only evaluated at instantiation time
    static_assert(sizeof(T) == 0, "device_only<DeviceId> requires CUDA support. "
                  "Rebuild with -DDTL_ENABLE_CUDA=ON");
    using type = memory_space_allocator<T, host_memory_space>;  // Never used
#endif
};

// Forward declaration for device_only_runtime
struct device_only_runtime;

/// @brief Select allocator for device_only_runtime placement
/// @details Uses cuda_device_memory_space_runtime which stores device ID at runtime.
///          The actual device ID is determined at container construction from the context.
///          When CUDA is disabled, static_assert fires when instantiated.
/// @since 0.1.0
template <typename T>
struct select_allocator<T, device_only_runtime> {
#if DTL_ENABLE_CUDA
    using type = memory_space_allocator<T, cuda::cuda_device_memory_space_runtime>;
#else
    // Make static_assert depend on T so it's only evaluated at instantiation time
    static_assert(sizeof(T) == 0, "device_only_runtime requires CUDA support. "
                  "Rebuild with -DDTL_ENABLE_CUDA=ON");
    using type = memory_space_allocator<T, host_memory_space>;  // Never used
#endif
};

/// @brief Select allocator for unified_memory placement
/// @details When CUDA is enabled, uses cuda_unified_memory_space.
///          When CUDA is disabled, static_assert fires when instantiated.
template <typename T>
struct select_allocator<T, unified_memory> {
#if DTL_ENABLE_CUDA
    using type = memory_space_allocator<T, cuda::cuda_unified_memory_space>;
#else
    // Make static_assert depend on T so it's only evaluated at instantiation time
    static_assert(sizeof(T) == 0, "unified_memory requires CUDA support. "
                  "Rebuild with -DDTL_ENABLE_CUDA=ON");
    using type = memory_space_allocator<T, host_memory_space>;  // Never used
#endif
};

/// @brief Select allocator for device_preferred placement
/// @details When CUDA is enabled, uses cuda_device_memory_space.
///          When CUDA is disabled, falls back to host_memory_space.
template <typename T>
struct select_allocator<T, device_preferred> {
#if DTL_ENABLE_CUDA
    using type = memory_space_allocator<T, cuda::cuda_device_memory_space>;
#else
    // Fallback to host when CUDA is not available
    using type = memory_space_allocator<T, host_memory_space>;
#endif
};

/// @brief Helper alias for select_allocator
template <typename T, typename Placement>
using select_allocator_t = typename select_allocator<T, Placement>::type;

// ============================================================================
// Allocator Utilities
// ============================================================================

/// @brief Create default allocator for a type
/// @tparam T Element type
/// @return Default allocator instance
template <typename T>
[[nodiscard]] constexpr default_allocator<T> make_default_allocator() noexcept {
    return default_allocator<T>{};
}

/// @brief Check if an allocator uses host memory
/// @tparam Alloc Allocator type
/// @return true if host allocator
template <typename Alloc>
[[nodiscard]] constexpr bool is_host_allocator() noexcept {
    return allocator_traits_ext<Alloc>::is_host_allocator;
}

/// @brief Check if an allocator uses device memory
/// @tparam Alloc Allocator type
/// @return true if device allocator
template <typename Alloc>
[[nodiscard]] constexpr bool is_device_allocator() noexcept {
    return allocator_traits_ext<Alloc>::is_device_allocator;
}

// ============================================================================
// Scoped Allocation
// ============================================================================

/// @brief RAII wrapper for allocator-managed memory
/// @tparam T Element type
/// @tparam Alloc Allocator type
template <typename T, typename Alloc = default_allocator<T>>
class scoped_allocation {
public:
    using value_type = T;
    using allocator_type = Alloc;
    using size_type = typename std::allocator_traits<Alloc>::size_type;
    using pointer = typename std::allocator_traits<Alloc>::pointer;

    /// @brief Construct with size
    /// @param n Number of elements
    /// @param alloc Allocator instance
    explicit scoped_allocation(size_type n, const Alloc& alloc = Alloc{})
        : alloc_(alloc)
        , ptr_(std::allocator_traits<Alloc>::allocate(alloc_, n))
        , size_(n) {}

    /// @brief Destructor deallocates memory
    ~scoped_allocation() {
        if (ptr_) {
            std::allocator_traits<Alloc>::deallocate(alloc_, ptr_, size_);
        }
    }

    /// @brief Non-copyable
    scoped_allocation(const scoped_allocation&) = delete;
    scoped_allocation& operator=(const scoped_allocation&) = delete;

    /// @brief Movable
    scoped_allocation(scoped_allocation&& other) noexcept
        : alloc_(std::move(other.alloc_))
        , ptr_(other.ptr_)
        , size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    scoped_allocation& operator=(scoped_allocation&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                std::allocator_traits<Alloc>::deallocate(alloc_, ptr_, size_);
            }
            alloc_ = std::move(other.alloc_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /// @brief Get pointer to memory
    [[nodiscard]] pointer get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }

    /// @brief Get size
    [[nodiscard]] size_type size() const noexcept { return size_; }

    /// @brief Release ownership
    [[nodiscard]] pointer release() noexcept {
        pointer p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return p;
    }

    /// @brief Element access
    [[nodiscard]] T& operator[](size_type i) noexcept { return ptr_[i]; }
    [[nodiscard]] const T& operator[](size_type i) const noexcept { return ptr_[i]; }

    /// @brief Range access
    [[nodiscard]] pointer begin() noexcept { return ptr_; }
    [[nodiscard]] pointer end() noexcept { return ptr_ + size_; }
    [[nodiscard]] const T* begin() const noexcept { return ptr_; }
    [[nodiscard]] const T* end() const noexcept { return ptr_ + size_; }

private:
    Alloc alloc_;
    pointer ptr_;
    size_type size_;
};

/// @brief Create a scoped allocation
/// @tparam T Element type
/// @tparam Alloc Allocator type
/// @param n Number of elements
/// @param alloc Allocator
/// @return Scoped allocation object
template <typename T, typename Alloc = default_allocator<T>>
[[nodiscard]] scoped_allocation<T, Alloc> make_scoped_allocation(
    typename std::allocator_traits<Alloc>::size_type n,
    const Alloc& alloc = Alloc{}) {
    return scoped_allocation<T, Alloc>(n, alloc);
}

}  // namespace dtl
