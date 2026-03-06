// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file storage.hpp
/// @brief Placement-aware storage abstraction for distributed containers
/// @details Provides storage implementations that respect placement policies,
///          ensuring device-only containers don't perform host initialization.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/policies/placement/unified_memory.hpp>
#include <dtl/memory/default_allocator.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/device_buffer.hpp>
#include <dtl/memory/cuda_memory_space.hpp>
#endif

#include <vector>
#include <algorithm>
#include <concepts>
#include <cstring>
#include <initializer_list>
#include <iterator>

namespace dtl {
namespace detail {

// ============================================================================
// Storage Category Tags
// ============================================================================

/// @brief Tag for host-accessible storage (std::vector)
struct host_storage_tag {};

/// @brief Tag for device-only storage (device_buffer)
struct device_storage_tag {};

/// @brief Tag for unified/managed memory storage
struct unified_storage_tag {};

// ============================================================================
// Storage Category Selection
// ============================================================================

/// @brief Select storage category based on placement policy
template <typename Placement>
struct storage_category {
    using type = host_storage_tag;  // Default to host
};

template <int DeviceId>
struct storage_category<device_only<DeviceId>> {
    using type = device_storage_tag;
};

template <>
struct storage_category<device_only_runtime> {
    using type = device_storage_tag;
};

template <>
struct storage_category<unified_memory> {
    using type = unified_storage_tag;
};

template <typename Placement>
using storage_category_t = typename storage_category<Placement>::type;

// ============================================================================
// Host Storage Implementation
// ============================================================================

/// @brief Host-based storage using std::vector
/// @tparam T Element type
/// @tparam Allocator Allocator type (default: std::allocator<T>)
template <typename T, typename Allocator = std::allocator<T>>
class host_storage {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using storage_tag = host_storage_tag;

    // ========================================================================
    // Constructors
    // ========================================================================

    host_storage() = default;

    explicit host_storage(size_type n)
        : data_(n) {}

    host_storage(size_type n, const T& value)
        : data_(n, value) {}

    template <typename InputIt>
        requires (!std::integral<InputIt>)
    host_storage(InputIt first, InputIt last)
        : data_(first, last) {}

    host_storage(std::initializer_list<T> init)
        : data_(init) {}

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] pointer data() noexcept { return data_.data(); }
    [[nodiscard]] const_pointer data() const noexcept { return data_.data(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] size_type capacity() const noexcept { return data_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    reference operator[](size_type i) noexcept { return data_[i]; }
    const_reference operator[](size_type i) const noexcept { return data_[i]; }

    [[nodiscard]] allocator_type get_allocator() const noexcept {
        return data_.get_allocator();
    }

    // ========================================================================
    // Modifiers
    // ========================================================================

    void resize(size_type n) { data_.resize(n); }
    void resize(size_type n, const T& value) { data_.resize(n, value); }
    void reserve(size_type n) { data_.reserve(n); }
    void clear() noexcept { data_.clear(); }
    void shrink_to_fit() { data_.shrink_to_fit(); }
    void swap(host_storage& other) noexcept { data_.swap(other.data_); }

    // ========================================================================
    // Iterators
    // ========================================================================

    auto begin() noexcept { return data_.begin(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto end() const noexcept { return data_.end(); }

    // ========================================================================
    // Placement Queries
    // ========================================================================

    [[nodiscard]] static constexpr bool is_host_accessible() noexcept { return true; }
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept { return false; }
    [[nodiscard]] constexpr int device_id() const noexcept { return -1; }

private:
    std::vector<T, Allocator> data_;
};

// ============================================================================
// Device Storage Implementation (CUDA)
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Device-only storage using device_buffer
/// @tparam T Element type (must satisfy DeviceStorable)
template <typename T>
    requires DeviceStorable<T>
class device_storage {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using storage_tag = device_storage_tag;

    // ========================================================================
    // Constructors
    // ========================================================================

    device_storage() = default;

    explicit device_storage(size_type n, int device_id = 0)
        : buffer_(n, device_id) {}

    // Note: No value constructor - device storage doesn't value-initialize
    // Use fill algorithms to set values after construction

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] pointer data() noexcept { return buffer_.data(); }
    [[nodiscard]] const_pointer data() const noexcept { return buffer_.data(); }
    [[nodiscard]] size_type size() const noexcept { return buffer_.size(); }
    [[nodiscard]] size_type capacity() const noexcept { return buffer_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return buffer_.empty(); }
    [[nodiscard]] int device_id() const noexcept { return buffer_.device_id(); }

    // Note: No operator[] - would require device access from host

    // ========================================================================
    // Modifiers
    // ========================================================================

    void resize(size_type n) { buffer_.resize(n); }
    void reserve(size_type n) { buffer_.reserve(n); }
    void clear() noexcept { buffer_.clear(); }
    void swap(device_storage& other) noexcept { buffer_.swap(other.buffer_); }

    /// @brief Zero-fill the storage
    void zero_fill() { buffer_.memset(0); }

    // ========================================================================
    // Placement Queries
    // ========================================================================

    [[nodiscard]] static constexpr bool is_host_accessible() noexcept { return false; }
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept { return true; }

private:
    cuda::device_buffer<T> buffer_;
};

#else  // !DTL_ENABLE_CUDA

/// @brief Stub device storage when CUDA is disabled
template <typename T>
class device_storage {
    static_assert(sizeof(T) == 0,
        "device_storage requires CUDA support. Rebuild with -DDTL_ENABLE_CUDA=ON");
};

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Unified Storage Implementation
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Unified/managed memory storage
/// @tparam T Element type (must satisfy DeviceStorable)
template <typename T>
    requires DeviceStorable<T>
class unified_storage {
public:
    using value_type = T;
    using allocator_type = memory_space_allocator<T, cuda::cuda_unified_memory_space>;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using storage_tag = unified_storage_tag;

    // ========================================================================
    // Constructors
    // ========================================================================

    unified_storage() = default;

    explicit unified_storage(size_type n)
        : data_(n) {}

    unified_storage(size_type n, const T& value)
        : data_(n, value) {}

    template <typename InputIt>
        requires (!std::integral<InputIt>)
    unified_storage(InputIt first, InputIt last)
        : data_(first, last) {}

    unified_storage(std::initializer_list<T> init)
        : data_(init) {}

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] pointer data() noexcept { return data_.data(); }
    [[nodiscard]] const_pointer data() const noexcept { return data_.data(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] size_type capacity() const noexcept { return data_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    reference operator[](size_type i) noexcept { return data_[i]; }
    const_reference operator[](size_type i) const noexcept { return data_[i]; }

    [[nodiscard]] allocator_type get_allocator() const noexcept {
        return data_.get_allocator();
    }

    // ========================================================================
    // Modifiers
    // ========================================================================

    void resize(size_type n) { data_.resize(n); }
    void resize(size_type n, const T& value) { data_.resize(n, value); }
    void reserve(size_type n) { data_.reserve(n); }
    void clear() noexcept { data_.clear(); }
    void swap(unified_storage& other) noexcept { data_.swap(other.data_); }

    // ========================================================================
    // Iterators
    // ========================================================================

    auto begin() noexcept { return data_.begin(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto end() const noexcept { return data_.end(); }

    // ========================================================================
    // Placement Queries
    // ========================================================================

    [[nodiscard]] static constexpr bool is_host_accessible() noexcept { return true; }
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept { return true; }
    [[nodiscard]] constexpr int device_id() const noexcept { return -2; }  // Any device

private:
    std::vector<T, allocator_type> data_;
};

#else  // !DTL_ENABLE_CUDA

/// @brief Stub unified storage when CUDA is disabled
template <typename T>
class unified_storage {
    static_assert(sizeof(T) == 0,
        "unified_storage requires CUDA support. Rebuild with -DDTL_ENABLE_CUDA=ON");
};

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Storage Type Selection
// ============================================================================

/// @brief Select the appropriate storage type based on placement policy
/// @tparam T Element type
/// @tparam Placement Placement policy type
template <typename T, typename Placement>
struct select_storage {
    using type = host_storage<T, select_allocator_t<T, Placement>>;
};

#if DTL_ENABLE_CUDA

template <typename T, int DeviceId>
struct select_storage<T, device_only<DeviceId>> {
    static_assert(DeviceStorable<T>,
        "device_only<N> requires a DeviceStorable type (trivially copyable). "
        "Types like std::string or std::vector cannot be used with device placement.");
    using type = device_storage<T>;
};

template <typename T>
struct select_storage<T, device_only_runtime> {
    static_assert(DeviceStorable<T>,
        "device_only_runtime requires a DeviceStorable type (trivially copyable). "
        "Types like std::string or std::vector cannot be used with device placement.");
    using type = device_storage<T>;
};

template <typename T>
struct select_storage<T, unified_memory> {
    static_assert(DeviceStorable<T>,
        "unified_memory requires a DeviceStorable type (trivially copyable). "
        "Types like std::string or std::vector cannot be used with unified memory.");
    using type = unified_storage<T>;
};

#endif  // DTL_ENABLE_CUDA

/// @brief Helper alias for storage type selection
template <typename T, typename Placement>
using select_storage_t = typename select_storage<T, Placement>::type;

// ============================================================================
// Storage Traits
// ============================================================================

/// @brief Query storage traits
template <typename Storage>
struct storage_traits {
    static constexpr bool is_host_accessible = Storage::is_host_accessible();
    static constexpr bool is_device_accessible = Storage::is_device_accessible();
    using tag = typename Storage::storage_tag;
};

}  // namespace detail
}  // namespace dtl
