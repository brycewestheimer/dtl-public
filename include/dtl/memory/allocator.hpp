// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file allocator.hpp
/// @brief Distributed allocator interface
/// @details Provides STL-compatible allocators using memory spaces.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/memory/host_memory_space.hpp>

#include <limits>
#include <memory>

namespace dtl {

// ============================================================================
// Memory Space Allocator
// ============================================================================

/// @brief STL-compatible allocator using a memory space
/// @tparam T Element type
/// @tparam Space Memory space type
template <typename T, MemorySpace Space = host_memory_space>
class memory_space_allocator {
public:
    // C++20 minimum allocator requirements: only value_type is required
    using value_type = T;

    // Propagation traits (these are C++20-valid)
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor
    constexpr memory_space_allocator() noexcept = default;

    /// @brief Copy constructor
    constexpr memory_space_allocator(const memory_space_allocator&) noexcept = default;

    /// @brief Copy constructor from different type
    template <typename U>
    constexpr memory_space_allocator(const memory_space_allocator<U, Space>&) noexcept {}

    /// @brief Destructor
    ~memory_space_allocator() = default;

    // ========================================================================
    // Allocation
    // ========================================================================

    /// @brief Allocate memory for n elements
    /// @param n Number of elements
    /// @return Pointer to allocated memory
    /// @throws std::bad_alloc on failure
    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* ptr = Space::allocate(n * sizeof(T), alignof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    /// @brief Deallocate memory
    /// @param ptr Pointer to memory
    /// @param n Number of elements
    void deallocate(T* ptr, std::size_t n) noexcept {
        if constexpr (requires { Space::deallocate(ptr, n * sizeof(T), alignof(T)); }) {
            Space::deallocate(ptr, n * sizeof(T), alignof(T));
        } else {
            Space::deallocate(ptr, n * sizeof(T));
        }
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    /// @brief Equality comparison (stateless allocators are always equal)
    template <typename U>
    [[nodiscard]] constexpr bool operator==(const memory_space_allocator<U, Space>&) const noexcept {
        return true;
    }

    // ========================================================================
    // Memory Space Access
    // ========================================================================

    /// @brief Get the memory space name
    [[nodiscard]] static constexpr const char* space_name() noexcept {
        return Space::name();
    }

    /// @brief Get the memory space properties
    [[nodiscard]] static constexpr memory_space_properties space_properties() noexcept {
        return Space::properties();
    }
};

// NOTE: polymorphic_allocator was removed as dead code. It depended on
// memory_space_base* (inheritance-based polymorphism), but DTL memory
// spaces use static concept-based interfaces and largely don't inherit
// from memory_space_base, making this allocator unusable in practice.

// ============================================================================
// Allocator Traits Extensions
// ============================================================================

/// @brief Extended traits for DTL allocators
template <typename Alloc>
struct allocator_traits_ext {
    using allocator_type = Alloc;
    using value_type = typename std::allocator_traits<Alloc>::value_type;

    /// @brief Whether allocator uses host memory
    static constexpr bool is_host_allocator = true;

    /// @brief Whether allocator uses device memory
    static constexpr bool is_device_allocator = false;

    /// @brief Whether allocator supports CUDA streams
    static constexpr bool supports_streams = false;
};

/// @brief Traits specialization for memory_space_allocator
template <typename T, MemorySpace Space>
struct allocator_traits_ext<memory_space_allocator<T, Space>> {
    using allocator_type = memory_space_allocator<T, Space>;
    using value_type = T;

    static constexpr bool is_host_allocator = memory_space_traits<Space>::is_host_space;
    static constexpr bool is_device_allocator = memory_space_traits<Space>::is_device_space;
    static constexpr bool supports_streams = false;
};

}  // namespace dtl
