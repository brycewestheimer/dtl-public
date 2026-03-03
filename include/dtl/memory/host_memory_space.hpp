// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file host_memory_space.hpp
/// @brief Standard CPU memory space implementation
/// @details Provides memory allocation using standard C++ allocators.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/memory/pinned_memory_space.hpp>

#include <cstdlib>
#include <new>

namespace dtl {

// ============================================================================
// Host Memory Space
// ============================================================================

/// @brief Standard CPU memory space using malloc/free
/// @details Provides basic memory allocation from the system heap.
class host_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = host_memory_space_tag;

    /// @brief Allocate memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size) {
        return std::malloc(size);
    }

    /// @brief Allocate aligned memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
        // Use C++17 aligned allocation if available
#if defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606L
        return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
#else
        // Fallback: allocate extra and align manually
        void* ptr = std::malloc(size + alignment);
        if (!ptr) return nullptr;
        void* aligned = reinterpret_cast<void*>(
            (reinterpret_cast<std::uintptr_t>(ptr) + alignment) & ~(alignment - 1));
        // Store original pointer for deallocation
        reinterpret_cast<void**>(aligned)[-1] = ptr;
        return aligned;
#endif
    }

    /// @brief Deallocate memory
    /// @param ptr Pointer to memory to deallocate
    /// @param size Size of the allocation (unused for standard malloc)
    static void deallocate(void* ptr, [[maybe_unused]] size_type size) noexcept {
        std::free(ptr);
    }

    /// @brief Deallocate aligned memory
    /// @param ptr Pointer to memory to deallocate
    /// @param size Size of the allocation
    /// @param alignment Alignment used for allocation
    static void deallocate(void* ptr, size_type size, size_type alignment) noexcept {
#if defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606L
        ::operator delete(ptr, size, std::align_val_t{alignment});
#else
        // Retrieve original pointer
        void* original = reinterpret_cast<void**>(ptr)[-1];
        std::free(original);
#endif
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = false,
            .unified = false,
            .supports_atomics = true,
            .pageable = true,
            .alignment = alignof(std::max_align_t)
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "host";
    }

    // ========================================================================
    // Typed Allocation
    // ========================================================================

    /// @brief Allocate typed memory
    /// @tparam T Element type
    /// @param count Number of elements
    /// @return Pointer to allocated memory
    template <typename T>
    [[nodiscard]] static T* allocate_typed(size_type count) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    /// @brief Deallocate typed memory
    /// @tparam T Element type
    /// @param ptr Pointer to memory
    /// @param count Number of elements
    template <typename T>
    static void deallocate_typed(T* ptr, size_type count) noexcept {
        deallocate(ptr, count * sizeof(T), alignof(T));
    }

    /// @brief Construct an object in place
    /// @tparam T Object type
    /// @tparam Args Constructor argument types
    /// @param ptr Pointer to memory
    /// @param args Constructor arguments
    template <typename T, typename... Args>
    static void construct(T* ptr, Args&&... args) {
        ::new (static_cast<void*>(ptr)) T(std::forward<Args>(args)...);
    }

    /// @brief Destroy an object
    /// @tparam T Object type
    /// @param ptr Pointer to object
    template <typename T>
    static void destroy(T* ptr) noexcept {
        ptr->~T();
    }
};

// ============================================================================
// Memory Space Traits Specialization
// ============================================================================

template <>
struct memory_space_traits<host_memory_space> {
    static constexpr bool is_host_space = true;
    static constexpr bool is_device_space = false;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = true;
};

// ============================================================================
// Pinned Memory Space (canonical definition in pinned_memory_space.hpp)
// ============================================================================

// ============================================================================
// Default Host Memory Space
// ============================================================================

/// @brief Type alias for the default host memory space
using default_host_space = host_memory_space;

/// @brief Get the default host memory space instance
/// @return Reference to the default host memory space
[[nodiscard]] inline host_memory_space& get_host_memory_space() {
    static host_memory_space space;
    return space;
}

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(MemorySpace<host_memory_space>,
              "host_memory_space must satisfy MemorySpace concept");
static_assert(MemorySpace<pinned_memory_space>,
              "pinned_memory_space must satisfy MemorySpace concept");

}  // namespace dtl
