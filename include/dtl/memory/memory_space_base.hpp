// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory_space_base.hpp
/// @brief Base memory space interface and utilities
/// @details Provides common types and utilities for memory space implementations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#include <cstring>
#include <memory>
#include <new>

namespace dtl {

// ============================================================================
// Allocation Result
// ============================================================================

/// @brief Result of a memory allocation
struct allocation_result {
    /// @brief Pointer to allocated memory (nullptr on failure)
    void* ptr = nullptr;

    /// @brief Actual size allocated (may be >= requested)
    size_type size = 0;

    /// @brief Actual alignment achieved
    size_type alignment = 0;

    /// @brief Check if allocation succeeded
    [[nodiscard]] bool success() const noexcept {
        return ptr != nullptr;
    }

    /// @brief Explicit conversion to bool
    explicit operator bool() const noexcept {
        return success();
    }
};

// ============================================================================
// Memory Space Base
// ============================================================================

/// @brief Abstract base class for memory spaces
/// @details Provides common interface for all memory space implementations.
class memory_space_base {
public:
    using pointer = void*;
    using size_type = dtl::size_type;

    /// @brief Virtual destructor
    virtual ~memory_space_base() = default;

    /// @brief Allocate memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] virtual void* allocate(size_type size) = 0;

    /// @brief Allocate aligned memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] virtual void* allocate(size_type size, size_type alignment) = 0;

    /// @brief Deallocate memory
    /// @param ptr Pointer to memory to deallocate
    /// @param size Size of the allocation
    virtual void deallocate(void* ptr, size_type size) = 0;

    /// @brief Get memory space properties
    [[nodiscard]] virtual memory_space_properties properties() const noexcept = 0;

    /// @brief Get memory space name
    [[nodiscard]] virtual const char* name() const noexcept = 0;

protected:
    memory_space_base() = default;
    memory_space_base(const memory_space_base&) = default;
    memory_space_base& operator=(const memory_space_base&) = default;
    memory_space_base(memory_space_base&&) = default;
    memory_space_base& operator=(memory_space_base&&) = default;
};

// NOTE: memory_space_handle was removed as dead code. It stored a
// memory_space_base* internally but most DTL memory spaces use static
// concept-based interfaces and don't inherit from memory_space_base,
// making the handle unusable in practice. If type-erasure is needed
// in the future, design a proper wrapper over the MemorySpace concept.

// ============================================================================
// Alignment Utilities
// ============================================================================

/// @brief Check if a pointer is aligned
/// @param ptr Pointer to check
/// @param alignment Required alignment
/// @return true if aligned
[[nodiscard]] inline bool is_aligned(const void* ptr, size_type alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

/// @brief Align a size up to the given alignment
/// @param size Size to align
/// @param alignment Alignment requirement
/// @return Aligned size
[[nodiscard]] inline constexpr size_type align_size(size_type size, size_type alignment) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

/// @brief Get the default alignment for a type
/// @tparam T Type
/// @return Default alignment
template <typename T>
[[nodiscard]] constexpr size_type default_alignment() noexcept {
    return alignof(T);
}

/// @brief Get the default alignment for the platform
[[nodiscard]] inline constexpr size_type platform_alignment() noexcept {
    return alignof(std::max_align_t);
}

// ============================================================================
// Memory Utilities
// ============================================================================

/// @brief Fill memory with zeros
/// @param ptr Pointer to memory
/// @param size Size in bytes
inline void zero_memory(void* ptr, size_type size) noexcept {
    std::memset(ptr, 0, size);
}

/// @brief Copy memory
/// @param dst Destination pointer
/// @param src Source pointer
/// @param size Size in bytes
inline void copy_memory(void* dst, const void* src, size_type size) noexcept {
    std::memcpy(dst, src, size);
}

/// @brief Move memory (handles overlapping regions)
/// @param dst Destination pointer
/// @param src Source pointer
/// @param size Size in bytes
inline void move_memory(void* dst, const void* src, size_type size) noexcept {
    std::memmove(dst, src, size);
}

}  // namespace dtl
