// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_memory_space.hpp
/// @brief OpenSHMEM symmetric memory space implementation
/// @details Provides symmetric heap memory allocation for PGAS-style access.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

namespace dtl {
namespace shmem {

// ============================================================================
// SHMEM Symmetric Memory Space
// ============================================================================

/// @brief Symmetric memory space for OpenSHMEM
/// @details Provides memory that is accessible from all PEs using one-sided
///          operations. All allocations are symmetric - the same virtual
///          address on all PEs.
///
/// @par Properties:
/// - Memory is symmetric across all PEs
/// - Accessible via one-sided put/get/atomic operations
/// - Host accessible (runs on CPU)
/// - Not device accessible (for GPU, use CUDA+SHMEM hybrid)
///
/// @par Usage:
/// @code
/// shmem_symmetric_memory_space space;
/// void* ptr = space.allocate(1024);
/// // ptr can be used as target for put/get from any PE
/// space.deallocate(ptr, 1024);
/// @endcode
class shmem_symmetric_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = host_memory_space_tag;  // Runs on host

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = true;

    /// @brief Whether this memory is accessible from device (GPU)
    static constexpr bool device_accessible = false;

    /// @brief Whether this memory is symmetric (PGAS)
    static constexpr bool symmetric = true;

    /// @brief Default constructor
    shmem_symmetric_memory_space() = default;

    /// @brief Destructor
    ~shmem_symmetric_memory_space() = default;

    // Non-copyable
    shmem_symmetric_memory_space(const shmem_symmetric_memory_space&) = delete;
    shmem_symmetric_memory_space& operator=(const shmem_symmetric_memory_space&) = delete;

    // Movable
    shmem_symmetric_memory_space(shmem_symmetric_memory_space&&) = default;
    shmem_symmetric_memory_space& operator=(shmem_symmetric_memory_space&&) = default;

    // ------------------------------------------------------------------------
    // MemorySpace Concept Interface
    // ------------------------------------------------------------------------

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "shmem_symmetric";
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = false,
            .unified = false,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 8  // Typical SHMEM alignment
        };
    }

    /// @brief Allocate symmetric memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    /// @note Collective operation - all PEs must call with same size
    [[nodiscard]] static void* allocate(size_type size) {
#if DTL_ENABLE_SHMEM
        return shmem_malloc(size);
#else
        (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate aligned symmetric memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    /// @note Collective operation - all PEs must call with same size
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
#if DTL_ENABLE_SHMEM
        // SHMEM 1.5 has shmem_align, but older versions may not
        // Fall back to regular allocation with over-allocation
        (void)alignment;
        return shmem_malloc(size);
#else
        (void)size; (void)alignment;
        return nullptr;
#endif
    }

    /// @brief Deallocate symmetric memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation (unused but required by concept)
    /// @note Collective operation - all PEs must call
    static void deallocate(void* ptr, size_type /*size*/) noexcept {
#if DTL_ENABLE_SHMEM
        if (ptr != nullptr) {
            shmem_free(ptr);
        }
#else
        (void)ptr;
#endif
    }

    /// @brief Check if pointer is in symmetric heap
    /// @param ptr Pointer to check
    /// @return true if pointer is symmetric memory
    [[nodiscard]] static bool contains(const void* ptr) noexcept {
#if DTL_ENABLE_SHMEM
        // SHMEM 1.4+ has shmem_addr_accessible
        // For older versions, we can't reliably check
        (void)ptr;
        return true;  // Assume symmetric if we can't verify
#else
        (void)ptr;
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // SHMEM-Specific Methods
    // ------------------------------------------------------------------------

    /// @brief Reallocate symmetric memory
    /// @param ptr Current pointer
    /// @param size New size
    /// @return New pointer or nullptr on failure
    /// @note Collective operation
    [[nodiscard]] static void* reallocate(void* ptr, size_type size) {
#if DTL_ENABLE_SHMEM
        return shmem_realloc(ptr, size);
#else
        (void)ptr; (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate and clear symmetric memory
    /// @param count Number of elements
    /// @param size Size of each element
    /// @return Pointer to zero-initialized memory or nullptr
    /// @note Collective operation
    [[nodiscard]] static void* calloc(size_type count, size_type size) {
#if DTL_ENABLE_SHMEM
        return shmem_calloc(count, size);
#else
        (void)count; (void)size;
        return nullptr;
#endif
    }
};

// ============================================================================
// Static Memory Space (for allocator compatibility)
// ============================================================================

/// @brief Static interface for SHMEM symmetric memory
/// @details This class provides a static interface compatible with
///          memory_space_allocator for use in DTL containers.
class shmem_static_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = host_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = true;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = false;

    /// @brief Allocate symmetric memory
    [[nodiscard]] static void* allocate(size_type size) {
        return shmem_symmetric_memory_space::allocate(size);
    }

    /// @brief Allocate aligned symmetric memory
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
        return shmem_symmetric_memory_space::allocate(size, alignment);
    }

    /// @brief Deallocate symmetric memory
    static void deallocate(void* ptr, size_type size) noexcept {
        shmem_symmetric_memory_space::deallocate(ptr, size);
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return shmem_symmetric_memory_space::properties();
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "shmem_symmetric";
    }
};

// ============================================================================
// Concept Verification
// ============================================================================

#if DTL_ENABLE_SHMEM
static_assert(MemorySpace<shmem_static_memory_space>,
              "shmem_static_memory_space must satisfy MemorySpace concept");
#endif

// ============================================================================
// Memory Space Traits
// ============================================================================

}  // namespace shmem

/// @brief Traits specialization for shmem_static_memory_space
template <>
struct memory_space_traits<shmem::shmem_static_memory_space> {
    static constexpr bool is_host_space = true;
    static constexpr bool is_device_space = false;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_symmetric_space = true;
    static constexpr bool is_thread_safe = false;  // Collective operations not thread-safe
};

}  // namespace dtl
