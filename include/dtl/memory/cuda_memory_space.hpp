// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_memory_space.hpp
/// @brief Static CUDA memory space interfaces for allocator integration
/// @details Provides static interface memory spaces compatible with memory_space_allocator
///          for use with placement policies.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {
namespace cuda {

// ============================================================================
// Static Interface Memory Spaces (for allocator compatibility)
// ============================================================================

/// @brief Static interface for CUDA device memory (uses device 0)
/// @details This class provides a static interface compatible with
///          memory_space_allocator. Uses static methods for allocation.
class cuda_device_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = device_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = false;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Allocate device memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size) {
#if DTL_ENABLE_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate aligned device memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement (CUDA already 256-byte aligned)
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
        // CUDA allocations are 256-byte aligned by default
        (void)alignment;
        return allocate(size);
    }

    /// @brief Deallocate device memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation (unused)
    static void deallocate(void* ptr, [[maybe_unused]] size_type size) noexcept {
#if DTL_ENABLE_CUDA
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
#else
        (void)ptr;
#endif
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return memory_space_properties{
            .host_accessible = false,
            .device_accessible = true,
            .unified = false,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256  // CUDA default alignment
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cuda_device";
    }
};

/// @brief Static interface for CUDA unified (managed) memory
/// @details Provides memory accessible from both host and device
///          with automatic page migration.
class cuda_unified_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = unified_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = true;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Allocate unified memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size) {
#if DTL_ENABLE_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate aligned unified memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
        // CUDA managed memory allocations are 256-byte aligned by default
        (void)alignment;
        return allocate(size);
    }

    /// @brief Deallocate unified memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation (unused)
    static void deallocate(void* ptr, [[maybe_unused]] size_type size) noexcept {
#if DTL_ENABLE_CUDA
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
#else
        (void)ptr;
#endif
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = true,
            .unified = true,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256  // CUDA default alignment
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cuda_unified";
    }

    /// @brief Prefetch memory to device
    /// @param ptr Pointer to managed memory
    /// @param size Number of bytes to prefetch
    /// @param device_id Target device ID (default: 0)
    static void prefetch_to_device(void* ptr, size_type size, int device_id = 0) {
#if DTL_ENABLE_CUDA
        cudaMemPrefetchAsync(ptr, size, device_id, nullptr);
#else
        (void)ptr; (void)size; (void)device_id;
#endif
    }

    /// @brief Prefetch memory to host
    /// @param ptr Pointer to managed memory
    /// @param size Number of bytes to prefetch
    static void prefetch_to_host(void* ptr, size_type size) {
#if DTL_ENABLE_CUDA
        cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, nullptr);
#else
        (void)ptr; (void)size;
#endif
    }
};

}  // namespace cuda

// ============================================================================
// Memory Space Traits for Static Interfaces
// ============================================================================

/// @brief Traits specialization for cuda_device_memory_space
template <>
struct memory_space_traits<cuda::cuda_device_memory_space> {
    static constexpr bool is_host_space = false;
    static constexpr bool is_device_space = true;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = false;  // CUDA allocation not thread-safe
};

/// @brief Traits specialization for cuda_unified_memory_space
template <>
struct memory_space_traits<cuda::cuda_unified_memory_space> {
    static constexpr bool is_host_space = true;  // Accessible from host
    static constexpr bool is_device_space = true;  // Accessible from device
    static constexpr bool is_unified_space = true;
    static constexpr bool is_thread_safe = false;  // CUDA allocation not thread-safe
};

// ============================================================================
// Concept Verification for Static Interfaces
// ============================================================================

static_assert(MemorySpace<cuda::cuda_device_memory_space>,
              "cuda_device_memory_space must satisfy MemorySpace concept");
static_assert(MemorySpace<cuda::cuda_unified_memory_space>,
              "cuda_unified_memory_space must satisfy MemorySpace concept");

}  // namespace dtl
