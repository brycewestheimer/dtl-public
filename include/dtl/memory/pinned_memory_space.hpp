// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file pinned_memory_space.hpp
/// @brief Pinned (page-locked) host memory space
/// @details Provides a memory space for pinned host memory that enables
///          faster host-device transfers. Falls back to `std::malloc`/`std::free`
///          when no GPU backend is available.
///
///          This is the canonical definition of pinned_memory_space.
///          It satisfies the MemorySpace concept with a static interface.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/memory/memory_space_base.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#elif DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace dtl {

// ============================================================================
// Pinned Memory Space
// ============================================================================

/// @brief Page-locked (pinned) host memory space
/// @details Memory that cannot be swapped out, enabling faster GPU transfers.
///          Uses cudaMallocHost (CUDA), hipHostMalloc (HIP), or std::malloc
///          (fallback) for allocation. Satisfies the MemorySpace concept.
///
/// @par Thread Safety
/// Allocation statistics are tracked with atomic counters (relaxed ordering)
/// so that concurrent allocate/deallocate calls from multiple threads do not
/// cause data races. The allocators themselves (cudaMallocHost, malloc) are
/// already thread-safe.
class pinned_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = pinned_memory_space_tag;

    // ------------------------------------------------------------------------
    // MemorySpace Concept Interface (static)
    // ------------------------------------------------------------------------

    /// @brief Allocate pinned host memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size) {
        if (size == 0) {
            return nullptr;
        }

        void* ptr = nullptr;

#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }
#elif DTL_ENABLE_HIP
        hipError_t err = hipHostMalloc(&ptr, size);
        if (err != hipSuccess) {
            return nullptr;
        }
#else
        ptr = std::malloc(size);
        if (ptr == nullptr) {
            return nullptr;
        }
#endif

        // Thread-safe statistics update (relaxed ordering is sufficient)
        auto current = total_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        // Atomic max via CAS loop
        auto peak = peak_allocated_.load(std::memory_order_relaxed);
        while (current > peak &&
               !peak_allocated_.compare_exchange_weak(peak, current,
                                                      std::memory_order_relaxed)) {}

        return ptr;
    }

    /// @brief Allocate aligned pinned host memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement (pinned memory is already well-aligned)
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) {
        // Pinned allocations (cudaMallocHost, malloc) are already well-aligned
        (void)alignment;
        return allocate(size);
    }

    /// @brief Deallocate pinned host memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation
    static void deallocate(void* ptr, size_type size) noexcept {
        if (ptr == nullptr) return;

#if DTL_ENABLE_CUDA
        cudaFreeHost(ptr);
#elif DTL_ENABLE_HIP
        hipHostFree(ptr);
#else
        std::free(ptr);
#endif

        // Thread-safe statistics update
        total_allocated_.fetch_sub(size, std::memory_order_relaxed);
    }

    /// @brief Get memory space properties
    [[nodiscard]] static constexpr memory_space_properties properties() noexcept {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = true,  // Accessible from GPU via DMA
            .unified = false,
            .supports_atomics = true,
            .pageable = false,  // Page-locked
            .alignment = alignof(std::max_align_t)
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "pinned";
    }

    // ------------------------------------------------------------------------
    // Statistics (thread-safe)
    // ------------------------------------------------------------------------

    /// @brief Get total currently allocated bytes
    [[nodiscard]] static size_type total_allocated() noexcept {
        return total_allocated_.load(std::memory_order_relaxed);
    }

    /// @brief Get peak allocated bytes
    [[nodiscard]] static size_type peak_allocated() noexcept {
        return peak_allocated_.load(std::memory_order_relaxed);
    }

    /// @brief Reset statistics counters (for testing)
    static void reset_statistics() noexcept {
        total_allocated_.store(0, std::memory_order_relaxed);
        peak_allocated_.store(0, std::memory_order_relaxed);
    }

private:
    static inline std::atomic<size_type> total_allocated_{0};
    static inline std::atomic<size_type> peak_allocated_{0};
};

// ============================================================================
// Memory Space Traits Specialization
// ============================================================================

/// @brief Traits specialization for pinned_memory_space
template <>
struct memory_space_traits<pinned_memory_space> {
    static constexpr bool is_host_space = true;
    static constexpr bool is_device_space = false;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = true;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(MemorySpace<pinned_memory_space>,
              "pinned_memory_space must satisfy MemorySpace concept");

}  // namespace dtl
