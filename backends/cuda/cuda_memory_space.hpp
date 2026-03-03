// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_memory_space.hpp
/// @brief CUDA device memory space implementation
/// @details Provides memory allocation and management for CUDA device memory.
///          Instance-based cuda_memory_space is defined here; static interface
///          classes (cuda_device_memory_space, cuda_unified_memory_space) are
///          defined in include/dtl/memory/cuda_memory_space.hpp and re-exported.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/memory/cuda_memory_space.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <atomic>
#include <memory>

namespace dtl {
namespace cuda {

// ============================================================================
// CUDA Device ID
// ============================================================================

/// @brief Represents a CUDA device identifier
using device_id_t = int;

/// @brief Invalid device ID sentinel
inline constexpr device_id_t invalid_device = -1;

/// @brief Get the current CUDA device
/// @return Current device ID or invalid_device if CUDA not enabled
[[nodiscard]] inline device_id_t current_device() noexcept {
#if DTL_ENABLE_CUDA
    int device;
    if (cudaGetDevice(&device) == cudaSuccess) {
        return device;
    }
#endif
    return invalid_device;
}

/// @brief Get the number of CUDA devices
/// @return Number of devices or 0 if CUDA not enabled
[[nodiscard]] inline int device_count() noexcept {
#if DTL_ENABLE_CUDA
    int count;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        return count;
    }
#endif
    return 0;
}

// ============================================================================
// CUDA Memory Space
// ============================================================================

/// @brief CUDA device memory space
/// @details Manages allocations in CUDA device (GPU) memory.
///          Satisfies the MemorySpace concept.
class cuda_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = device_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = false;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Default constructor (uses current device)
    cuda_memory_space()
        : device_id_(current_device()) {}

    /// @brief Construct for specific device
    /// @param device_id CUDA device ID
    explicit cuda_memory_space(device_id_t device_id)
        : device_id_(device_id) {}

    /// @brief Destructor
    ~cuda_memory_space() = default;

    // Non-copyable
    cuda_memory_space(const cuda_memory_space&) = delete;
    cuda_memory_space& operator=(const cuda_memory_space&) = delete;

    // Movable (custom: std::atomic is not movable)
    cuda_memory_space(cuda_memory_space&& other) noexcept
        : device_id_(other.device_id_)
        , total_allocated_(other.total_allocated_.load(std::memory_order_relaxed))
        , peak_allocated_(other.peak_allocated_.load(std::memory_order_relaxed)) {
        other.device_id_ = invalid_device;
        other.total_allocated_.store(0, std::memory_order_relaxed);
        other.peak_allocated_.store(0, std::memory_order_relaxed);
    }

    cuda_memory_space& operator=(cuda_memory_space&& other) noexcept {
        if (this != &other) {
            device_id_ = other.device_id_;
            total_allocated_.store(other.total_allocated_.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            peak_allocated_.store(other.peak_allocated_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            other.device_id_ = invalid_device;
            other.total_allocated_.store(0, std::memory_order_relaxed);
            other.peak_allocated_.store(0, std::memory_order_relaxed);
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // MemorySpace Concept Interface
    // ------------------------------------------------------------------------

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cuda_device";
    }

    /// @brief Get memory space properties
    [[nodiscard]] memory_space_properties properties() const noexcept {
        return memory_space_properties{
            .host_accessible = false,
            .device_accessible = true,
            .unified = false,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256  // CUDA default alignment
        };
    }

    /// @brief Allocate device memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] void* allocate(size_type size) {
#if DTL_ENABLE_CUDA
        // Set device if needed
        if (device_id_ != invalid_device) {
            cudaError_t err = cudaSetDevice(device_id_);
            if (err != cudaSuccess) {
                return nullptr;
            }
        }

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }

        auto new_total = total_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        auto current_peak = peak_allocated_.load(std::memory_order_relaxed);
        while (new_total > current_peak &&
               !peak_allocated_.compare_exchange_weak(current_peak, new_total,
                   std::memory_order_relaxed)) {}

        return ptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate aligned device memory
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    /// @note CUDA guarantees 256-byte alignment from cudaMalloc.
    ///       Alignments > 256 are rejected to prevent memory leaks.
    [[nodiscard]] void* allocate(size_type size, size_type alignment) {
#if DTL_ENABLE_CUDA
        // CUDA allocations are already 256-byte aligned from cudaMalloc.
        // For alignments <= 256, regular allocation is sufficient.
        if (alignment <= 256) {
            return allocate(size);
        }

        // Reject alignments > 256: over-allocating loses the original pointer
        // returned by cudaMalloc, causing memory leaks and UB on cudaFree.
        return nullptr;
#else
        (void)size; (void)alignment;
        return nullptr;
#endif
    }

    /// @brief Deallocate device memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation
    void deallocate(void* ptr, size_type size) noexcept {
#if DTL_ENABLE_CUDA
        if (ptr == nullptr) return;

        cudaError_t err = cudaFree(ptr);
        if (err == cudaSuccess) {
            total_allocated_.fetch_sub(size, std::memory_order_relaxed);
        }
#else
        (void)ptr; (void)size;
#endif
    }

    /// @brief Check if pointer is in this memory space
    /// @param ptr Pointer to check
    /// @return true if pointer is device memory
    [[nodiscard]] bool contains(const void* ptr) const noexcept {
#if DTL_ENABLE_CUDA
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
        if (err != cudaSuccess) {
            cudaGetLastError();  // Clear error
            return false;
        }
        return attrs.type == cudaMemoryTypeDevice;
#else
        (void)ptr;
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // CUDA-Specific Methods
    // ------------------------------------------------------------------------

    /// @brief Get the device ID for this memory space
    [[nodiscard]] device_id_t device_id() const noexcept { return device_id_; }

    /// @brief Get total allocated bytes
    [[nodiscard]] size_type total_allocated() const noexcept {
        return total_allocated_.load(std::memory_order_relaxed);
    }

    /// @brief Get peak allocated bytes
    [[nodiscard]] size_type peak_allocated() const noexcept {
        return peak_allocated_.load(std::memory_order_relaxed);
    }

    /// @brief Get available device memory
    [[nodiscard]] size_type available_memory() const noexcept {
#if DTL_ENABLE_CUDA
        if (device_id_ != invalid_device) {
            cudaSetDevice(device_id_);
        }

        size_t free_mem, total_mem;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess) {
            return 0;
        }
        return static_cast<size_type>(free_mem);
#else
        return 0;
#endif
    }

    /// @brief Memset device memory
    /// @param ptr Device pointer
    /// @param value Value to set
    /// @param size Number of bytes
    void memset(void* ptr, int value, size_type size) noexcept {
#if DTL_ENABLE_CUDA
        cudaMemset(ptr, value, size);
#else
        (void)ptr; (void)value; (void)size;
#endif
    }

private:
    device_id_t device_id_ = invalid_device;
    std::atomic<size_type> total_allocated_{0};
    std::atomic<size_type> peak_allocated_{0};
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(MemorySpace<cuda_memory_space>,
              "cuda_memory_space must satisfy MemorySpace concept");

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the default CUDA memory space for current device
/// @return Reference to default memory space
[[nodiscard]] inline cuda_memory_space& default_cuda_memory_space() {
    static cuda_memory_space space;
    return space;
}

/// @brief Create a CUDA memory space for a specific device
/// @param device_id Device ID
/// @return Memory space for the device
[[nodiscard]] inline std::unique_ptr<cuda_memory_space>
make_cuda_memory_space(device_id_t device_id) {
    return std::make_unique<cuda_memory_space>(device_id);
}

// Static interface memory spaces (cuda_device_memory_space, cuda_unified_memory_space)
// and their memory_space_traits are defined in include/dtl/memory/cuda_memory_space.hpp
// and included at the top of this file to avoid ODR violations.

}  // namespace cuda
}  // namespace dtl
