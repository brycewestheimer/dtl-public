// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hip_memory_space.hpp
/// @brief HIP device memory space implementation
/// @details Provides memory allocation and management for AMD GPU memory via HIP.
///          Uses concept-based static dispatch (matching CUDA's pattern) with
///          std::atomic allocation tracking for thread safety.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <atomic>
#include <memory>

namespace dtl {
namespace hip {

// ============================================================================
// HIP Device ID
// ============================================================================

/// @brief Represents a HIP device identifier
using device_id_t = int;

/// @brief Invalid device ID sentinel
inline constexpr device_id_t invalid_device = -1;

/// @brief Get the current HIP device
/// @return Current device ID or invalid_device if HIP not enabled
[[nodiscard]] inline device_id_t current_device() noexcept {
#if DTL_ENABLE_HIP
    int device;
    if (hipGetDevice(&device) == hipSuccess) {
        return device;
    }
#endif
    return invalid_device;
}

/// @brief Get the number of HIP devices
/// @return Number of devices or 0 if HIP not enabled
[[nodiscard]] inline int device_count() noexcept {
#if DTL_ENABLE_HIP
    int count;
    if (hipGetDeviceCount(&count) == hipSuccess) {
        return count;
    }
#endif
    return 0;
}

// ============================================================================
// HIP Memory Space
// ============================================================================

/// @brief HIP device memory space
/// @details Manages allocations in HIP device (AMD GPU) memory.
///          Satisfies the MemorySpace concept via static dispatch (no virtual).
class hip_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = device_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = false;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Default constructor (uses current device)
    hip_memory_space()
        : device_id_(current_device()) {}

    /// @brief Construct for specific device
    /// @param device_id HIP device ID
    explicit hip_memory_space(device_id_t device_id)
        : device_id_(device_id) {}

    /// @brief Destructor
    ~hip_memory_space() = default;

    // Non-copyable
    hip_memory_space(const hip_memory_space&) = delete;
    hip_memory_space& operator=(const hip_memory_space&) = delete;

    // Movable (custom: std::atomic is not movable)
    hip_memory_space(hip_memory_space&& other) noexcept
        : device_id_(other.device_id_)
        , total_allocated_(other.total_allocated_.load(std::memory_order_relaxed))
        , peak_allocated_(other.peak_allocated_.load(std::memory_order_relaxed)) {
        other.device_id_ = invalid_device;
        other.total_allocated_.store(0, std::memory_order_relaxed);
        other.peak_allocated_.store(0, std::memory_order_relaxed);
    }

    hip_memory_space& operator=(hip_memory_space&& other) noexcept {
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
        return "hip_device";
    }

    /// @brief Get memory space properties
    [[nodiscard]] memory_space_properties properties() const noexcept {
        return memory_space_properties{
            .host_accessible = false,
            .device_accessible = true,
            .unified = false,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256  // HIP default alignment
        };
    }

    /// @brief Allocate device memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] void* allocate(size_type size) {
#if DTL_ENABLE_HIP
        if (device_id_ != invalid_device) {
            hipError_t err = hipSetDevice(device_id_);
            if (err != hipSuccess) {
                return nullptr;
            }
        }

        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
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
    /// @note HIP guarantees 256-byte alignment from hipMalloc.
    ///       Alignments > 256 are rejected to prevent memory leaks.
    [[nodiscard]] void* allocate(size_type size, size_type alignment) {
#if DTL_ENABLE_HIP
        if (alignment <= 256) {
            return allocate(size);
        }
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
#if DTL_ENABLE_HIP
        if (ptr == nullptr) return;

        hipError_t err = hipFree(ptr);
        if (err == hipSuccess) {
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
#if DTL_ENABLE_HIP
        hipPointerAttribute_t attrs;
        hipError_t err = hipPointerGetAttributes(&attrs, ptr);
        if (err != hipSuccess) {
            return false;
        }
        return attrs.memoryType == hipMemoryTypeDevice;
#else
        (void)ptr;
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // HIP-Specific Methods
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
#if DTL_ENABLE_HIP
        if (device_id_ != invalid_device) {
            hipSetDevice(device_id_);
        }

        size_t free_mem, total_mem;
        hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
        if (err != hipSuccess) {
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
#if DTL_ENABLE_HIP
        hipMemset(ptr, value, size);
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

static_assert(MemorySpace<hip_memory_space>,
              "hip_memory_space must satisfy MemorySpace concept");

// ============================================================================
// HIP Managed Memory Space
// ============================================================================

/// @brief HIP managed (unified) memory space
/// @details Memory accessible from both host and device with automatic migration.
///          Satisfies the MemorySpace concept via static dispatch.
class hip_managed_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = unified_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = true;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Default constructor
    hip_managed_memory_space() = default;

    /// @brief Destructor
    ~hip_managed_memory_space() = default;

    // Non-copyable
    hip_managed_memory_space(const hip_managed_memory_space&) = delete;
    hip_managed_memory_space& operator=(const hip_managed_memory_space&) = delete;

    // Movable
    hip_managed_memory_space(hip_managed_memory_space&&) = default;
    hip_managed_memory_space& operator=(hip_managed_memory_space&&) = default;

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "hip_managed";
    }

    /// @brief Get memory space properties
    [[nodiscard]] memory_space_properties properties() const noexcept {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = true,
            .unified = true,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256
        };
    }

    /// @brief Allocate managed memory
    [[nodiscard]] void* allocate(size_type size) {
#if DTL_ENABLE_HIP
        void* ptr = nullptr;
        hipError_t err = hipMallocManaged(&ptr, size);
        if (err != hipSuccess) {
            return nullptr;
        }
        return ptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    /// @brief Allocate aligned managed memory
    [[nodiscard]] void* allocate(size_type size, size_type alignment) {
#if DTL_ENABLE_HIP
        if (alignment <= 256) {
            return allocate(size);
        }
        return nullptr;
#else
        (void)size; (void)alignment;
        return nullptr;
#endif
    }

    /// @brief Deallocate managed memory
    void deallocate(void* ptr, size_type size) noexcept {
#if DTL_ENABLE_HIP
        (void)size;
        if (ptr == nullptr) return;
        hipFree(ptr);
#else
        (void)ptr; (void)size;
#endif
    }

    /// @brief Check if pointer is managed memory
    [[nodiscard]] bool contains(const void* ptr) const noexcept {
#if DTL_ENABLE_HIP
        hipPointerAttribute_t attrs;
        hipError_t err = hipPointerGetAttributes(&attrs, ptr);
        if (err != hipSuccess) {
            return false;
        }
        return attrs.isManaged;
#else
        (void)ptr;
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // Prefetch and Memory Advice
    // ------------------------------------------------------------------------

    /// @brief Prefetch managed memory to a specific device
    result<void> prefetch_to_device(void* ptr, size_type size, int device_id = 0) {
#if DTL_ENABLE_HIP
        hipError_t err = hipMemPrefetchAsync(ptr, size, device_id, nullptr);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemPrefetchAsync to device failed");
        }
        return {};
#else
        (void)ptr; (void)size; (void)device_id;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Prefetch managed memory to the host
    result<void> prefetch_to_host(void* ptr, size_type size) {
#if DTL_ENABLE_HIP
        hipError_t err = hipMemPrefetchAsync(ptr, size, hipCpuDeviceId, nullptr);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemPrefetchAsync to host failed");
        }
        return {};
#else
        (void)ptr; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Set memory advice for managed allocation
    result<void> advise(void* ptr, size_type size, int advice, int device_id = 0) {
#if DTL_ENABLE_HIP
        hipError_t err = hipMemAdvise(ptr, size,
                                      static_cast<hipMemoryAdvise>(advice),
                                      device_id);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemAdvise failed");
        }
        return {};
#else
        (void)ptr; (void)size; (void)advice; (void)device_id;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(MemorySpace<hip_managed_memory_space>,
              "hip_managed_memory_space must satisfy MemorySpace concept");

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the default HIP memory space for current device
[[nodiscard]] inline hip_memory_space& default_hip_memory_space() {
    static hip_memory_space space;
    return space;
}

/// @brief Create a HIP memory space for a specific device
[[nodiscard]] inline std::unique_ptr<hip_memory_space>
make_hip_memory_space(device_id_t device_id) {
    return std::make_unique<hip_memory_space>(device_id);
}

/// @brief Get the default managed memory space
[[nodiscard]] inline hip_managed_memory_space& default_hip_managed_memory_space() {
    static hip_managed_memory_space space;
    return space;
}

}  // namespace hip
}  // namespace dtl
