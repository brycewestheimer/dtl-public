// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_device_memory_space.hpp
/// @brief Device-specific CUDA memory space implementations
/// @details Provides memory spaces that allocate on specific CUDA devices,
///          using device guards to ensure correct device selection.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/cuda/device_guard.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {
namespace cuda {

// ============================================================================
// Device-Specific Memory Space (Compile-Time Device ID)
// ============================================================================

/// @brief CUDA memory space for a specific device (compile-time device ID)
/// @tparam DeviceId The target CUDA device ID
/// @details Allocations and deallocations are guarded to ensure they occur
///          on the specified device, regardless of the current CUDA context.
///          The previous device is restored after each operation.
///
/// @par Thread Safety
/// Each thread's device context is managed independently. Device guards
/// ensure no cross-thread device confusion.
///
/// @par Example
/// @code
/// using gpu0_space = dtl::cuda::cuda_device_memory_space_for<0>;
/// using gpu1_space = dtl::cuda::cuda_device_memory_space_for<1>;
///
/// void* p0 = gpu0_space::allocate(1024);  // Always on device 0
/// void* p1 = gpu1_space::allocate(1024);  // Always on device 1
/// @endcode
template <int DeviceId>
class cuda_device_memory_space_for {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = device_memory_space_tag;

    /// @brief The compile-time device ID
    static constexpr int device_id = DeviceId;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = false;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Allocate device memory on the specified device
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size) noexcept {
#if DTL_ENABLE_CUDA
        device_guard guard(DeviceId);
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

    /// @brief Allocate aligned device memory on the specified device
    /// @param size Number of bytes to allocate
    /// @param alignment Alignment requirement (CUDA already 256-byte aligned)
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] static void* allocate(size_type size, size_type alignment) noexcept {
        // CUDA allocations are 256-byte aligned by default
        (void)alignment;
        return allocate(size);
    }

    /// @brief Deallocate device memory on the specified device
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation (unused)
    static void deallocate(void* ptr, [[maybe_unused]] size_type size) noexcept {
#if DTL_ENABLE_CUDA
        if (ptr != nullptr) {
            device_guard guard(DeviceId);
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

    /// @brief Get the device ID for this memory space
    [[nodiscard]] static constexpr int get_device_id() noexcept {
        return DeviceId;
    }

    /// @brief Query device memory info for this device
    /// @param free_bytes Output: free memory in bytes
    /// @param total_bytes Output: total memory in bytes
    /// @return true on success, false on failure
    [[nodiscard]] static bool memory_info(size_type& free_bytes, size_type& total_bytes) noexcept {
#if DTL_ENABLE_CUDA
        device_guard guard(DeviceId);
        size_t free_mem, total_mem;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err == cudaSuccess) {
            free_bytes = static_cast<size_type>(free_mem);
            total_bytes = static_cast<size_type>(total_mem);
            return true;
        }
#else
        (void)free_bytes;
        (void)total_bytes;
#endif
        return false;
    }
};

// ============================================================================
// Runtime Device Memory Space
// ============================================================================

/// @brief CUDA memory space with runtime device ID
/// @details Stateful memory space that stores the device ID at runtime.
///          Compatible with polymorphic allocator patterns.
///
/// @par Example
/// @code
/// int gpu_id = get_available_gpu();
/// cuda_device_memory_space_runtime space(gpu_id);
/// void* ptr = space.allocate(1024);
/// @endcode
class cuda_device_memory_space_runtime {
public:
    using pointer = void*;
    using size_type = dtl::size_type;
    using tag_type = device_memory_space_tag;

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = false;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Default constructor (uses device 0)
    cuda_device_memory_space_runtime() noexcept : device_id_(0) {}

    /// @brief Construct for specific device
    /// @param device_id CUDA device ID
    explicit cuda_device_memory_space_runtime(int device_id) noexcept
        : device_id_(device_id) {}

    /// @brief Allocate device memory on the configured device
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] void* allocate(size_type size) const noexcept {
#if DTL_ENABLE_CUDA
        device_guard guard(device_id_);
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
    /// @param alignment Alignment requirement
    /// @return Pointer to allocated memory or nullptr on failure
    [[nodiscard]] void* allocate(size_type size, size_type alignment) const noexcept {
        (void)alignment;
        return allocate(size);
    }

    /// @brief Deallocate device memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation (unused)
    void deallocate(void* ptr, [[maybe_unused]] size_type size) const noexcept {
#if DTL_ENABLE_CUDA
        if (ptr != nullptr) {
            device_guard guard(device_id_);
            cudaFree(ptr);
        }
#else
        (void)ptr;
#endif
    }

    /// @brief Get memory space properties
    [[nodiscard]] memory_space_properties properties() const noexcept {
        return memory_space_properties{
            .host_accessible = false,
            .device_accessible = true,
            .unified = false,
            .supports_atomics = true,
            .pageable = false,
            .alignment = 256
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cuda_device_runtime";
    }

    /// @brief Get the device ID for this memory space
    [[nodiscard]] int get_device_id() const noexcept {
        return device_id_;
    }

    /// @brief Equality comparison (same device ID)
    [[nodiscard]] bool operator==(const cuda_device_memory_space_runtime& other) const noexcept {
        return device_id_ == other.device_id_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] bool operator!=(const cuda_device_memory_space_runtime& other) const noexcept {
        return device_id_ != other.device_id_;
    }

private:
    int device_id_;
};

// ============================================================================
// Pointer Attribute Helpers
// ============================================================================

/// @brief Query which device a pointer was allocated on
/// @param ptr Device pointer to query
/// @return Device ID or invalid_device_id on error
[[nodiscard]] inline int get_pointer_device(const void* ptr) noexcept {
#if DTL_ENABLE_CUDA
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err == cudaSuccess) {
        return attrs.device;
    }
    cudaGetLastError();  // Clear error state
#else
    (void)ptr;
#endif
    return invalid_device_id;
}

/// @brief Check if a pointer is device memory
/// @param ptr Pointer to check
/// @return true if pointer is device memory
[[nodiscard]] inline bool is_device_pointer(const void* ptr) noexcept {
#if DTL_ENABLE_CUDA
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err == cudaSuccess) {
        return attrs.type == cudaMemoryTypeDevice;
    }
    cudaGetLastError();  // Clear error state
#else
    (void)ptr;
#endif
    return false;
}

}  // namespace cuda

// ============================================================================
// Memory Space Traits for Device-Specific Memory Spaces
// ============================================================================

/// @brief Traits specialization for cuda_device_memory_space_for
template <int DeviceId>
struct memory_space_traits<cuda::cuda_device_memory_space_for<DeviceId>> {
    static constexpr bool is_host_space = false;
    static constexpr bool is_device_space = true;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = true;  // Guarded operations are thread-safe
    static constexpr int device_id = DeviceId;
};

/// @brief Traits specialization for cuda_device_memory_space_runtime
template <>
struct memory_space_traits<cuda::cuda_device_memory_space_runtime> {
    static constexpr bool is_host_space = false;
    static constexpr bool is_device_space = true;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = true;  // Guarded operations are thread-safe
};

}  // namespace dtl
