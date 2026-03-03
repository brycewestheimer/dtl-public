// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_managed_memory_space.hpp
/// @brief CUDA unified/managed memory space implementation
/// @details Provides memory allocation using CUDA managed memory for
///          automatic data migration between host and device.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/memory/memory_space_base.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>

namespace dtl {
namespace cuda {

// ============================================================================
// Managed Memory Hints
// ============================================================================

/// @brief Hints for managed memory allocation and migration
enum class managed_memory_hint {
    /// @brief No specific hint (let driver decide)
    none,

    /// @brief Prefer to keep data on device
    prefer_device,

    /// @brief Prefer to keep data on host
    prefer_host,

    /// @brief Data will be accessed from multiple GPUs
    multi_gpu,

    /// @brief Read-mostly from device, written from host
    read_mostly
};

// ============================================================================
// CUDA Managed Memory Space
// ============================================================================

/// @brief CUDA unified/managed memory space
/// @details Memory is automatically migrated between host and device
///          by the CUDA runtime. Accessible from both CPU and GPU.
///          Satisfies the MemorySpace concept.
class cuda_managed_memory_space : public memory_space_base {
public:
    /// @brief Memory space identifier
    static constexpr const char* name = "cuda_managed";

    /// @brief Whether this memory is accessible from host
    static constexpr bool host_accessible = true;

    /// @brief Whether this memory is accessible from device
    static constexpr bool device_accessible = true;

    /// @brief Default constructor
    cuda_managed_memory_space() = default;

    /// @brief Construct with memory hint
    /// @param hint Memory placement/migration hint
    explicit cuda_managed_memory_space(managed_memory_hint hint)
        : hint_(hint) {}

    /// @brief Destructor
    ~cuda_managed_memory_space() override = default;

    // Non-copyable
    cuda_managed_memory_space(const cuda_managed_memory_space&) = delete;
    cuda_managed_memory_space& operator=(const cuda_managed_memory_space&) = delete;

    // Movable
    cuda_managed_memory_space(cuda_managed_memory_space&&) = default;
    cuda_managed_memory_space& operator=(cuda_managed_memory_space&&) = default;

    // ------------------------------------------------------------------------
    // Memory Space Interface
    // ------------------------------------------------------------------------

    /// @brief Get memory space name
    [[nodiscard]] const char* space_name() const noexcept override {
        return name;
    }

    /// @brief Allocate managed memory
    /// @param size Number of bytes to allocate
    /// @return Pointer to allocated memory or error
    [[nodiscard]] result<void*> allocate(size_type size) override {
#if DTL_ENABLE_CUDA
        void* ptr = nullptr;

        // Determine flags based on hint
        unsigned int flags = cudaMemAttachGlobal;
        if (hint_ == managed_memory_hint::prefer_host) {
            flags = cudaMemAttachHost;
        }

        cudaError_t err = cudaMallocManaged(&ptr, size, flags);
        if (err != cudaSuccess) {
            return make_error<void*>(status_code::out_of_memory,
                                    "cudaMallocManaged failed");
        }

        // Apply additional hints
        apply_hints(ptr, size);

        total_allocated_ += size;
        if (total_allocated_ > peak_allocated_) {
            peak_allocated_ = total_allocated_;
        }

        return ptr;
#else
        (void)size;
        return make_error<void*>(status_code::not_supported,
                                "CUDA support not enabled");
#endif
    }

    /// @brief Deallocate managed memory
    /// @param ptr Pointer to deallocate
    /// @param size Size of allocation
    /// @return Success or error
    result<void> deallocate(void* ptr, size_type size) override {
#if DTL_ENABLE_CUDA
        if (ptr == nullptr) return {};

        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaFree failed");
        }

        total_allocated_ -= size;
        return {};
#else
        (void)ptr; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Check if pointer is managed memory
    /// @param ptr Pointer to check
    /// @return true if pointer is managed memory
    [[nodiscard]] bool contains(const void* ptr) const noexcept override {
#if DTL_ENABLE_CUDA
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
        if (err != cudaSuccess) {
            cudaGetLastError();  // Clear error
            return false;
        }
        return attrs.type == cudaMemoryTypeManaged;
#else
        (void)ptr;
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // Managed Memory Specific Methods
    // ------------------------------------------------------------------------

    /// @brief Get the current hint setting
    [[nodiscard]] managed_memory_hint hint() const noexcept { return hint_; }

    /// @brief Prefetch memory to a device
    /// @param ptr Pointer to prefetch
    /// @param size Size in bytes
    /// @param device_id Target device (-1 for host)
    /// @return Success or error
    result<void> prefetch(void* ptr, size_type size, int device_id) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemPrefetchAsync(ptr, size, device_id, nullptr);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemPrefetchAsync failed");
        }
        return {};
#else
        (void)ptr; (void)size; (void)device_id;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Prefetch to host
    /// @param ptr Pointer to prefetch
    /// @param size Size in bytes
    /// @return Success or error
    result<void> prefetch_to_host(void* ptr, size_type size) {
#if DTL_ENABLE_CUDA
        return prefetch(ptr, size, cudaCpuDeviceId);
#else
        (void)ptr; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Prefetch to current device
    /// @param ptr Pointer to prefetch
    /// @param size Size in bytes
    /// @return Success or error
    result<void> prefetch_to_device(void* ptr, size_type size) {
#if DTL_ENABLE_CUDA
        int device;
        cudaGetDevice(&device);
        return prefetch(ptr, size, device);
#else
        (void)ptr; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Advise the driver about memory usage patterns
    /// @param ptr Pointer to memory region
    /// @param size Size in bytes
    /// @param advice Memory advice
    /// @param device Target device
    /// @return Success or error
    result<void> advise(void* ptr, size_type size, managed_memory_hint advice, int device) {
#if DTL_ENABLE_CUDA
        cudaMemoryAdvise cuda_advice;
        switch (advice) {
            case managed_memory_hint::prefer_device:
                cuda_advice = cudaMemAdviseSetPreferredLocation;
                break;
            case managed_memory_hint::read_mostly:
                cuda_advice = cudaMemAdviseSetReadMostly;
                break;
            default:
                return {};  // No-op for other hints
        }

        cudaError_t err = cudaMemAdvise(ptr, size, cuda_advice, device);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemAdvise failed");
        }
        return {};
#else
        (void)ptr; (void)size; (void)advice; (void)device;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Get total allocated bytes
    [[nodiscard]] size_type total_allocated() const noexcept {
        return total_allocated_;
    }

    /// @brief Get peak allocated bytes
    [[nodiscard]] size_type peak_allocated() const noexcept {
        return peak_allocated_;
    }

private:
    void apply_hints([[maybe_unused]] void* ptr, [[maybe_unused]] size_type size) {
#if DTL_ENABLE_CUDA
        int device;
        cudaGetDevice(&device);

        switch (hint_) {
            case managed_memory_hint::prefer_device:
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
                break;
            case managed_memory_hint::prefer_host:
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                break;
            case managed_memory_hint::read_mostly:
                cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device);
                break;
            case managed_memory_hint::multi_gpu:
            case managed_memory_hint::none:
                // No specific advice
                break;
        }
#endif
    }

    managed_memory_hint hint_ = managed_memory_hint::none;
    size_type total_allocated_ = 0;
    size_type peak_allocated_ = 0;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the default managed memory space
/// @return Reference to default managed memory space
[[nodiscard]] inline cuda_managed_memory_space& default_managed_memory_space() {
    static cuda_managed_memory_space space;
    return space;
}

/// @brief Create a managed memory space with specific hint
/// @param hint Memory placement hint
/// @return Managed memory space
[[nodiscard]] inline std::unique_ptr<cuda_managed_memory_space>
make_managed_memory_space(managed_memory_hint hint = managed_memory_hint::none) {
    return std::make_unique<cuda_managed_memory_space>(hint);
}

}  // namespace cuda
}  // namespace dtl
