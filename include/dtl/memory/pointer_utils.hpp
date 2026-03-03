// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file pointer_utils.hpp
/// @brief Utilities for pointer kind detection and GPU-aware MPI interop
/// @details Provides functions to query whether a pointer refers to host,
///          device, or managed memory, and to detect GPU-aware MPI support.
///          Includes an RAII staging buffer for transparently staging device
///          memory through pinned host buffers when MPI is not GPU-aware.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <string>

namespace dtl {

// ============================================================================
// Pointer Kind Enumeration
// ============================================================================

/// @brief Kind of memory a pointer refers to
enum class pointer_kind {
    host,           ///< Standard host (CPU) memory
    device,         ///< GPU device memory (cudaMalloc)
    managed,        ///< Unified/managed memory (cudaMallocManaged)
    unregistered,   ///< Host memory not registered with CUDA
    unknown         ///< Cannot determine (CUDA not enabled or error)
};

// ============================================================================
// Pointer Kind Query
// ============================================================================

/// @brief Query what kind of memory a pointer refers to
/// @param ptr Pointer to query
/// @return The kind of memory the pointer belongs to
/// @details Uses cudaPointerGetAttributes() when CUDA is enabled.
///          Returns pointer_kind::unknown when CUDA is not enabled.
[[nodiscard]] inline pointer_kind query_pointer_kind(const void* ptr) {
    if (!ptr) {
        return pointer_kind::unknown;
    }

#if DTL_ENABLE_CUDA
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (err != cudaSuccess) {
        // Clear the error state
        cudaGetLastError();
        // On older CUDA versions, unregistered host memory may error
        return pointer_kind::unregistered;
    }

    switch (attrs.type) {
        case cudaMemoryTypeHost:
            return pointer_kind::host;
        case cudaMemoryTypeDevice:
            return pointer_kind::device;
        case cudaMemoryTypeManaged:
            return pointer_kind::managed;
        case cudaMemoryTypeUnregistered:
            return pointer_kind::unregistered;
        default:
            return pointer_kind::unknown;
    }
#else
    (void)ptr;
    return pointer_kind::unknown;
#endif
}

// ============================================================================
// Pointer Accessibility Queries
// ============================================================================

/// @brief Check if a pointer refers to GPU device memory
/// @param ptr Pointer to check
/// @return true if pointer is device or managed memory
[[nodiscard]] inline bool is_device_accessible(const void* ptr) {
    auto kind = query_pointer_kind(ptr);
    return kind == pointer_kind::device || kind == pointer_kind::managed;
}

/// @brief Check if a pointer refers to host-accessible memory
/// @param ptr Pointer to check
/// @return true if pointer is host, managed, or unregistered memory
[[nodiscard]] inline bool is_host_accessible(const void* ptr) {
    auto kind = query_pointer_kind(ptr);
    return kind == pointer_kind::host ||
           kind == pointer_kind::managed ||
           kind == pointer_kind::unregistered ||
           kind == pointer_kind::unknown;
}

// ============================================================================
// GPU-Aware MPI Detection
// ============================================================================

/// @brief Detect if the MPI implementation is GPU-aware (can handle device pointers)
/// @return true if GPU-aware MPI is available
/// @details Detection strategy:
///   1. Check for MPIX_CUDA_AWARE_SUPPORT (Open MPI extension)
///   2. Check for MVAPICH2 GPU support environment variable
///   3. Check for Open MPI runtime CUDA support environment variable
///   4. Default to false (conservative)
/// @note Thread-safe: uses C++11 magic statics for one-time initialization.
[[nodiscard]] inline bool is_gpu_aware_mpi() {
#if DTL_ENABLE_CUDA
    // C++11 guarantees thread-safe initialization of function-local statics
    // (6.7/4). This eliminates the data race of the previous cached/result pair.
    static const bool result = []() -> bool {
        // Method 1: Open MPI CUDA-aware support compile-time check
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (MPIX_CUDA_AWARE_SUPPORT) {
            return true;
        }
    #endif

        // Method 2: Check MVAPICH2 environment variable
        const char* mvapich2_gpu = std::getenv("MV2_USE_CUDA");
        if (mvapich2_gpu && std::string(mvapich2_gpu) == "1") {
            return true;
        }

        // Method 3: Check for Open MPI runtime query
        const char* ompi_cuda = std::getenv("OMPI_MCA_opal_cuda_support");
        if (ompi_cuda && std::string(ompi_cuda) == "true") {
            return true;
        }

        return false;
    }();
    return result;
#else
    return false;
#endif
}

// ============================================================================
// Device Staging Buffer
// ============================================================================

/// @brief Stage buffer through host if needed for non-GPU-aware MPI
/// @details If the pointer is on the device and MPI is not GPU-aware,
///          this class provides RAII staging through a pinned host buffer.
///          If the pointer is already host-accessible or MPI is GPU-aware,
///          no staging occurs.
class device_staging_buffer {
public:
    /// @brief Construct staging buffer for a send or receive operation
    /// @param ptr Source pointer (may be device or host)
    /// @param size_bytes Size of data in bytes
    /// @param for_send true if staging for send (copy device->host on construction),
    ///                 false for recv (copy host->device on destruction)
    device_staging_buffer(const void* ptr, size_t size_bytes, bool for_send)
        : original_ptr_(const_cast<void*>(ptr))
        , size_(size_bytes)
        , for_send_(for_send)
        , needs_staging_(false)
        , staging_ptr_(nullptr)
        , pinned_(false) {
#if DTL_ENABLE_CUDA
        if (is_device_accessible(ptr) && !is_gpu_aware_mpi()) {
            needs_staging_ = true;
            // Allocate pinned host memory for staging
            cudaError_t err = cudaMallocHost(&staging_ptr_, size_bytes);
            if (err != cudaSuccess) {
                // Fallback: use regular host allocation
                staging_ptr_ = std::malloc(size_bytes);
                pinned_ = false;
            } else {
                pinned_ = true;
            }

            if (for_send && staging_ptr_) {
                // Copy device -> host for send
                cudaMemcpy(staging_ptr_, ptr, size_bytes, cudaMemcpyDeviceToHost);
            }
        }
#else
        (void)ptr; (void)size_bytes; (void)for_send;
#endif
    }

    /// @brief Destructor - completes recv staging and cleans up
    ~device_staging_buffer() {
#if DTL_ENABLE_CUDA
        if (staging_ptr_) {
            if (!for_send_) {
                // Copy host -> device for recv completion
                cudaMemcpy(original_ptr_, staging_ptr_, size_, cudaMemcpyHostToDevice);
            }
            if (pinned_) {
                cudaFreeHost(staging_ptr_);
            } else {
                std::free(staging_ptr_);
            }
        }
#endif
    }

    // Non-copyable
    device_staging_buffer(const device_staging_buffer&) = delete;
    device_staging_buffer& operator=(const device_staging_buffer&) = delete;

    // Movable
    device_staging_buffer(device_staging_buffer&& other) noexcept
        : original_ptr_(other.original_ptr_)
        , size_(other.size_)
        , for_send_(other.for_send_)
        , needs_staging_(other.needs_staging_)
        , staging_ptr_(other.staging_ptr_)
        , pinned_(other.pinned_) {
        other.staging_ptr_ = nullptr;
    }

    device_staging_buffer& operator=(device_staging_buffer&&) = delete;

    /// @brief Get the pointer to use for MPI operations
    /// @return Staging buffer if needed, original pointer otherwise
    [[nodiscard]] void* data() noexcept {
        return needs_staging_ ? staging_ptr_ : original_ptr_;
    }

    /// @brief Get the pointer to use for MPI operations (const)
    /// @return Staging buffer if needed, original pointer otherwise
    [[nodiscard]] const void* data() const noexcept {
        return needs_staging_ ? staging_ptr_ : original_ptr_;
    }

    /// @brief Check if staging was needed
    /// @return true if the buffer is staging through host memory
    [[nodiscard]] bool is_staged() const noexcept {
        return needs_staging_;
    }

private:
    void* original_ptr_;
    size_t size_;
    bool for_send_;
    bool needs_staging_;
    void* staging_ptr_;
    bool pinned_;
};

}  // namespace dtl
