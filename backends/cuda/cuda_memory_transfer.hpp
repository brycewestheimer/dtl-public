// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_memory_transfer.hpp
/// @brief Host-device memory transfers for CUDA
/// @details Provides synchronous and asynchronous memory copy operations
///          between host and device memory.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/memory_transfer.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>

namespace dtl {
namespace cuda {

// Forward declarations
class cuda_stream;
class cuda_event;

// ============================================================================
// Transfer Direction
// ============================================================================

/// @brief Memory transfer direction
enum class transfer_direction {
    /// @brief Host to device
    host_to_device,

    /// @brief Device to host
    device_to_host,

    /// @brief Device to device
    device_to_device,

    /// @brief Host to host (via GPU)
    host_to_host,

    /// @brief Automatic (determined by pointer attributes)
    automatic
};

// ============================================================================
// CUDA Memory Transfer
// ============================================================================

/// @brief Memory transfer operations for CUDA
/// @details Provides copy operations between host and device memory.
///          Satisfies the MemoryTransfer concept.
class cuda_memory_transfer {
public:
    /// @brief Default constructor
    cuda_memory_transfer() = default;

    /// @brief Destructor
    ~cuda_memory_transfer() = default;

    // Non-copyable, non-movable (stateless utility)
    cuda_memory_transfer(const cuda_memory_transfer&) = delete;
    cuda_memory_transfer& operator=(const cuda_memory_transfer&) = delete;
    cuda_memory_transfer(cuda_memory_transfer&&) = delete;
    cuda_memory_transfer& operator=(cuda_memory_transfer&&) = delete;

    // ------------------------------------------------------------------------
    // Synchronous Transfers
    // ------------------------------------------------------------------------

    /// @brief Copy memory from host to device
    /// @param dst Device destination pointer
    /// @param src Host source pointer
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy_host_to_device(void* dst,
                                                           const void* src,
                                                           size_type size) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy H2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Copy memory from device to host
    /// @param dst Host destination pointer
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy_device_to_host(void* dst,
                                                           const void* src,
                                                           size_type size) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy D2H failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Copy memory from device to device
    /// @param dst Device destination pointer
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy_device_to_device(void* dst,
                                                             const void* src,
                                                             size_type size) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy D2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Copy with automatic direction detection
    /// @param dst Destination pointer
    /// @param src Source pointer
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy(void* dst, const void* src, size_type size) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Asynchronous Transfers
    // ------------------------------------------------------------------------

    /// @brief Async copy from host to device
    /// @param dst Device destination pointer
    /// @param src Host source pointer (must be pinned)
    /// @param size Number of bytes
    /// @param stream CUDA stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_host_to_device(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_CUDA
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                          cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpyAsync H2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Async copy from device to host
    /// @param dst Host destination pointer (must be pinned)
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @param stream CUDA stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_device_to_host(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_CUDA
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                          cudaMemcpyDeviceToHost, cuda_stream);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpyAsync D2H failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Async copy from device to device
    /// @param dst Device destination pointer
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @param stream CUDA stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_device_to_device(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_CUDA
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                          cudaMemcpyDeviceToDevice, cuda_stream);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpyAsync D2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Async copy with automatic direction
    /// @param dst Destination pointer
    /// @param src Source pointer
    /// @param size Number of bytes
    /// @param stream CUDA stream
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_CUDA
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                          cudaMemcpyDefault, cuda_stream);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpyAsync failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // 2D/3D Transfers
    // ------------------------------------------------------------------------

    /// @brief Copy 2D memory region
    /// @param dst Destination pointer
    /// @param dst_pitch Destination pitch in bytes
    /// @param src Source pointer
    /// @param src_pitch Source pitch in bytes
    /// @param width Width in bytes
    /// @param height Height in rows
    /// @param direction Transfer direction
    /// @return Success or error
    [[nodiscard]] static result<void> copy_2d(
        void* dst, size_type dst_pitch,
        const void* src, size_type src_pitch,
        size_type width, size_type height,
        transfer_direction direction) {
#if DTL_ENABLE_CUDA
        cudaMemcpyKind kind;
        switch (direction) {
            case transfer_direction::host_to_device:
                kind = cudaMemcpyHostToDevice;
                break;
            case transfer_direction::device_to_host:
                kind = cudaMemcpyDeviceToHost;
                break;
            case transfer_direction::device_to_device:
                kind = cudaMemcpyDeviceToDevice;
                break;
            default:
                kind = cudaMemcpyDefault;
                break;
        }

        cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                       width, height, kind);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy2D failed");
        }
        return {};
#else
        (void)dst; (void)dst_pitch; (void)src; (void)src_pitch;
        (void)width; (void)height; (void)direction;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Pinned Memory Helpers
    // ------------------------------------------------------------------------

    /// @brief Allocate pinned (page-locked) host memory
    /// @param size Number of bytes
    /// @return Pointer or error
    [[nodiscard]] static result<void*> allocate_pinned(size_type size) {
#if DTL_ENABLE_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            return make_error<void*>(status_code::out_of_memory,
                                    "cudaMallocHost failed");
        }
        return ptr;
#else
        (void)size;
        return make_error<void*>(status_code::not_supported,
                                "CUDA support not enabled");
#endif
    }

    /// @brief Free pinned host memory
    /// @param ptr Pointer to free
    /// @return Success or error
    static result<void> free_pinned(void* ptr) {
#if DTL_ENABLE_CUDA
        if (ptr == nullptr) return {};

        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaFreeHost failed");
        }
        return {};
#else
        (void)ptr;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Peer-to-Peer Transfers
    // ------------------------------------------------------------------------

    /// @brief Enable peer access between devices
    /// @param device Device to access from
    /// @param peer_device Device to access
    /// @return Success or error
    [[nodiscard]] static result<void> enable_peer_access(int device, int peer_device) {
#if DTL_ENABLE_CUDA
        // Check if access is possible
        int can_access;
        cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
        if (err != cudaSuccess || !can_access) {
            return make_error<void>(status_code::not_supported,
                                   "Peer access not supported");
        }

        // Enable access
        cudaSetDevice(device);
        err = cudaDeviceEnablePeerAccess(peer_device, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            return make_error<void>(status_code::backend_error,
                                   "cudaDeviceEnablePeerAccess failed");
        }
        return {};
#else
        (void)device; (void)peer_device;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Copy between devices
    /// @param dst Destination pointer (on dst_device)
    /// @param dst_device Destination device
    /// @param src Source pointer (on src_device)
    /// @param src_device Source device
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy_peer(
        void* dst, int dst_device,
        const void* src, int src_device,
        size_type size) {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, size);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpyPeer failed");
        }
        return {};
#else
        (void)dst; (void)dst_device; (void)src; (void)src_device; (void)size;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }
};

// ============================================================================
// Pinned Memory RAII Wrapper
// ============================================================================

/// @brief RAII wrapper for pinned host memory
/// @tparam T Element type
template <typename T>
class pinned_buffer {
public:
    /// @brief Construct empty buffer
    pinned_buffer() = default;

    /// @brief Construct with size
    /// @param count Number of elements
    explicit pinned_buffer(size_type count) : count_(count) {
        auto result = cuda_memory_transfer::allocate_pinned(count * sizeof(T));
        if (result) {
            data_ = static_cast<T*>(result.value());
        }
    }

    /// @brief Destructor
    ~pinned_buffer() {
        if (data_) {
            cuda_memory_transfer::free_pinned(data_);
        }
    }

    // Non-copyable
    pinned_buffer(const pinned_buffer&) = delete;
    pinned_buffer& operator=(const pinned_buffer&) = delete;

    // Movable
    pinned_buffer(pinned_buffer&& other) noexcept
        : data_(other.data_)
        , count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    pinned_buffer& operator=(pinned_buffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cuda_memory_transfer::free_pinned(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    /// @brief Get data pointer
    [[nodiscard]] T* data() noexcept { return data_; }

    /// @brief Get const data pointer
    [[nodiscard]] const T* data() const noexcept { return data_; }

    /// @brief Get element count
    [[nodiscard]] size_type size() const noexcept { return count_; }

    /// @brief Check if valid
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    /// @brief Element access
    T& operator[](size_type i) { return data_[i]; }
    const T& operator[](size_type i) const { return data_[i]; }

private:
    T* data_ = nullptr;
    size_type count_ = 0;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Copy typed array from host to device
/// @tparam T Element type
/// @param dst Device pointer
/// @param src Host pointer
/// @param count Number of elements
template <typename T>
[[nodiscard]] result<void> copy_to_device(T* dst, const T* src, size_type count) {
    return cuda_memory_transfer::copy_host_to_device(
        dst, src, count * sizeof(T));
}

/// @brief Copy typed array from device to host
/// @tparam T Element type
/// @param dst Host pointer
/// @param src Device pointer
/// @param count Number of elements
template <typename T>
[[nodiscard]] result<void> copy_to_host(T* dst, const T* src, size_type count) {
    return cuda_memory_transfer::copy_device_to_host(
        dst, src, count * sizeof(T));
}

}  // namespace cuda
}  // namespace dtl
