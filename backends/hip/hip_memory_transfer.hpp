// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hip_memory_transfer.hpp
/// @brief Host-device memory transfers for HIP
/// @details Provides synchronous and asynchronous memory copy operations
///          between host and device memory using HIP runtime.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/memory_transfer.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory>

namespace dtl {
namespace hip {

// Forward declarations
class hip_stream;
class hip_event;

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
// HIP Memory Transfer
// ============================================================================

/// @brief Memory transfer operations for HIP
/// @details Provides copy operations between host and device memory.
///          Satisfies the MemoryTransfer concept.
class hip_memory_transfer {
public:
    /// @brief Default constructor
    hip_memory_transfer() = default;

    /// @brief Destructor
    ~hip_memory_transfer() = default;

    // Non-copyable, non-movable (stateless utility)
    hip_memory_transfer(const hip_memory_transfer&) = delete;
    hip_memory_transfer& operator=(const hip_memory_transfer&) = delete;
    hip_memory_transfer(hip_memory_transfer&&) = delete;
    hip_memory_transfer& operator=(hip_memory_transfer&&) = delete;

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
#if DTL_ENABLE_HIP
        hipError_t err = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpy H2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
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
#if DTL_ENABLE_HIP
        hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpy D2H failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
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
#if DTL_ENABLE_HIP
        hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpy D2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Copy with automatic direction detection
    /// @param dst Destination pointer
    /// @param src Source pointer
    /// @param size Number of bytes
    /// @return Success or error
    [[nodiscard]] static result<void> copy(void* dst, const void* src, size_type size) {
#if DTL_ENABLE_HIP
        hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDefault);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpy failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Asynchronous Transfers
    // ------------------------------------------------------------------------

    /// @brief Async copy from host to device
    /// @param dst Device destination pointer
    /// @param src Host source pointer (must be pinned)
    /// @param size Number of bytes
    /// @param stream HIP stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_host_to_device(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_HIP
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipMemcpyAsync(dst, src, size,
                                        hipMemcpyHostToDevice, hip_stream);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpyAsync H2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Async copy from device to host
    /// @param dst Host destination pointer (must be pinned)
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @param stream HIP stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_device_to_host(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_HIP
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipMemcpyAsync(dst, src, size,
                                        hipMemcpyDeviceToHost, hip_stream);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpyAsync D2H failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Async copy from device to device
    /// @param dst Device destination pointer
    /// @param src Device source pointer
    /// @param size Number of bytes
    /// @param stream HIP stream (nullptr for default)
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy_device_to_device(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_HIP
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipMemcpyAsync(dst, src, size,
                                        hipMemcpyDeviceToDevice, hip_stream);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpyAsync D2D failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Async copy with automatic direction
    /// @param dst Destination pointer
    /// @param src Source pointer
    /// @param size Number of bytes
    /// @param stream HIP stream
    /// @return Success or error
    [[nodiscard]] static result<void> async_copy(
        void* dst, const void* src, size_type size, void* stream = nullptr) {
#if DTL_ENABLE_HIP
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipMemcpyAsync(dst, src, size,
                                        hipMemcpyDefault, hip_stream);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpyAsync failed");
        }
        return {};
#else
        (void)dst; (void)src; (void)size; (void)stream;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // 2D Transfers
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
#if DTL_ENABLE_HIP
        hipMemcpyKind kind;
        switch (direction) {
            case transfer_direction::host_to_device:
                kind = hipMemcpyHostToDevice;
                break;
            case transfer_direction::device_to_host:
                kind = hipMemcpyDeviceToHost;
                break;
            case transfer_direction::device_to_device:
                kind = hipMemcpyDeviceToDevice;
                break;
            default:
                kind = hipMemcpyDefault;
                break;
        }

        hipError_t err = hipMemcpy2D(dst, dst_pitch, src, src_pitch,
                                     width, height, kind);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpy2D failed");
        }
        return {};
#else
        (void)dst; (void)dst_pitch; (void)src; (void)src_pitch;
        (void)width; (void)height; (void)direction;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Pinned Memory Helpers
    // ------------------------------------------------------------------------

    /// @brief Allocate pinned (page-locked) host memory
    /// @param size Number of bytes
    /// @return Pointer or error
    [[nodiscard]] static result<void*> allocate_pinned(size_type size) {
#if DTL_ENABLE_HIP
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, size);
        if (err != hipSuccess) {
            return make_error<void*>(status_code::out_of_memory,
                                    "hipHostMalloc failed");
        }
        return ptr;
#else
        (void)size;
        return make_error<void*>(status_code::not_supported,
                                "HIP support not enabled");
#endif
    }

    /// @brief Free pinned host memory
    /// @param ptr Pointer to free
    /// @return Success or error
    static result<void> free_pinned(void* ptr) {
#if DTL_ENABLE_HIP
        if (ptr == nullptr) return {};

        hipError_t err = hipHostFree(ptr);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipHostFree failed");
        }
        return {};
#else
        (void)ptr;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
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
#if DTL_ENABLE_HIP
        // Check if access is possible
        int can_access;
        hipError_t err = hipDeviceCanAccessPeer(&can_access, device, peer_device);
        if (err != hipSuccess || !can_access) {
            return make_error<void>(status_code::not_supported,
                                   "Peer access not supported");
        }

        // Enable access
        hipSetDevice(device);
        err = hipDeviceEnablePeerAccess(peer_device, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
            return make_error<void>(status_code::backend_error,
                                   "hipDeviceEnablePeerAccess failed");
        }
        return {};
#else
        (void)device; (void)peer_device;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
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
#if DTL_ENABLE_HIP
        hipError_t err = hipMemcpyPeer(dst, dst_device, src, src_device, size);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipMemcpyPeer failed");
        }
        return {};
#else
        (void)dst; (void)dst_device; (void)src; (void)src_device; (void)size;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
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
        auto result = hip_memory_transfer::allocate_pinned(count * sizeof(T));
        if (result) {
            data_ = static_cast<T*>(result.value());
        }
    }

    /// @brief Destructor
    ~pinned_buffer() {
        if (data_) {
            hip_memory_transfer::free_pinned(data_);
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
                hip_memory_transfer::free_pinned(data_);
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
    return hip_memory_transfer::copy_host_to_device(
        dst, src, count * sizeof(T));
}

/// @brief Copy typed array from device to host
/// @tparam T Element type
/// @param dst Host pointer
/// @param src Device pointer
/// @param count Number of elements
template <typename T>
[[nodiscard]] result<void> copy_to_host(T* dst, const T* src, size_type count) {
    return hip_memory_transfer::copy_device_to_host(
        dst, src, count * sizeof(T));
}

}  // namespace hip
}  // namespace dtl
