// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_buffer.hpp
/// @brief RAII device memory buffer for HIP device-only storage
/// @details Provides a move-only wrapper around HIP device memory that
///          never constructs elements on the host, ensuring safe device-only
///          storage for trivially copyable types.
/// @since 0.1.0
/// @see Phase 21: HIP Backend Parity

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/hip/device_guard.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <utility>
#include <cstring>

namespace dtl {
namespace hip {

/// @brief RAII device buffer for HIP device memory
/// @tparam T Element type (must satisfy DeviceStorable)
/// @details This buffer allocates raw device memory without host-side element
///          construction. It is the foundation for device-only container storage
///          on AMD GPUs.
///
/// @par Key Properties
/// - Move-only semantics (non-copyable)
/// - No host-side element construction or destruction
/// - Device-affinity tracked for multi-GPU support
/// - Supports resize without value initialization
///
/// @par Example
/// @code
/// dtl::hip::device_buffer<float> buf(1024, 0);  // 1024 floats on device 0
/// float* ptr = buf.data();
/// // Use with kernels or rocThrust...
/// buf.resize(2048);  // Grow buffer (no initialization)
/// @endcode
template <typename T>
    requires DeviceStorable<T>
class device_buffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = dtl::size_type;

    // ========================================================================
    // Constructors / Destructor
    // ========================================================================

    /// @brief Default constructor (empty buffer)
    device_buffer() noexcept
        : ptr_(nullptr)
        , size_(0)
        , capacity_(0)
        , device_id_(invalid_device_id) {}

    /// @brief Construct buffer with specified size on device
    /// @param size Number of elements
    /// @param device_id Target HIP device ID
    /// @note Memory is allocated but NOT initialized
    explicit device_buffer(size_type size, int device_id = 0)
        : ptr_(nullptr)
        , size_(0)
        , capacity_(0)
        , device_id_(device_id) {
        if (size > 0) {
            allocate(size);
            size_ = size;
        }
    }

    /// @brief Destructor - frees device memory
    ~device_buffer() noexcept {
        deallocate();
    }

    // Non-copyable
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    /// @brief Move constructor
    device_buffer(device_buffer&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
        , capacity_(other.capacity_)
        , device_id_(other.device_id_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    /// @brief Move assignment
    device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            ptr_ = other.ptr_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            device_id_ = other.device_id_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// @brief Get pointer to device memory
    [[nodiscard]] pointer data() noexcept {
        return ptr_;
    }

    /// @brief Get const pointer to device memory
    [[nodiscard]] const_pointer data() const noexcept {
        return ptr_;
    }

    /// @brief Get number of elements
    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    /// @brief Get allocated capacity (in elements)
    [[nodiscard]] size_type capacity() const noexcept {
        return capacity_;
    }

    /// @brief Check if buffer is empty
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    /// @brief Get the device ID this buffer is allocated on
    [[nodiscard]] int device_id() const noexcept {
        return device_id_;
    }

    /// @brief Get size in bytes
    [[nodiscard]] size_type size_bytes() const noexcept {
        return size_ * sizeof(T);
    }

    // ========================================================================
    // Modifiers
    // ========================================================================

    /// @brief Resize buffer (no initialization)
    /// @param new_size New number of elements
    /// @note If new_size > capacity, reallocates. Does NOT initialize new elements.
    void resize(size_type new_size) {
        if (new_size <= capacity_) {
            size_ = new_size;
            return;
        }

        // Need to reallocate
        pointer new_ptr = nullptr;
#if DTL_ENABLE_HIP
        device_guard guard(device_id_);
        hipError_t err = hipMalloc(&new_ptr, new_size * sizeof(T));
        if (err != hipSuccess) {
            return;  // Allocation failed, size unchanged
        }

        // Copy old data if any
        if (ptr_ != nullptr && size_ > 0) {
            hipMemcpy(new_ptr, ptr_, size_ * sizeof(T), hipMemcpyDeviceToDevice);
            hipFree(ptr_);
        }
#else
        (void)new_size;
        return;
#endif

        ptr_ = new_ptr;
        capacity_ = new_size;
        size_ = new_size;
    }

    /// @brief Reserve capacity without changing size
    /// @param new_capacity Minimum capacity to reserve
    void reserve(size_type new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

#if DTL_ENABLE_HIP
        pointer new_ptr = nullptr;
        device_guard guard(device_id_);
        hipError_t err = hipMalloc(&new_ptr, new_capacity * sizeof(T));
        if (err != hipSuccess) {
            return;
        }

        if (ptr_ != nullptr && size_ > 0) {
            hipMemcpy(new_ptr, ptr_, size_ * sizeof(T), hipMemcpyDeviceToDevice);
            hipFree(ptr_);
        }

        ptr_ = new_ptr;
        capacity_ = new_capacity;
#else
        (void)new_capacity;
#endif
    }

    /// @brief Clear buffer (size = 0, memory retained)
    void clear() noexcept {
        size_ = 0;
    }

    /// @brief Release ownership and return pointer
    /// @return Device pointer (caller takes ownership)
    [[nodiscard]] pointer release() noexcept {
        pointer p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        return p;
    }

    /// @brief Swap with another buffer
    void swap(device_buffer& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(device_id_, other.device_id_);
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// @brief Fill buffer with a value (device memset for byte patterns)
    /// @param value Byte value to fill with (0 recommended for zeroing)
    /// @note Only meaningful for byte patterns; for arbitrary values use thrust::fill
    void memset(int value) {
#if DTL_ENABLE_HIP
        if (ptr_ != nullptr && size_ > 0) {
            device_guard guard(device_id_);
            hipMemset(ptr_, value, size_ * sizeof(T));
        }
#else
        (void)value;
#endif
    }

    /// @brief Async memset on a stream
    void memset_async(int value, hipStream_t stream) {
#if DTL_ENABLE_HIP
        if (ptr_ != nullptr && size_ > 0) {
            device_guard guard(device_id_);
            hipMemsetAsync(ptr_, value, size_ * sizeof(T), stream);
        }
#else
        (void)value;
        (void)stream;
#endif
    }

private:
    /// @brief Allocate device memory
    void allocate(size_type count) {
#if DTL_ENABLE_HIP
        device_guard guard(device_id_);
        hipError_t err = hipMalloc(&ptr_, count * sizeof(T));
        if (err == hipSuccess) {
            capacity_ = count;
        }
#else
        (void)count;
#endif
    }

    /// @brief Deallocate device memory
    void deallocate() noexcept {
#if DTL_ENABLE_HIP
        if (ptr_ != nullptr) {
            device_guard guard(device_id_);
            hipFree(ptr_);
            ptr_ = nullptr;
        }
#endif
        capacity_ = 0;
    }

    pointer ptr_;
    size_type size_;
    size_type capacity_;
    int device_id_;
};

/// @brief Swap two device buffers
template <typename T>
    requires DeviceStorable<T>
void swap(device_buffer<T>& a, device_buffer<T>& b) noexcept {
    a.swap(b);
}

}  // namespace hip
}  // namespace dtl
