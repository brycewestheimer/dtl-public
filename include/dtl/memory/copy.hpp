// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file copy.hpp
/// @brief Host-device memory copy utilities
/// @details Provides explicit copy helpers for moving data between host and
///          device memory. These are the primary mechanism for accessing
///          device-only container data from the host.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/views/local_view.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <dtl/cuda/device_guard.hpp>
#endif

#include <vector>
#include <span>
#include <cstring>
#include <stdexcept>
#include <string>

namespace dtl {

// ============================================================================
// Copy Direction Tags
// ============================================================================

/// @brief Tag for host-to-device copy
struct host_to_device_tag {};

/// @brief Tag for device-to-host copy
struct device_to_host_tag {};

/// @brief Tag for device-to-device copy
struct device_to_device_tag {};

/// @brief Tag instances
inline constexpr host_to_device_tag host_to_device{};
inline constexpr device_to_host_tag device_to_host{};
inline constexpr device_to_device_tag device_to_device{};

// ============================================================================
// Result Type for Copy Operations
// ============================================================================

/// @brief Result of a copy operation
struct copy_result {
    bool success = false;
    size_type bytes_copied = 0;
    int error_code = 0;

    [[nodiscard]] explicit operator bool() const noexcept {
        return success;
    }
};

// ============================================================================
// Synchronous Copy Operations
// ============================================================================

/// @brief Copy device memory to host vector
/// @tparam T Element type (must be DeviceStorable)
/// @param device_data Device pointer to source data
/// @param count Number of elements to copy
/// @param device_id Device ID for guard (optional, -1 for current)
/// @return std::vector<T> with copied data
template <typename T>
    requires DeviceStorable<T>
[[nodiscard]] std::vector<T> copy_to_host(const T* device_data, size_type count,
                                           int device_id = -1) {
    std::vector<T> result(count);
    if (count == 0) {
        return result;
    }

#if DTL_ENABLE_CUDA
    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpy(result.data(), device_data, count * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        err = cudaMemcpy(result.data(), device_data, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_to_host failed: ") + cudaGetErrorString(err));
    }
#else
    // Without CUDA, assume it's host memory
    std::memcpy(result.data(), device_data, count * sizeof(T));
#endif

    return result;
}

/// @brief Copy device view to host vector
/// @tparam T Element type
/// @param view Device view to copy from
/// @return std::vector<T> with copied data
template <typename T>
    requires DeviceStorable<T>
[[nodiscard]] std::vector<T> copy_to_host(const device_view<T>& view) {
    return copy_to_host(view.data(), view.size(), view.device_id());
}

/// @brief Copy host data to device memory
/// @tparam T Element type
/// @param host_data Pointer to host source data
/// @param device_data Pointer to device destination
/// @param count Number of elements
/// @param device_id Device ID for guard
/// @return copy_result indicating success/failure
template <typename T>
    requires DeviceStorable<T>
copy_result copy_from_host(const T* host_data, T* device_data, size_type count,
                           int device_id = -1) {
    if (count == 0) {
        return {.success = true, .bytes_copied = 0};
    }

#if DTL_ENABLE_CUDA
    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpy(device_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    } else {
        err = cudaMemcpy(device_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    }
    return {
        .success = (err == cudaSuccess),
        .bytes_copied = (err == cudaSuccess) ? count * sizeof(T) : 0,
        .error_code = static_cast<int>(err)
    };
#else
    std::memcpy(device_data, host_data, count * sizeof(T));
    return {.success = true, .bytes_copied = count * sizeof(T)};
#endif
}

/// @brief Copy host span to device view
/// @tparam T Element type
/// @param host_span Source span on host
/// @param device_view Destination device view
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_from_host(std::span<const T> host_span, device_view<T> device_view) {
    size_type count = std::min(host_span.size(), static_cast<std::size_t>(device_view.size()));
    return copy_from_host(host_span.data(), device_view.data(), count, device_view.device_id());
}

/// @brief Copy host vector to device view
/// @tparam T Element type
/// @param host_vec Source vector on host
/// @param device_view Destination device view
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_from_host(const std::vector<T>& host_vec, device_view<T> device_view) {
    return copy_from_host(std::span<const T>{host_vec}, device_view);
}

/// @brief Copy device memory to device memory
/// @tparam T Element type
/// @param src_data Source device pointer
/// @param dst_data Destination device pointer
/// @param count Number of elements
/// @param device_id Device ID (for same-device copy)
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_device_to_device(const T* src_data, T* dst_data, size_type count,
                                   int device_id = -1) {
    if (count == 0) {
        return {.success = true, .bytes_copied = 0};
    }

#if DTL_ENABLE_CUDA
    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpy(dst_data, src_data, count * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
        err = cudaMemcpy(dst_data, src_data, count * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    return {
        .success = (err == cudaSuccess),
        .bytes_copied = (err == cudaSuccess) ? count * sizeof(T) : 0,
        .error_code = static_cast<int>(err)
    };
#else
    std::memcpy(dst_data, src_data, count * sizeof(T));
    return {.success = true, .bytes_copied = count * sizeof(T)};
#endif
}

// ============================================================================
// Asynchronous Copy Operations
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Async copy device memory to host
/// @tparam T Element type
/// @param device_data Source device pointer
/// @param host_data Destination host pointer (must be pinned for true async)
/// @param count Number of elements
/// @param stream CUDA stream for async operation
/// @param device_id Device ID for guard
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_to_host_async(const T* device_data, T* host_data, size_type count,
                                cudaStream_t stream, int device_id = -1) {
    if (count == 0) {
        return {.success = true, .bytes_copied = 0};
    }

    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpyAsync(host_data, device_data, count * sizeof(T),
                              cudaMemcpyDeviceToHost, stream);
    } else {
        err = cudaMemcpyAsync(host_data, device_data, count * sizeof(T),
                              cudaMemcpyDeviceToHost, stream);
    }
    return {
        .success = (err == cudaSuccess),
        .bytes_copied = (err == cudaSuccess) ? count * sizeof(T) : 0,
        .error_code = static_cast<int>(err)
    };
}

/// @brief Async copy host memory to device
/// @tparam T Element type
/// @param host_data Source host pointer (should be pinned for true async)
/// @param device_data Destination device pointer
/// @param count Number of elements
/// @param stream CUDA stream for async operation
/// @param device_id Device ID for guard
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_from_host_async(const T* host_data, T* device_data, size_type count,
                                  cudaStream_t stream, int device_id = -1) {
    if (count == 0) {
        return {.success = true, .bytes_copied = 0};
    }

    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpyAsync(device_data, host_data, count * sizeof(T),
                              cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpyAsync(device_data, host_data, count * sizeof(T),
                              cudaMemcpyHostToDevice, stream);
    }
    return {
        .success = (err == cudaSuccess),
        .bytes_copied = (err == cudaSuccess) ? count * sizeof(T) : 0,
        .error_code = static_cast<int>(err)
    };
}

/// @brief Async device-to-device copy
/// @tparam T Element type
/// @param src_data Source device pointer
/// @param dst_data Destination device pointer
/// @param count Number of elements
/// @param stream CUDA stream
/// @param device_id Device ID
/// @return copy_result
template <typename T>
    requires DeviceStorable<T>
copy_result copy_device_to_device_async(const T* src_data, T* dst_data, size_type count,
                                         cudaStream_t stream, int device_id = -1) {
    if (count == 0) {
        return {.success = true, .bytes_copied = 0};
    }

    cudaError_t err;
    if (device_id >= 0) {
        cuda::device_guard guard(device_id);
        err = cudaMemcpyAsync(dst_data, src_data, count * sizeof(T),
                              cudaMemcpyDeviceToDevice, stream);
    } else {
        err = cudaMemcpyAsync(dst_data, src_data, count * sizeof(T),
                              cudaMemcpyDeviceToDevice, stream);
    }
    return {
        .success = (err == cudaSuccess),
        .bytes_copied = (err == cudaSuccess) ? count * sizeof(T) : 0,
        .error_code = static_cast<int>(err)
    };
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Container-Level Copy Helpers
// ============================================================================

/// @brief Copy a container's local partition to host
/// @tparam Container Container type with device_view() method
/// @param container Container to copy from
/// @return std::vector with local data
template <typename Container>
    requires requires(const Container& c) {
        { c.device_view() } -> std::same_as<device_view<typename Container::value_type>>;
    }
[[nodiscard]] auto copy_local_to_host(const Container& container) {
    return copy_to_host(container.device_view());
}

/// @brief Copy host data to container's local partition
/// @tparam Container Container type with device_view() method
/// @tparam T Element type
/// @param container Container to copy to
/// @param host_data Source data on host
/// @return copy_result
template <typename Container, typename T>
    requires requires(Container& c) {
        { c.device_view() } -> std::same_as<device_view<typename Container::value_type>>;
    }
copy_result copy_local_from_host(Container& container, std::span<const T> host_data) {
    return copy_from_host(host_data, container.device_view());
}

// ============================================================================
// Synchronization Helpers
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Synchronize a CUDA stream
/// @param stream Stream to synchronize
/// @return true on success
inline bool stream_synchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

/// @brief Synchronize current device
/// @return true on success
inline bool device_synchronize() {
    return cudaDeviceSynchronize() == cudaSuccess;
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl
