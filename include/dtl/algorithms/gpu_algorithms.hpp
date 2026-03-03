// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file gpu_algorithms.hpp
/// @brief Placement-aware GPU algorithm dispatch
/// @details Provides algorithm overloads that dispatch to GPU implementations
///          based on container placement and execution policy.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/device_concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/views/device_view.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/cuda_algorithms.hpp>
#include <dtl/cuda/stream_handle.hpp>
#include <dtl/cuda/device_guard.hpp>
#include <cuda_runtime.h>
#endif

#include <type_traits>
#include <concepts>

namespace dtl {

// ============================================================================
// Algorithm Dispatch Concepts
// ============================================================================

/// @brief Concept for containers with device-only placement
template <typename Container>
concept DeviceOnlyContainer = requires {
    { Container::is_host_accessible() } -> std::convertible_to<bool>;
    { Container::is_device_accessible() } -> std::convertible_to<bool>;
} && !Container::is_host_accessible() && Container::is_device_accessible();

/// @brief Concept for containers with host-accessible placement
template <typename Container>
concept HostAccessibleContainer = requires {
    { Container::is_host_accessible() } -> std::convertible_to<bool>;
} && Container::is_host_accessible();

/// @brief Concept for containers with device-accessible placement
template <typename Container>
concept DeviceAccessibleContainer = requires {
    { Container::is_device_accessible() } -> std::convertible_to<bool>;
} && Container::is_device_accessible();

/// @brief Concept for GPU execution policies
template <typename Policy>
concept GpuExecutionPolicy = requires {
    { Policy::is_device_execution() } -> std::convertible_to<bool>;
} && Policy::is_device_execution();

// ============================================================================
// Dispatch Error Type
// ============================================================================

/// @brief Error code for algorithm dispatch failures
enum class dispatch_error {
    success = 0,
    unsupported_placement,      ///< Placement/execution combination not supported
    host_access_denied,         ///< Cannot run host algorithm on device-only data
    device_access_denied,       ///< Cannot run device algorithm on host-only data
    cuda_not_available,         ///< CUDA required but not enabled
    hip_not_available,          ///< HIP required but not enabled
    type_not_device_storable,   ///< Element type not suitable for device
    internal_error              ///< Unexpected internal error
};

/// @brief Convert dispatch_error to string
[[nodiscard]] inline constexpr const char* to_string(dispatch_error err) noexcept {
    switch (err) {
        case dispatch_error::success: return "success";
        case dispatch_error::unsupported_placement: return "unsupported_placement";
        case dispatch_error::host_access_denied: return "host_access_denied";
        case dispatch_error::device_access_denied: return "device_access_denied";
        case dispatch_error::cuda_not_available: return "cuda_not_available";
        case dispatch_error::hip_not_available: return "hip_not_available";
        case dispatch_error::type_not_device_storable: return "type_not_device_storable";
        case dispatch_error::internal_error: return "internal_error";
    }
    return "unknown";
}

// ============================================================================
// GPU Fill Algorithm
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Fill device memory with a value using Thrust
/// @tparam T Element type
/// @param data Device pointer
/// @param count Number of elements
/// @param value Value to fill with
/// @param stream CUDA stream (default: 0)
/// @param device_id Device to target (default: -1 for current)
/// @return Result indicating success/failure
template <typename T>
    requires DeviceStorable<T>
result<void> fill_device(T* data, size_type count, const T& value,
                          cudaStream_t stream = 0, int device_id = -1) {
    if (count == 0) {
        return result<void>::success();
    }

    try {
        if (device_id >= 0) {
            cuda::device_guard guard(device_id);
            cuda::fill_device(data, count, value, stream);
        } else {
            cuda::fill_device(data, count, value, stream);
        }
        return result<void>::success();
    } catch (...) {
        return result<void>::failure(status{status_code::operation_failed});
    }
}

/// @brief Fill a device view with a value
/// @tparam T Element type
/// @param view Device view to fill
/// @param value Value to fill with
/// @param stream CUDA stream
template <typename T>
    requires DeviceStorable<T>
result<void> fill_device(device_view<T> view, const T& value, cudaStream_t stream = 0) {
    return fill_device(view.data(), view.size(), value, stream, view.device_id());
}

// ============================================================================
// GPU Transform Algorithm
// ============================================================================

/// @brief Transform device memory using a unary operation
/// @tparam T Element type
/// @tparam UnaryOp Unary operation type (must be __device__ callable)
/// @param in Input device pointer
/// @param out Output device pointer
/// @param count Number of elements
/// @param op Unary operation
/// @param stream CUDA stream
/// @param device_id Device to target
template <typename T, typename UnaryOp>
    requires DeviceStorable<T>
result<void> transform_device(const T* in, T* out, size_type count, UnaryOp op,
                               cudaStream_t stream = 0, int device_id = -1) {
    if (count == 0) {
        return result<void>::success();
    }

    try {
        if (device_id >= 0) {
            cuda::device_guard guard(device_id);
            cuda::transform_device(in, out, count, op, stream);
        } else {
            cuda::transform_device(in, out, count, op, stream);
        }
        return result<void>::success();
    } catch (...) {
        return result<void>::failure(status{status_code::operation_failed});
    }
}

/// @brief Transform a device view using a unary operation (in-place)
template <typename T, typename UnaryOp>
    requires DeviceStorable<T>
result<void> transform_device(device_view<T> view, UnaryOp op, cudaStream_t stream = 0) {
    return transform_device(view.data(), view.data(), view.size(), op, stream, view.device_id());
}

// ============================================================================
// GPU Reduce Algorithm
// ============================================================================

/// @brief Reduce device memory using a binary operation
/// @tparam T Element type
/// @tparam BinaryOp Binary operation type (must be __device__ callable)
/// @param data Device pointer
/// @param count Number of elements
/// @param init Initial value
/// @param op Binary reduction operation
/// @param stream CUDA stream
/// @param device_id Device to target
/// @return Result containing the reduction result
template <typename T, typename BinaryOp>
    requires DeviceStorable<T>
result<T> reduce_device(const T* data, size_type count, T init, BinaryOp op,
                         cudaStream_t stream = 0, int device_id = -1) {
    if (count == 0) {
        return result<T>::success_in_place(init);
    }

    try {
        T result_value;
        if (device_id >= 0) {
            cuda::device_guard guard(device_id);
            result_value = cuda::reduce_device(data, count, init, op, stream);
        } else {
            result_value = cuda::reduce_device(data, count, init, op, stream);
        }
        return result<T>::success_in_place(result_value);
    } catch (...) {
        return result<T>::failure(status{status_code::operation_failed});
    }
}

/// @brief Reduce a device view
template <typename T, typename BinaryOp>
    requires DeviceStorable<T>
result<T> reduce_device(const device_view<T>& view, T init, BinaryOp op,
                         cudaStream_t stream = 0) {
    return reduce_device(view.data(), view.size(), init, op, stream, view.device_id());
}

/// @brief Sum reduction on device
template <typename T>
    requires DeviceStorable<T>
result<T> sum_device(const T* data, size_type count, cudaStream_t stream = 0, int device_id = -1) {
    if (count == 0) {
        return result<T>::success_in_place(T{});
    }

    try {
        T result_value;
        if (device_id >= 0) {
            cuda::device_guard guard(device_id);
            result_value = cuda::reduce_sum_device(data, count, stream);
        } else {
            result_value = cuda::reduce_sum_device(data, count, stream);
        }
        return result<T>::success_in_place(result_value);
    } catch (...) {
        return result<T>::failure(status{status_code::operation_failed});
    }
}

/// @brief Sum reduction on device view
template <typename T>
    requires DeviceStorable<T>
result<T> sum_device(const device_view<T>& view, cudaStream_t stream = 0) {
    return sum_device(view.data(), view.size(), stream, view.device_id());
}

// ============================================================================
// GPU Sort Algorithm
// ============================================================================

/// @brief Sort device memory in place
/// @tparam T Element type
/// @param data Device pointer
/// @param count Number of elements
/// @param stream CUDA stream
/// @param device_id Device to target
template <typename T>
    requires DeviceStorable<T>
result<void> sort_device(T* data, size_type count, cudaStream_t stream = 0, int device_id = -1) {
    if (count <= 1) {
        return result<void>::success();
    }

    try {
        if (device_id >= 0) {
            cuda::device_guard guard(device_id);
            cuda::sort_device(data, count, stream);
        } else {
            cuda::sort_device(data, count, stream);
        }
        return result<void>::success();
    } catch (...) {
        return result<void>::failure(status{status_code::operation_failed});
    }
}

/// @brief Sort a device view in place
template <typename T>
    requires DeviceStorable<T>
result<void> sort_device(device_view<T> view, cudaStream_t stream = 0) {
    return sort_device(view.data(), view.size(), stream, view.device_id());
}

#else  // !DTL_ENABLE_CUDA

// Stub implementations when CUDA is not available

template <typename T>
result<void> fill_device(T*, size_type, const T&, int = 0, int = -1) {
    return result<void>::failure(status{status_code::not_supported,
        "fill_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

template <typename T>
result<void> fill_device(device_view<T>, const T&, int = 0) {
    return result<void>::failure(status{status_code::not_supported,
        "fill_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

template <typename T, typename UnaryOp>
result<void> transform_device(const T*, T*, size_type, UnaryOp, int = 0, int = -1) {
    return result<void>::failure(status{status_code::not_supported,
        "transform_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

template <typename T, typename BinaryOp>
result<T> reduce_device(const T*, size_type, T init, BinaryOp, int = 0, int = -1) {
    return result<T>::failure(status{status_code::not_supported,
        "reduce_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

template <typename T>
result<T> sum_device(const T*, size_type, int = 0, int = -1) {
    return result<T>::failure(status{status_code::not_supported,
        "sum_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

template <typename T>
result<void> sort_device(T*, size_type, int = 0, int = -1) {
    return result<void>::failure(status{status_code::not_supported,
        "sort_device requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON"});
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Placement-Aware Algorithm Dispatch
// ============================================================================

/// @brief Static check that prevents host algorithms on device-only containers
/// @details This function template is deleted to produce a compile-time error
///          when attempting to use host algorithms on device-only containers.
template <typename Container, typename... Args>
    requires DeviceOnlyContainer<Container>
void static_host_algorithm_check(const Container&, Args&&...) = delete;

/// @brief Trait to check if a container/policy combination supports host algorithms
template <typename Container, typename ExecutionPolicy>
struct supports_host_algorithm {
    static constexpr bool value =
        Container::is_host_accessible() &&
        !ExecutionPolicy::is_device_execution();
};

template <typename Container, typename ExecutionPolicy>
inline constexpr bool supports_host_algorithm_v =
    supports_host_algorithm<Container, ExecutionPolicy>::value;

/// @brief Trait to check if a container/policy combination supports device algorithms
template <typename Container, typename ExecutionPolicy>
struct supports_device_algorithm {
    static constexpr bool value =
        Container::is_device_accessible() &&
        ExecutionPolicy::is_device_execution();
};

template <typename Container, typename ExecutionPolicy>
inline constexpr bool supports_device_algorithm_v =
    supports_device_algorithm<Container, ExecutionPolicy>::value;

}  // namespace dtl
